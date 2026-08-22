# Day 15 | backward 該存誰、該重算誰？min-cut 分家公證人

## 前言

還記得 Day 12 留下的懸案嗎？那時候我們看到 forward 圖除了 loss 之外，還多輸出了 `le` 和 `permute` 這兩個中間值給 backward 用。當時只說「存哪些、存多少，是一個真正的取捨」，然後就把問題晾在那裡了。今天就來把這個洞補起來。

答案的形狀其實蠻有趣。AOTAutograd 並不是先做好 forward 圖、再想辦法配一張 backward 圖，而是一開始就把兩者 trace 成**同一張圖**，叫做 joint graph。你可以把它想成一份夫妻共同財產的清冊，forward 和 backward 的所有運算都列在上面，中間值就是共同持有的家當。而 Partitioner 就是那位精打細算的分家公證人，它拿著清冊決定哪些節點歸 forward、哪些歸 backward，至於被切在分界線上的家當，就是 forward 結束時必須保存、留給 backward 用的 activation。

這一刀落在哪，直接決定了訓練時 GPU 記憶體的大頭要花在哪裡，也是速度與記憶體之間最重要的一顆旋鈕。正文開始！

## 先看那張沒切開的圖

`TORCH_LOGS="aot_joint_graph"` 可以看到切割前的原貌。拿一個最小的訓練形狀 `f(x, w) = tanh(x @ w).sum()` 實跑。

```python
def f(x, w):
    return torch.tanh(x @ w).sum()

x = torch.randn(64, 64, device="cuda")
w = torch.randn(64, 64, device="cuda", requires_grad=True)

torch._logging.set_logs(aot_joint_graph=True, aot_graphs=True)
out = torch.compile(f)(x, w)
out.backward()
```

印出來的 joint graph 長成下面這樣（為了版面，省略了幾個 `alias` 簿記節點）。

```python
 ===== Joint graph 0 =====
def forward(self, primals, tangents):
    mm = torch.ops.aten.mm.default(primals_1, primals_2)
    tanh = torch.ops.aten.tanh.default(mm)
    sum_1 = torch.ops.aten.sum.default(tanh)          # <- forward 的輸出
    expand = torch.ops.aten.expand.default(tangents_1, [64, 64])
    mul = torch.ops.aten.mul.Tensor(tanh, tanh)       # tanh 的導數 1 - tanh^2
    sub = torch.ops.aten.sub.Tensor(1, mul)
    mul_1 = torch.ops.aten.mul.Tensor(expand, sub)
    permute = torch.ops.aten.permute.default(primals_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(permute, mul_1)  # <- w 的梯度
    return pytree.tree_unflatten([sum_1, None, mm_1], self._out_spec)
```

這張圖的輸入同時有 `primals`（forward 的輸入）和 `tangents`（上游梯度），輸出同時有 loss 和梯度。前三行是 forward，後面六行是 backward，兩段之間沒有任何邊界標記，純粹靠資料流相連，backward 用到了 `tanh`（tanh 的導數是 `1 - tanh^2`，算導數要用它自己的輸出）和 `primals_1`（`w` 的梯度是 `x^T @ grad`，所以要轉置 `x`）。

順帶一提，`primals` 和 `tangents` 這兩個名字來自微分幾何的術語，functorch 的 JVP 世界觀就是這樣稱呼「原始輸入」和「方向導數」的，AOTAutograd 從 functorch 一路繼承了這套詞彙。

## 為什麼要先合成一張 joint graph

Day 12 講過 AOTAutograd 的展開手法。拿 FakeTensor 把 forward 重新執行一遍，讓 autograd 引擎在上面跑出 backward，再把整個過程 trace 下來。更精確地說，它會先把你的函式包成一個 joint function，大概是這個形狀。

```python
def joint(primals, tangents):
    outs = f(*primals)
    grads = torch.autograd.grad(outs, inputs, grad_outputs=tangents)
    return outs, grads
```

然後對這個 joint function 做一次 trace，得到的就是上面那張 joint graph。forward 和 backward 天生就在同一張圖上，這不是實作上的偷懶，而是刻意的設計。**只有兩邊都在手上，「存什麼、重算什麼」才是一個可以全局規劃的問題**。如果 backward 是事後才配出來的，forward 早就把「要保存哪些值」寫死了，公證人根本無從介入。

而 eager 模式的 autograd 引擎正是後者，它在 forward 執行時把「backward 會用到的值」全部存進 tape，存哪些是每個 op 的 derivative 公式各自決定的，沒有任何全局視野。joint graph 把這個固定行為變成了一道可以最佳化的圖論題。

## 存，還是重算？

在看切割結果之前，得先弄清楚這場分家在爭什麼。訓練時 GPU 記憶體的大頭通常不是權重，而是 activation，也就是每一層 forward 算出來的中間值，都得一路留到 backward 用完才能釋放。模型越深、batch 越大、序列越長，這筆帳就越可觀。

但「存」從來不是唯一的選項。任何一個中間值，backward 要用的時候其實有兩條路。

- **存下來**：forward 多輸出一個值，佔一份記憶體，從 forward 結束一路佔到 backward 用完。
- **重算**：forward 不存，backward 開頭拿更上游的值把它重新算一遍，付一次計算的代價。

切線往 forward 靠是多存，記憶體漲、速度快。往 backward 靠是多算，記憶體省、速度慢。這兩個極端之間的每一個中間點都是合法解，而 Partitioner（[`torch/_functorch/partitioners.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/partitioners.py)）的工作就是在這條光譜上挑一個好位置。它有兩個策略。`default_partition` 模仿 eager autograd 的行為，backward 用到什麼就存什麼。預設真正上場的則是 `min_cut_rematerialization_partition`，也就是今天的主角。

## 實際看 partitioner 怎麼切

同一份 log 往下捲，`aot_graphs` 印出切完的兩張圖。

```python
 ===== Forward graph 0 =====
def forward(self, primals_1, primals_2):
    mm = torch.ops.aten.mm.default(primals_1, primals_2)
    tanh = torch.ops.aten.tanh.default(mm)
    sum_1 = torch.ops.aten.sum.default(tanh);  tanh = None
    permute = torch.ops.aten.permute.default(primals_1, [1, 0])
    return (sum_1, mm, permute)              # 保存 mm 和 permute

 ===== Backward graph 0 =====
def forward(self, mm, permute, tangents_1):
    expand = torch.ops.aten.expand.default(tangents_1, [64, 64])
    tanh = torch.ops.aten.tanh.default(mm);  mm = None   # <- 重算 tanh！
    mul = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
    sub = torch.ops.aten.sub.Tensor(1, mul)
    mul_1 = torch.ops.aten.mul.Tensor(expand, sub)
    mm_1 = torch.ops.aten.mm.default(permute, mul_1)
    return (None, mm_1)
```

切完的兩張圖藏著幾個玄機，逐條對下去。

**第一，backward 開頭多了一行 joint graph 裡沒有的 `tanh = tanh(mm)`。** joint graph 裡 backward 直接引用 forward 算好的 `tanh`，但切完之後保存的是 `mm`，backward 拿到 `mm` 自己又算了一次 `tanh`。這就是重算（rematerialization），公證人判定這個值「讓 backward 自己做一份」比「存著帶過去」划算。

**第二，為什麼存 `mm` 不存 `tanh`？** 兩個都是 `f32[64, 64]`，一樣是 16 KB，跨線傳輸的成本相同，單看 min-cut 兩種切法同分。但 partitioner 的整體偏好是把切線往輸入方向推、讓便宜的 pointwise op 留給 backward 重算。`tanh` 重算幾乎免費，而且這行重算會被 Inductor 直接融合進 backward 的第一個 kernel，實際代價趨近於零。反過來 `mm` 是矩陣乘法，計算密集，partitioner 直接把它列為禁止重算，它的輸出天生就是切線的好落點。

**第三，`permute` 根本不佔記憶體。** 它只改 stride 這種 metadata，不碰資料，「存」它等於免費，所以留在 forward 直接傳過去，backward 連轉置都不用做。

**第四，Day 12 的懸案順便破了。** 那時的例子是 `relu`，forward 存的是 `le`（relu 的 mask）而不是 `relu` 的輸出。現在可以讀懂了。`le` 的 dtype 是 `b8`，一個元素一個 byte，只有 `f32` 的四分之一，公證人挑了最便宜的那件家當過戶。存什麼從來不是「backward 公式寫了什麼就存什麼」，而是成本算出來的。

## 切在哪裡是一道 min-cut 問題

那「跨線傳輸的量最小」這件事是怎麼算的？這就是 min-cut 這個名字的來源。`min_cut_rematerialization_partition` 把問題建模成經典的最大流最小割。

- 把 joint graph 攤開成一張流網路，源點那一側接著 forward 的輸入，匯點那一側接著 backward 真正要消耗的節點。
- 每個節點拆成 in、out 兩半，中間那條邊的容量就是「保存這個值要花的記憶體」，用 `_size_of()` 按 `numel * dtype 大小` 算出 bytes。
- 不可重算的節點，容量設成無限大，逼切線繞開它們。哪些不可重算？`mm`、`conv` 這類計算密集的 op（重算太貴）、隨機 op（重算一次結果就不一樣了）都在名單上，view 類的 metadata 操作則幾乎零成本。除此之外還有一批啟發式，例如重算鏈拉得太長、或是被重算的值離 backward 太遠時也會被禁止，避免「省記憶體」反過來變成新的計算瓶頸。
- 對這張網路跑一次最大流（實作直接呼叫 networkx 的 `minimum_cut`），得到的最小割就是答案，割開的那些邊，對應的值就是要保存的 activation。

所以「forward 圖該輸出什麼」這個看起來很工程的問題，最後是用一條 1956 年的 max-flow min-cut 定理解掉的，個人覺得相當浪漫。整個流程視覺化出來就是下面這樣。

![min-cut partitioner 在 joint graph 上比較兩種切法後選擇存 mm，節點各自歸隊成 forward 與 backward 兩張圖，最後把旋鈕轉到底變成 checkpoint](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day15/min_cut.gif)

*圖一：分家公證人的完整流程。上半是 `tanh(x @ w).sum()` 的 joint graph，forward 與 backward 靠資料流相連。先標出 backward 需要的值與各條邊的保存成本，一刀落在 `tanh` 右側（存 tanh）與左側（存 mm）同為 16 KB，但 pointwise 可以免費重算，於是刀往輸入方向推、`tanh` 複製一份歸隊到 backward。接著節點各自歸隊成 FORWARD 與 BACKWARD 兩張圖，中間跨線的 `mm` 與 `permute` 就是保存值。最後把旋鈕轉到底，checkpoint 只存 `primals`，backward 開頭重播整段 forward。*

## 轉到底就是 activation checkpointing

min-cut 是自動找的折衷點，但這顆旋鈕也可以手動轉到底。把同一個函式包進 `torch.utils.checkpoint`。

```python
import torch.utils.checkpoint as cp

def g(x, w):
    return cp.checkpoint(lambda a: torch.tanh(a @ w).sum(), x, use_reentrant=False)

out = torch.compile(g)(x, w)
out.backward()
```

`aot_graphs` 印出來的兩張圖變成下面這樣。

```python
 ===== Forward graph 1 =====
def forward(self, primals_1, primals_2):
    mm = torch.ops.aten.mm.default(primals_1, primals_2)
    tanh = torch.ops.aten.tanh.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(tanh);  tanh = None
    return (sum_1, primals_1, primals_2)     # 只存輸入！

 ===== Backward graph 1 =====
def forward(self, primals_1, primals_2, tangents_1):
    expand = torch.ops.aten.expand.default(tangents_1, [64, 64])
    mm = torch.ops.aten.mm.default(primals_1, primals_2)   # 整段 forward
    tanh = torch.ops.aten.tanh.default(mm);  mm = None     #    重算一遍
    mul = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
    sub = torch.ops.aten.sub.Tensor(1, mul)
    mul_1 = torch.ops.aten.mul.Tensor(expand, sub)
    permute = torch.ops.aten.permute.default(primals_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(permute, mul_1)
    return (None, mm_1)
```

Forward 一個中間值都不存，只把原始輸入原封不動傳過去。backward 開頭把 `mm`、`tanh` 整段重算，連本來禁止重算的 `mm` 都重算了，因為這是使用者明確要求的。記憶體從「存 activation」變成「存輸入」，代價是 backward 多付一次 forward 的計算量。

大模型訓練裡人人都在用的 activation checkpointing，在編譯棧裡就是這麼做出來的。它不是什麼獨立的魔法機制，只是 partitioner 收到指示，把切線推到最極端的位置而已。min-cut 和 checkpoint 是同一個問題的兩個解，一個由成本模型自動找，一個由你手動指定。

其實兩個極端之間還有一段可以微調的空間。`torch._functorch.config.activation_memory_budget` 是一個 0 到 1 之間的旋鈕，1 是預設的 min-cut 行為，0 等於整段 checkpoint，中間值則會讓 partitioner 在給定的記憶體預算內，用背包問題的解法挑出最划算的一組重算對象。旋鈕這個比喻不是修辭，它真的是一顆連續的旋鈕。

## 這一刀省下多少記憶體

最後把鏡頭拉遠一點。這一刀之所以是編譯式訓練的關鍵設計，有兩個層次的原因。

第一層是前面說的記憶體。activation 是訓練記憶體的大頭，而 partitioner 把「存什麼」從 autograd 引擎的固定行為，變成一個帶成本模型的全局最佳化問題。同樣一張卡，切得好就能塞下更大的 batch 或更長的序列。

第二層更隱微。**重算在編譯世界裡比在 eager 世界裡便宜得多**。eager 下做 activation checkpointing，重算就是實打實地再跑一遍那些 kernel，每個都要 launch、都要讀寫記憶體。但在這裡，backward 也是 Inductor 要編譯的一張完整的圖，重算出來的 pointwise op 往往直接融進 backward 本來就要跑的 kernel 裡，多算一個 `tanh` 只是暫存器裡多一條指令，記憶體流量一點都沒多。Day 2 算過 elementwise 的瓶頸是記憶體頻寬不是計算，所以這種重算的邊際成本趨近於零。這讓 min-cut partitioner 敢於激進地選擇重算，也是 `torch.compile` 訓練加速裡一塊很實在的來源，不只是每個 kernel 變快，而是整個「存與算」的帳本都被重新算過一遍。

## 結語

到今天，AOTAutograd 這一站的全貌就完整了。FakeTensor 重跑 forward、autograd 引擎展開 backward，合成一張 joint graph（Day 12）。Functionalization 去掉 mutation 和 aliasing（Day 13）。Decomposition 把高階 op 拆成基本運算（Day 14）。最後由 min-cut partitioner 這位分家公證人一刀切開，切線上的值就是要保存的 activation，便宜的 pointwise 傾向重算，`mm` 這類貴重家當禁止重算，checkpoint 則是把刀推到極端、只存輸入。兩張乾淨的 ATen 圖，一前一後交給下一站。

明天就進入第三站 TorchInductor 了。它拿到圖之後的第一步不是生程式碼，而是把每個 ATen op 翻譯成它自己的中間表示，也就是一種「用 Python 函式描述的迴圈」。搞懂這層 IR，後面的 fusion 和 codegen 才讀得懂。那我們明天見！

## 參考資料

- [torch/_functorch/partitioners.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/partitioners.py)
- [torch/_functorch/config.py（activation_memory_budget）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/config.py)
- [Min-cut optimal recomputation with AOTAutograd（PyTorch dev-discuss）](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)
- [torch.utils.checkpoint 文件](https://pytorch.org/docs/stable/checkpoint.html)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 4 節）
