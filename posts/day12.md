# Day 12 | torch.compile 的沙盤推演師：AOTAutograd

## 前言

Dynamo 的故事在昨天正式收尾，今天進入 pipeline 的第二站。先把問題攤開來。Dynamo 交出來的 FX Graph 只有 forward，而訓練的每一步都要 backward。Eager 模式下這不是問題，autograd 引擎在執行時動態建 tape、動態回放。但我們現在要的是編譯，backward 也得變成一張可以最佳化的圖，而且要在真正執行之前就拿到。這就是 AOTAutograd 名字裡 Ahead-of-Time 的意思。

如果要幫它取一個系列裡的角色，我會說它是一位沙盤推演師。軍隊真的開拔之前，先在沙盤上把整場仗完整推演一遍，進攻的路線（forward）畫出來，撤退的路線（backward）也一併畫好，沿途哪些補給要先囤在哪個據點（要保存哪些中間值）全部標記清楚。而且沙盤上用的是模型不是真兵，AOTAutograd 用的也不是真資料，是只有 shape、dtype、device 的 FakeTensor。推演完，兩張地圖各自交給 Inductor 去修路。

它住在 [`torch/_functorch/aot_autograd.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/aot_autograd.py)，是 Dynamo 和 Inductor 之間的中間層。今天先看清楚它的輸入輸出長什麼樣、中間那場推演是怎麼跑的，細節的三層轉換（Functionalization、Decomposition、Partitioner）留給後面幾天各自拆開。

正文開始！

## Dynamo 交給 AOTAutograd 什麼

整個系列的實驗一樣跑在 Modal 的 L40S、PyTorch 2.8.0 上，拿一個最小的訓練形狀。

```python
def f(x, w):
    return (x @ w).relu().sum()

x = torch.randn(4, 4, device="cuda")
w = torch.randn(4, 4, device="cuda", requires_grad=True)
```

`TORCH_LOGS="graph_code"` 可以印出 Dynamo 收工時交出來的東西。

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4][4, 1]cuda:0", L_w_: "f32[4, 4][4, 1]cuda:0"):
        l_x_ = L_x_
        l_w_ = L_w_
        # File: aot_overview.py:14 in f, code: return (x @ w).relu().sum()
        matmul: "f32[4, 4][4, 1]cuda:0" = l_x_ @ l_w_;  l_x_ = l_w_ = None
        relu: "f32[4, 4][4, 1]cuda:0" = matmul.relu();  matmul = None
        sum_1: "f32[][]cuda:0" = relu.sum();  relu = None
        return (sum_1,)
```

這張圖有兩個特徵。第一，它是 torch 層的。`l_x_ @ l_w_`、`.relu()`、`.sum()`，你寫什麼它就是什麼，連原始碼行號都帶著，這是給人看的層級。第二，它只有 forward。三行運算、一個回傳，`w` 的梯度要從哪裡來，圖上一個字都沒提。

交接的介面也很單純。Dynamo 的 `OutputGraph` 收完圖之後，把 GraphModule 和 example inputs 一起交給 backend 函式，而 `aot_eager`、`inductor` 這些 backend 內部都繞經同一個入口 [`aot_module_simplified`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/aot_autograd.py)。它收 `fw_compiler`、`bw_compiler`、`partition_fn` 三個關鍵的 callback，意思是「展開完之後，forward 圖交給誰編、backward 圖交給誰編、以及兩張圖之間那一刀怎麼切」。Day 2 的 `backend="aot_eager"` 就是把前兩個 callback 換成「什麼都不做、直接 eager 跑」的版本，所以能單獨觀察 AOTAutograd 這一段。

## 實際跑一步訓練

把 `aot_graphs` 打開。

```python
torch._logging.set_logs(aot_graphs=True)
out = torch.compile(f)(x, w)
out.backward()
print("w.grad shape:", w.grad.shape)
```

log 裡印出了兩張圖。先看 Forward。

```python
 ===== Forward graph 1 =====
def forward(self, primals_1: "f32[4, 4][4, 1]cuda:0", primals_2: "f32[4, 4][4, 1]cuda:0"):
    mm: "f32[4, 4][4, 1]cuda:0" = torch.ops.aten.mm.default(primals_1, primals_2);  primals_2 = None
    relu: "f32[4, 4][4, 1]cuda:0" = torch.ops.aten.relu.default(mm);  mm = None
    sum_1: "f32[][]cuda:0" = torch.ops.aten.sum.default(relu)
    le: "b8[4, 4][4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    permute: "f32[4, 4][1, 4]cuda:0" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    return (sum_1, le, permute)
```

再看 Backward。

```python
 ===== Backward graph 1 =====
def forward(self, le: "b8[4, 4][4, 1]cuda:0", permute: "f32[4, 4][1, 4]cuda:0", tangents_1: "f32[][]cuda:0"):
    expand: "f32[4, 4][0, 0]cuda:0" = torch.ops.aten.expand.default(tangents_1, [4, 4]);  tangents_1 = None
    full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, ...)
    where: "f32[4, 4][4, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, expand);  le = full_default = expand = None
    mm_1: "f32[4, 4][4, 1]cuda:0" = torch.ops.aten.mm.default(permute, where);  permute = where = None
    return (None, mm_1)
```

最後一行輸出是 `w.grad shape: torch.Size([4, 4])`，梯度真的算出來了。這兩張圖藏了不少東西，一件一件挑出來講。

**Forward 的輸出不只 loss。** 使用者只要 `sum_1`，但圖還回傳了 `le`（relu 的 mask）和 `permute`（`x` 的轉置）。這些是 backward 需要的中間值，被「存」下來從 forward 傳給 backward。仔細看 `le` 的型別是 `b8`，一個 bool tensor，每個元素只佔 1 byte。backward 算 relu 的梯度其實只需要「哪些位置小於等於零」這個資訊，存一張 f32 的 `relu` 輸出（每個元素 4 bytes）就浪費了。存哪些、存多少、存成什麼形式，是一個真正的取捨，Day 15 的 partitioner 專門管這件事。

**Backward 也是一張普通的 FX Graph。** 輸入是存下來的 `le`、`permute`，加上 `tangents_1`（上游梯度，這裡 loss 對自己的梯度是純量 1，所以是 `f32[]`）。輸出對齊 forward 的輸入順序，`x` 不需要梯度所以第一格是 `None`，`w` 的梯度是 `mm_1`。因為它就是一張普通的 ATen 圖，它跟 forward 一樣交給 Inductor 編譯，backward 也享受 fusion，訓練加速的一半來自這裡。

**relu 的 backward 不是查表得來的。** 圖裡是 `le` 加 `where`，小於等於 0 的位置梯度歸零，其他位置放行上游梯度。AOTAutograd 沒有維護一份「每個 op 的導數公式」，它是讓 PyTorch 本來的 autograd 引擎在 FakeTensor 上真的跑出 backward，再把整個過程 trace 下來。所以微分規則來自 PyTorch 原生的 derivatives 定義，trace 的過程中又被進一步拆解成 `le`、`where` 這種更基本的運算。

**backward 的第一個 op 是 `expand`。** 它把純量的 `tangents_1` 撐成 `[4, 4]`，這就是 `sum` 的導數，每個元素對 loss 的貢獻都是 1，梯度就是上游梯度廣播到整個 shape。你平常呼叫 `loss.backward()` 時 autograd 引擎默默做的事，在這裡全部變成了圖上可以指著看的節點。

## no_grad 底下只剩一張圖

同一個函式，包在 `torch.no_grad()` 裡編一次。

```python
with torch.no_grad():
    torch.compile(f)(x, w)
```

`aot_graphs` 這次只印一張圖。

```python
 ===== Forward graph 0 =====
def forward(self, arg0_1: "f32[4, 4][4, 1]cuda:0", arg1_1: "f32[4, 4][4, 1]cuda:0"):
    mm: "f32[4, 4][4, 1]cuda:0" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    relu: "f32[4, 4][4, 1]cuda:0" = torch.ops.aten.relu.default(mm);  mm = None
    sum_1: "f32[][]cuda:0" = torch.ops.aten.sum.default(relu);  relu = None
    return (sum_1,)
```

沒有 `le`、沒有 `permute`，輸出只有 `(sum_1,)`。連輸入的名字都不一樣，訓練那張叫 `primals_1`，這張叫 `arg0_1`，因為推論路徑根本不需要區分「哪些是原始輸入、哪些是之後要對齊梯度的」。log 裡每張圖前面還附了一份 `fw_metadata`（`ViewAndMutationMeta`），訓練那份寫著 `is_train=True`、`traced_tangents=[FakeTensor(..., size=())]`，推論這份是 `is_train=False`、`traced_tangents=[]`，AOTAutograd 在 trace 之前就已經把兩種情境分析清楚了。

要不要展開 backward，是看輸入的 `requires_grad` 和當下的 grad mode 決定的。Day 6 埋的一個伏筆也在這裡回收。`GLOBAL_STATE grad_mode` 那條 Guard 之所以存在，就是因為開著 grad 編出來的圖和 `no_grad` 底下編的根本是兩個東西，grad mode 一變，整套編譯產物都得換。

## AOTAutograd 是怎麼 trace 的

現在回頭講機制。AOTAutograd 拿到 Dynamo 的 torch 層圖之後，做的事可以拆成三步。

第一步，**用 FakeTensor 把 forward 重新演一遍**。它按照圖上的節點逐一執行，但餵進去的是 FakeTensor，只有 shape、dtype、device，沒有資料。每個 op 都會經過 PyTorch 的 dispatcher 走完整的分發流程，所以 view、in-place、broadcasting 這些語意全部照真實規則走，只是最後沒有人真的去碰記憶體。

第二步，**讓 autograd 引擎在推演上把 backward 展開**。FakeTensor 一樣會被 autograd 引擎記 tape，AOTAutograd 對推演的輸出呼叫反向傳播，引擎按 tape 回放，`sum` 的導數展開成 `expand`、`relu` 的導數展開成 `le` 加 `where`、`mm` 的導數展開成 `permute` 加另一個 `mm`。這整段回放同樣被 trace 下來，於是 forward 和 backward 的所有運算都落在同一張圖上，這張圖叫 **joint graph**。

第三步，**把 joint graph 切成兩半**。`partition_fn` 在 joint graph 上決定哪些節點屬於 forward、哪些屬於 backward，而橫跨切口的值（這裡是 `le` 和 `permute`）就變成 forward 的額外輸出、backward 的輸入。切在哪裡不是唯一解。多存一點，backward 就少算一點。少存一點，記憶體就省一點但 backward 得重算。預設的 min-cut 演算法怎麼在這兩者之間找平衡，是 Day 15 的主題。

把三步連起來看一遍。

![一張 torch 層圖被推演展開成 joint graph，再切成 forward 與 backward 兩張 ATen 圖](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day12/aot_two_graphs.gif)

*圖一：AOTAutograd 的完整流程。左邊是 Dynamo 交出的 torch 層 forward 圖。中間 AOTAutograd 拿 FakeTensor 重演一遍收成 joint graph 的前半，autograd 引擎接著把 backward 逐節點展開補上後半。partitioner 在 joint graph 上切一刀，`le`、`permute` 被搬到 forward 側存下來。最後分裂成右邊 Forward 與 Backward 兩張 ATen 層的圖，中間那條 saved 箭頭就是兩張圖的臍帶。*

## 圖也從 torch 層降到 ATen 層

比較一下 Dynamo 圖和 AOT 圖裡同一行程式的長相。

```
Dynamo:  matmul = l_x_ @ l_w_          （torch 層，使用者寫什麼就是什麼）
AOT:     mm = torch.ops.aten.mm.default(primals_1, primals_2)   （ATen 層）
```

這個轉換是第一步推演的副產品。FakeTensor 重跑 forward 時，每個 op 都經過 dispatcher，落到 ATen 層的正式名字。`@` 這個 Python 運算子分發下去就是 `aten.mm`，`.relu()` 就是 `aten.relu`，變數名也從帶著來源資訊的 `l_x_` 變成編號的 `primals_1`。torch 層是給人看的，ATen 層是給編譯器看的，後端只要理解 ATen 這一套詞彙就夠了。

這場推演還連帶做了兩件事，各佔一天。in-place 和 view 被改寫成純函數式（Functionalization，明天），高階 op 被拆成更小的基本運算（Decomposition，後天）。所以 Inductor 拿到的圖，跟你寫的 Python 已經隔了好幾層轉換，但每一層都留著 log 可以對照，這也是這個系列一路 dump 中間產物的底氣。

## 圖編好了，backward 誰來呼叫？

還剩最後一塊拼圖。兩張圖各自被 Inductor 編成 kernel 之後，執行期的 `out.backward()` 是怎麼知道要去跑那張編好的 backward 圖的？

答案是 AOTAutograd 並沒有取代執行期的 autograd 引擎，它是把自己「掛」了上去。編譯的產物最後被包進一個 `torch.autograd.Function`（實作在 [`runtime_wrappers.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/runtime_wrappers.py) 裡的 `CompiledFunction`）。它的 `forward` 呼叫編譯好的 forward kernel，拿到 `(sum_1, le, permute)`，把 `le`、`permute` 用 `save_for_backward` 存進 autograd 的 context，只把 `sum_1` 還給使用者。它的 `backward` 把存的值取出來，連同上游梯度一起餵給編譯好的 backward kernel。

所以從 eager 世界看過去，整個編譯區段就是 tape 上的**一個**節點，裡面幾十個運算的微分細節全部被折疊掉了，autograd 引擎只知道「這個節點有一個對應的 backward，到時候呼叫它就對了」。這個設計順便解決了混搭的問題。模型裡有 Graph Break 時，每個編譯區段各自是一個 `CompiledFunction` 節點，區段之間沒被編譯的 eager 運算照常記 tape，backward 時引擎按順序回放，走到編譯節點就跑編譯的 kernel，走到 eager 節點就走原本的路，兩個世界無縫接在同一條 tape 上。

## 為什麼非得 ahead-of-time？

最後把「為什麼要這麼大費周章」講清楚。Eager 的 autograd 是 define-by-run，forward 跑到哪、tape 記到哪，backward 按 tape 回放。好處是彈性，代價是 backward 永遠是一連串分開的 kernel，沒有全局可看、沒有 fusion 可做。AOT 展開之後就不一樣了。

- backward 是一張完整的圖，Inductor 能融合它。訓練的計算量大約一半在 backward，這一半在 eager 時代是完全編譯不到的。
- forward 和 backward 之間「存什麼」從 autograd 引擎的預設行為（存所有需要的中間值），變成一個可以全局規劃的決策。上面那個「存 b8 的 mask 而不是 f32 的輸出」就是最小的例子，省記憶體的 activation checkpointing，在這個框架下也變成 partitioner 的一個策略而已。
- 展開發生在編譯期，用的是 FakeTensor，不碰真資料、不佔 GPU 記憶體，成本一次付清。之後每一步訓練都直接跑編好的兩張圖，不再有 trace 的開銷。

## 結語

AOTAutograd 是 pipeline 第二站的沙盤推演師。拿 Dynamo 交出的 torch 層 forward 圖，在 FakeTensor 上重演一遍、讓 autograd 引擎把撤退路線也畫出來，收成一張 joint graph，再一刀切成 ATen 層的 forward 和 backward 兩張圖。forward 多輸出一批要保存的中間值，backward 拿著它們和上游梯度算出對輸入的梯度，兩張各自交給 Inductor，最後包進一個 `autograd.Function` 掛回 eager 的 tape 上。微分不是它自己算的，是引擎跑一遍、它錄下來的。圖也不是它編的，是切好之後轉交的。它的本事全在「推演」和「切分」這兩件事上。

明天拆推演過程中的第一層轉換，也就是 Functionalization。`x.add_(1)`、view 這些會就地改記憶體的操作，是怎麼被改寫成純函數式、又保證語意不變的。那我們明天見！

## 參考資料

- [torch/_functorch/aot_autograd.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/aot_autograd.py)
- [torch/_functorch/_aot_autograd/runtime_wrappers.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/runtime_wrappers.py)
- [torch/_functorch/partitioners.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/partitioners.py)
- [functorch 的 aot_autograd 文件](https://pytorch.org/functorch/stable/aot_autograd.html)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 4 節）
