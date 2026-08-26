# Day 20 | 磁鐵與牆：Inductor Fusion 的邊界

## 前言

昨天看了 Scheduler 怎麼把 lowering 產出的 buffer 排成一列 node、算好依賴，然後一輪一輪嘗試把 node 兩兩配對合併。昨天回答的是「誰跟誰有機會配對」，今天要回答的是「憑什麼可以配對」。Fusion 省下的是中間結果寫出記憶體再讀回來的流量，是 Inductor 最重要的一筆收益，但它不能亂融。有些組合像磁鐵一樣一碰就黏住，有些地方則是一面怎麼推都推不動的牆。今天就沿著 v2.8.0 [`scheduler.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/scheduler.py) 的 `can_fuse` 把這條邊界一段一段摸出來。本篇實驗全部跑在 CPU 上（torch 2.8.0），和 GPU 行為不同的地方會特別標註。

正文開始！

## 兩種融法，一個入口

先講清楚為什麼值得融。pointwise 這類 op 是典型的 memory-bound 運算，算一次加法的時間遠遠小於等一次記憶體，瓶頸從頭到尾都在資料搬運上。eager mode 每個 op 各自是一顆 kernel，每一步都把整份中間結果寫出記憶體，下一步再原封不動讀回來，資料在主記憶體和運算單元之間白跑好幾趟。把幾個 op 融進同一個 kernel，中間值留在暫存器裡直接交棒，讀寫次數從每 op 一輪變成頭尾各一次，這就是 fusion 的全部動機。

昨天點過名的兩種融法，今天先把差別講清楚。**垂直融合**融的是 producer 和 consumer，下游 node 讀的正是上游 node 剛寫出的 buffer，融在一起之後，值直接在暫存器裡交棒，中間那顆 buffer 從此不存在。**水平融合**融的是兄弟，兩個 node 誰也不吃誰的輸出，但讀同一份 input，而且 loop 範圍相同，併進同一個 kernel 之後，input 只需要從記憶體讀一次。

判斷的入口是 `Scheduler.can_fuse`。它先擋掉昨天照過面的那幾種硬性不合格，extern kernel 不融、device 要相同、融了不能把依賴圖繞成 cycle。接著估這一對共享了多少讀寫。最後分流，兩個 node 之間有依賴就走 `can_fuse_vertical`，沒有就走 `can_fuse_horizontal`，並且各自再問 backend 一次，因為 loop 縫不縫得起來只有負責 codegen 的人知道。

配對本身也有誘因門檻。融合的收益全部來自省下的記憶體流量，所以配對階段只把有共享資料的 node 湊成候選，一對兄弟要是什麼都不共享，連上場的資格都沒有。昨天看過的 `fuse_nodes` 外圈就拿著這些判準一輪一輪融到收斂為止，今天要做的就是把每一關的判準攤開來看。

## 用對照組把邊界畫出來

方法很單純，準備幾組對照函式，開 `TORCH_LOGS="fusion,output_code"` 實際編譯。`fusion` 這個 artifact 會把每一輪的候選名單、成功的配對和失敗的理由全部印出來，`output_code` 則拿到最後生成的 kernel。先看磁鐵這一側的三組。

```python
def chain(x):
    return torch.sin(torch.relu(x + 1) * 2)

def rowsum(x):
    return torch.relu(x + 1).sum(dim=1)

def siblings(x):
    return x.amax(dim=1), x.sum(dim=1)
```

`chain` 是純 pointwise 鏈，log 開場就只有一個 node，四個 op 全擠在同一個 `Pointwise` 裡。

    SchedulerNode(name='op0'), Pointwise(['[1024, 1024]', 'origins=OrderedSet([sin, mul, relu, add])'])
    found 0 possible fusions

有趣的是這裡連 Scheduler 都沒出手。只有單一使用者的 pointwise 在 lowering 階段就被直接 inline 進下游了，Scheduler 負責融的是 lowering 吸收不掉的那些。`rowsum` 也一樣只剩一個 node，不過型別變成了 `Reduction`，`origins` 裡帶著 `sum_1, relu, add`，pointwise 被垂直融進了 reduction 的 loop 裡面。`siblings` 則是水平融合的標準案例，`amax` 和 `sum` 互不相欠，但都要把 `x` 從頭到尾掃一遍。

    SchedulerNode(name='op0'), Reduction(['[1024]', 'max', 'origins=OrderedSet([amax])'])
    SchedulerNode(name='op1'), Reduction(['[1024]', 'sum', 'origins=OrderedSet([sum_1])'])
    found 1 possible fusions
    fusing op0 with op1

兩顆 reduction 併成一顆 kernel，`x` 只讀一次就同時算出兩個統計量。反過來做一組對照，讓兩個分支各自操作兩個不相干的 tensor，shape 完全相同、loop 範圍也完全相同，log 卻變成 `found 0 possible fusions`。不是不能融，是沒有理由融，沒有共享資料就沒有可以省的流量，融了只是把兩段互不相干的 loop 塞在一起，Scheduler 連試都不試。這一組對照把水平融合的本質講得很白，它省的不是 kernel launch 的次數，是同一份資料的重複讀取。

## reduction 是天然的牆

現在來撞牆。pointwise 的 loop 是「每個元素獨立跑一遍」，reduction 的 loop 是「外圈走輸出、內圈把一整段收成一個值」，兩種 loop 結構天生不同。Scheduler 幫每個 node 記了一組 `(numel, rnumel)` 描述它的 loop 長相，前者是平行維度的大小，後者是收縮維度的大小。拿 `rowsum` 來說，`relu(x + 1)` 攤平是 `(1048576, 1)`，逐 row 的 `sum` 是 `(1024, 1024)`，兩邊相乘對得起來，pointwise 就能順著資料流融進 reduction 的內圈，一邊掃一邊加。但方向反過來，reduction 的輸出再接 pointwise，shape 已經塌掉了，麻煩就來了。

```python
def wall(x):
    return torch.relu(x - x.mean())
```

`mean` 把 1024x1024 收成一個 scalar，而下游任何一個輸出元素都要等整份 input 掃完才能動筆。這不是啟發式的取捨，是依賴上的死結，log 也乾脆。

    SchedulerNode(name='op0'), Reduction(['[1024, 1024]', 'sum', 'origins=OrderedSet([mean])'])
    SchedulerNode(name='op1'), Pointwise(['[1024, 1024]', 'origins=OrderedSet([relu, sub, mean])'])
    found 0 possible fusions

這裡有個 CPU 特有的判讀陷阱。`output_code` 印出來只有一個 `cpp_fused_mean_relu_sub_0` 函式，但打開一看裡面是兩個獨立的 loop nest，`x` 從記憶體讀了兩次。cpp backend 會把沒融合的 node 打包進同一個 C++ 函式，所以在 CPU 上函式數量不等於融合結果，牆的痕跡要看 loop nest。換到 GPU 上，這就是實打實的兩次 kernel launch。

不過牆不是碰到 reduction 就無條件成立。把 `wall` 換成 softmax，同樣有 reduction 又有後續 pointwise，結局完全不同。

    cannot fuse op1 with op3: intermediate nodes between node1 & node2
    found 1 possible fusions
    fusing op1 with op2
    completed fusion round (1/10): fused 4 nodes into 3 nodes
    ...
    fusing op0 with op1_op2
    fusing op0_op1_op2 with op3
    completed fusion round (2/10): fused 3 nodes into 1 nodes

`amax`、`exp` 加 `sum`、`div` 開場四個 node，兩輪之內融成一個。差別在粒度，softmax 的 reduction 是逐 row 的，每個 row 的 `div` 只依賴自己這一 row 的統計量，外圈同樣是 1024 個 row，cpp backend 讓幾段 loop 共用同一個外圈，每個 row 的資料進了 cache 就一路算完。Triton backend 的版本在 [`codegen/simd.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/simd.py) 的 `can_fuse`，判準寫得很直白，pointwise 要融進 reduction 得滿足 `numel1 == numel2 * rnumel2`，兩顆 reduction 要融則 `(numel, rnumel)` 得完全相等。所以牆的本質不是 reduction 這個 op，而是它把 loop 結構切成了對不上的兩半。全域收縮把牆砌死，逐 row 收縮則留了一扇門。

![op 方塊沿資料流互相磁吸，撞上 reduction 的牆之後另起一顆 kernel](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day20/fusion.gif)

*圖一：七個 op 的資料流。前段 pointwise 鏈像磁鐵一樣一路吸進 `mean` 的 reduction loop，牆後的 `sub`、`relu` 等不到全域統計量出爐，只能另起一顆 kernel，最後收在 7 ops、2 kernels。*

動畫裡那個函式也真的跑了一遍，就是 log 裡的 case H。開場只有兩個 node，`sin`、`mul`、`relu`、`add` 這條鏈在 lowering 時就被吸進了 `mean` 的 `Reduction`，牆後的 `Pointwise` 自成一國，`found 0 possible fusions`。裡面還藏了一個彩蛋，牆後那個 node 的 `origins` 又出現了一整條 `sin, mul, relu, add`。`y` 被兩邊用到，Inductor 沒有把它留成 buffer，而是選擇在第二顆 kernel 裡重算一次，用一點計算換掉一整輪 1024x1024 的讀寫，這個取捨跟 recompute 的邏輯一脈相承。

## 牆的其他幾種長相

reduction 之外還有幾面牆。第一面是 extern kernel。matmul、convolution 這類 op 調的是外部程式庫或預先寫好的 template，Inductor 手上根本沒有它的 loop 可以縫。拿 `relu((x + 1) @ w)` 一試，`mm` 被包成一個呼叫外部 kernel 的節點進了候選名單，前後兩段 pointwise 只能各自成家。

    call sequence: ['cpp_fused_add_0', 'extern_kernels.mm', 'cpp_fused_relu_1']

想把 matmul 前後的 pointwise 也吃進來，得換一條路，讓 Inductor 用 Triton template 自己生 matmul，再把 pointwise 當 epilogue 縫上去，`can_fuse` 開頭有幾個專門為 template 開的分支，處理的就是這件事，這裡先點到為止。

第二面是 layout。共享同一個 buffer 不代表共享了資料，兩個 node 要是用不同的 stride 去讀同一塊記憶體，例如一個順著讀、一個轉置著讀，索引式對不上，省流量就無從談起，`can_fuse` 會留下 `no shared data due to indexing mismatch` 的紀錄。Triton 那邊還會再檢查一次 tiling，兩個 node 各自選好的分塊方式拼不起來，一樣作罷。

第三面牆是 Inductor 自己砌的。融合不是越多越好，一顆 kernel 同時算的東西越多，活著的中間值就越多，而它們全都佔著暫存器，融過頭 register 溢出到慢速記憶體，省下的頻寬全數賠回去。所以 [`choices.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/choices.py) 裡有幾條明著踩剎車的規則，單顆 kernel 融進的 node 數超過上限直接喊停，融合會推高 peak memory 的擋下，兩個 node 在原圖裡離得太遠的也擋下，免得中間值的生命週期被拉得老長。另外 `can_fuse` 回 True 也只是及格，同一輪裡誰先融，還是由昨天看過的 `score_fusion` 依省下的流量和距離排序。

## 從 output_code 判讀融合結果

最後整理一份實務小抄。kernel 的名字本身就是融合報告，`cpp_fused_add_relu_sum_0` 或 GPU 上的 `triton_per_fused_...`，名字裡的 op 清單就是被融進來的 origins，一個名字對應一顆 kernel。想數 kernel 就看 wrapper 裡的 call 順序，每一行呼叫就是一次 launch，中間插著的 `extern_kernels.*` 是編譯器管不到的地界。而想知道「為什麼沒融」，開 `TORCH_LOGS="fusion"`，每一次失敗的理由都會印給你，從 loop 對不上到嫌你們離太遠都有。唯一要記得的是 CPU 的 cpp backend 會把多個沒融合的 node 包進同一個函式，數函式會數錯，得看 loop nest 或直接看 fusion log 的收斂結果。

## 結語

Fusion 的邊界今天算是走完一圈。垂直融合讓 producer 和 consumer 在暫存器交棒，水平融合讓兄弟共享一次讀取，入口都是 `can_fuse`，誘因都是省下的記憶體流量。牆有四種，loop 結構對不上的 reduction、縫不進去的 extern kernel、索引對不上的 layout，以及 Inductor 為了 register pressure 自己踩的剎車。判讀結果就看 kernel 名字、call 順序和 fusion log。

node 該怎麼配對已經定案，下一步就是把每顆 fused node 真的寫成 GPU 程式碼。明天來看 Triton Codegen，看 Inductor 怎麼把一個 node 的 loop 變成一份 tl.load、tl.store 的 Triton kernel。那我們明天見！

## 參考資料

- [torch/_inductor/scheduler.py：can_fuse 與 fuse_nodes（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/scheduler.py)
- [torch/_inductor/choices.py：InductorChoices 的融合啟發式（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/choices.py)
- [torch/_inductor/codegen/simd.py：SIMDScheduling.can_fuse（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/simd.py)
- [torch/_inductor/codegen/cpp.py：CppScheduling（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/cpp.py)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）
