# Day 19 | 誰跟誰可以同桌？TorchInductor 的宴席總管 Scheduler

## 前言

昨天 Lowering 把 FX Graph 翻成了 loop-level IR，每個 node 都知道「自己的第 i 格怎麼算」。但 IR 就緒後還缺兩個答案，這串 node 誰先誰後執行，以及誰跟誰可以生成同一個 kernel。這決定了最後生出幾個 kernel、每個 kernel 搬多少記憶體，也就是 Inductor 快不快的關鍵。今天的主角 `Scheduler` 負責回答這兩題，原始碼在 [`torch/_inductor/scheduler.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/scheduler.py)。

今天的比喻是宴席總管。IR node 是賓客，kernel 是桌子。總管先弄清楚賓客的關係，排出入座順序，再盡量把互動最密切的賓客排在同一桌，因為隔桌傳話的成本比同桌講話高出一個數量級。誰跟誰同桌，就是 fusion 的決定。

正文開始！

## fusion 省的是搬運，不是計算

把兩個 kernel 融成一個，計算一次也沒有少做，省下的是中間結果在記憶體之間的往返。

一個 pointwise kernel 的一生很單純，從記憶體把 input 讀進來，算幾個算術指令，把結果寫回去。對現代硬體來說中間那步幾乎免費，一秒能做的浮點運算次數，比能搬進來的浮點數多了一到兩個數量級，真正貴的是頭尾兩步。如果 `sin` 和 `add` 各自是一個 kernel，`sin` 的結果得先寫回記憶體再由 `add` 讀回來，以 1024x1024 的 float32 Tensor 來說，這一趟來回就是 8MB 流量，換到的只是一個加法。融成同一個 kernel 之後，中間值活在暫存器裡，那 8MB 直接消失。

深度學習的圖裡絕大多數 op 都是這種搬得多、算得少的體質，這也是 fusion 成為 Inductor 主要加速來源的原因。打分邏輯自然就長成了「這次融合能省下幾個 byte 的讀寫」，而不是省下幾條指令。有了這個視角，Scheduler 的每條規則都會變得好懂。

## 先包成 SchedulerNode，再把依賴接起來

`Scheduler` 收到 lowering 完的一串 IR node，第一件事是把每個 node 包成 `SchedulerNode`，這是排程與融合的基本單位。不能融合的另外包，像 matmul 這種呼叫外部 kernel 的、以及什麼都不做的空節點，各自有專門的殼。

第二件事是把 node 之間的依賴邊接出來。規則只有一條，你寫的 buffer 我讀，我就依賴你。這裡的依賴不是 FX Graph 那種「值的流向」，而是重新用 buffer 的讀寫算出來的，因為經過 lowering 的 inline 之後，誰真的碰了哪塊記憶體，只有 loop-level IR 自己知道。每條依賴記成一個 `MemoryDep`，除了 buffer 名字，還記著「用什麼 index 式、讀多大範圍」，這細節等一下會決定融合的生死。有了邊就能排出合法的拓撲順序，再幫每個 node 算出祖先集合，之後判斷「A 融 B 會不會繞出 cycle」靠的就是這份名單。一切就緒，`fuse_nodes` 開始配對。

## 拿四個小函式實測

以下實驗在 CPU 上跑（torch 2.8.0），Inductor 走 C++ 後端，但排程與融合的決策所有後端共用同一個 `Scheduler`。以 `TORCH_LOGS="fusion,ir_pre_fusion"` 印出決策過程，完整程式在 `code/day19/`。

第一個對照組是三步 pointwise。

```python
def chain(x):
    return torch.relu(torch.sin(x) + 1)
```

fusion log 印出來卻只有一位候選人（節錄）。

```
===== attempting fusion (1/10): 1 nodes =====
  SchedulerNode(name='op0'), Pointwise(['[1024, 1024]', 'origins=OrderedSet([relu, add, sin])'])
found 0 possible fusions
```

三個 op 只有一個 node，`origins` 裡 `relu, add, sin` 全擠在一起。這其實是昨天就發生的事，中間值只有一個使用者的 pointwise，lowering 時直接被 inline 進消費者的 loop body，輪不到 Scheduler 出手，kernel 只有一個 `cpp_fused_add_relu_sin_0`。要看 Scheduler 真的做決定，得讓中間結果「不得不」寫進記憶體。

最簡單的辦法就是插一個 reduction。

```python
def epilogue(x):
    z = torch.sin(x) + 1
    s = z.sum(dim=1)
    return torch.relu(s) * 2
```

`ir_pre_fusion` 這次印出兩個 node，依賴邊看得一清二楚（節錄）。

```
op0.writes = [MemoryDep('buf0', c0, {c0: 1024})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 1024})]
op1.writes = [MemoryDep('buf1', c0, {c0: 1024})]
```

`op0` 是那個 sum（`sin`、`add` 照樣被 inline 進 reduction 裡），寫出 1024 格的 `buf0`。`op1` 是後面的 `relu` 和乘法，`unmet_dependencies` 指著 `buf0`，這條邊就是從 buffer 名字長出來的。Scheduler 怎麼處理這對候選人？

```
===== attempting fusion (1/10): 2 nodes =====
op0 and op1 has 4096 shared data
found 1 possible fusions
fusing op0 with op1
completed fusion round (1/10): fused 2 nodes into 1 nodes
```

那個 4096 就是打分的核心，1024 格 float32 恰好 4096 bytes，也就是「融合能省下的流量」。算法是把兩個 node 的讀寫集合取交集，共同的 `MemoryDep` 大小加起來，`buf0` 一個寫一個讀正好對上。融合成立，kernel 只剩一個 `cpp_fused_add_mul_relu_sin_sum_0`，外層迴圈每算完一列的 sum，順手把 `relu` 和乘法做掉，`buf0` 那趟往返就省下來了。

流程用動畫走一遍。

![IR node 之間長出依賴邊，pointwise 被吸進同一桌，reduction 擋下融合](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day19/scheduler.gif)

*圖一：宴席總管的工作流程。先把 IR node 包成 SchedulerNode 並依讀寫關係接出依賴邊，再把共享資料最多的兩個 node 融成一桌，中間的 buffer 與它的 store、load 一起消失，記憶體流量跟著往下掉。reduction 出現時 iteration group 對不上，融合被擋下，kernel 邊界就留在那裡。*

## 一道 reduction 就是一道牆

上面的 reduction 跟後面的 pointwise 融在一起了，但它更常見的角色是牆。把 sum 換成全域的，讓後面每一格都依賴它。

```python
def wall(x):
    y = torch.sin(x)
    s = y.sum()
    return torch.relu(y + s)
```

`relu(y + s)` 的每一格都需要 sum 的最終值，reduction 跑完之前 pointwise 連第一格都不能開算。log 也是這麼說的（節錄）。

```
op0.group.iteration = ((), (1048576,))
op1.group.iteration = ((1048576,), ())
op0 and op1 has 4194308 shared data
found 0 possible fusions
```

這對候選人共享 4194308 bytes 的資料，比剛才的 4096 大了三個數量級，卻融不成。原因在 `group.iteration`，`op0` 要先走完 1048576 步 reduction 才吐出一個值，`op1` 的 1048576 步每一步都要用那個值，兩個迴圈形狀對不上，語意上就不可能，分數再高也沒用。產物是兩個前後接續的 loop nest（log 裡 `loops: 2`），中間隔著一次完整的等待，這就是 kernel 邊界。在 GPU 上這道牆就是兩次 kernel launch 加一次全域同步，所有 thread 都得等 reduction 的最後一個值落地。

這個實驗還藏了一個彩蛋。`sin` 同時出現在 `op0` 和 `op1` 的 `origins` 裡，而 allocs 清單只有一個 scalar 的 `buf0` 和輸出 `buf1`。`y` 被兩邊用到，Inductor 卻沒把它存成 4MB 的 buffer，而是在兩個 loop 裡各算一次 `sin`。重算比多搬 8MB 便宜，這跟 min-cut Partitioner 是同一個世界觀，計算便宜，記憶體貴。

還有一種更隱晦的失敗，兩個 node 讀同一個 input，但一個順著讀、一個轉置著讀。

```python
def mismatch(x):
    return torch.sin(x), torch.cos(x.t()).contiguous()
```

```
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 1048576})]
op1.met_dependencies = [MemoryDep('arg0_1', c0 + 1024*c1, {c0: 1024, c1: 1024})]
op0 and op1 has 0 shared data
cannot fuse op0 with op1: no shared data
```

同一個 `arg0_1`，兩條 `MemoryDep` 的 index 式卻不一樣，一個 `c0`、一個 `c0 + 1024*c1`。「共享資料」的定義是用同一個 index 式讀寫同一個 buffer，不是名字一樣就算。index 對不上，融合之後每一次 load 照舊得做，score 是 0，Scheduler 直接放棄。

## 配對、打分、十個回合

把上面的行為對回原始碼，`fuse_nodes` 的主迴圈最多跑十輪，融到沒有變化為止，前一輪融出來的合體 node 下一輪繼續當候選人，鏈狀的融合就這樣一輪一輪長大。

每一輪的分工是這樣。先把 node 按「用到同一個 buffer」分組，只在組內兩兩配對，避免全圖 O(n²) 亂配，畢竟沒碰過同一塊記憶體的兩個 node 融了也省不到東西。每一對過 `can_fuse` 的合法性檢查，extern 和 nop 不融、device 要相同、融了不能在依賴圖上繞出 cycle。這裡還分兩種局面，有 producer 和 consumer 關係的叫垂直融合，要再過一關垂直方向的檢查，consumer 的每條 unmet dependency 都要能對上 producer 的 write，對不上就是剛剛那兩種下場。彼此沒有依賴、只是讀同一批 input 的叫水平融合，門檻反而更高，因為沒有中間 buffer 可省，賺頭只剩共用的讀取。

活下來的配對交給 `score_fusion` 排序，分數是一個 tuple，省下的記憶體流量是主要項，兩個 node 在原圖中的距離當次要項，位置近的優先，因為隔太遠的融合容易把別的 buffer 生命週期拉長，墊高峰值記憶體。分數高的先融，融完更新候選名單，繼續消化下一對。

值得注意的是這套機制沒有任何一條「sin 跟 add 可以融、softmax 跟 matmul 不行」的規則，Scheduler 不認識 op，只看讀寫。所有判準最後都繞回同一個問題，這次融合值不值一趟記憶體往返。

## 結語

Scheduler 把 IR node 包成 SchedulerNode，靠 buffer 的讀寫關係建出依賴邊，排出拓撲順序，再用十輪配對把值得的融合一個個敲定。fusion 的本質是省 memory bandwidth，score 用省下的 byte 數計價，省不到流量的融合沒有意義，迴圈形狀對不上的融合不合法，這三句話就是今天實驗的全部。

不過「誰跟誰能融」還有一整張規則表沒攤開，垂直與水平融合的差別在哪、兩個 reduction 什麼條件下能融、pointwise 怎麼搭上 matmul 的便車。明天就把 fusion 的邊界一條一條畫清楚。那我們明天見！

## 參考資料

- [torch/_inductor/scheduler.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/scheduler.py)
- [torch/_inductor/choices.py：score_fusion 與 can_fuse（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/choices.py)
- [torch/_inductor/dependencies.py：MemoryDep（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/dependencies.py)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）
