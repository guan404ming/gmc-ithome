# Day 12 | AOTAutograd：為什麼 backward 也要先編好

## 前言

進入第二站。Dynamo 交出來的 FX Graph 只有 forward，而訓練的每一步都要 backward。Eager 模式下這不是問題，autograd 引擎在執行時動態建 tape、動態回放；但我們現在要的是編譯，backward 也得變成一張可以最佳化的圖，而且要在真正執行之前就拿到。這就是 AOTAutograd 名字裡 Ahead-of-Time 的意思。

它住在 [`torch/_functorch/aot_autograd.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/aot_autograd.py)，是 Dynamo 和 Inductor 之間的中間層。今天看它的輸入輸出長什麼樣，後面幾天再拆開它中間做的事。

正文開始！

![AOTAutograd 把一張 torch 層圖變成兩張 ATen 層圖](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day12/aot_two_graphs.png)

*圖一：AOTAutograd 的輸入輸出。Dynamo 的 torch 層 forward 圖經 FakeTensor 重跑、autograd 引擎展開，變成 ATen 層的 Forward 與 Backward 兩張圖；forward 多輸出 `le`、`permute` 這些保存值傳給 backward。*

## 從一張圖變兩張圖

拿一個最小的訓練形狀實跑：

```python
def f(x, w):
    return (x @ w).relu().sum()

x = torch.randn(4, 4, device="cuda")
w = torch.randn(4, 4, device="cuda", requires_grad=True)

torch._logging.set_logs(aot_graphs=True)
out = torch.compile(f)(x, w)
out.backward()
```

`aot_graphs` 印出兩張圖。Forward：

```python
 ===== Forward graph 1 =====
def forward(self, primals_1: "f32[4, 4]", primals_2: "f32[4, 4]"):
    mm = torch.ops.aten.mm.default(primals_1, primals_2)
    relu = torch.ops.aten.relu.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(relu)
    le = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    permute = torch.ops.aten.permute.default(primals_1, [1, 0])
    return (sum_1, le, permute)
```

Backward：

```python
 ===== Backward graph 1 =====
def forward(self, le: "b8[4, 4]", permute: "f32[4, 4]", tangents_1: "f32[]"):
    expand = torch.ops.aten.expand.default(tangents_1, [4, 4])
    full_default = torch.ops.aten.full.default([], 0.0, ...)
    where = torch.ops.aten.where.self(le, full_default, expand)
    mm_1 = torch.ops.aten.mm.default(permute, where)
    return (None, mm_1)
```

幾個值得盯著看的地方：

- **Forward 的輸出不只 loss**。使用者只要 `sum_1`，但圖還回傳了 `le`（relu 的 mask）和 `permute`（`x` 的轉置）。這些是 backward 需要的中間值，被「存」下來從 forward 傳給 backward。存哪些、存多少，是一個真正的取捨，Day 15 的 partitioner 專門管這件事。
- **Backward 也是一張普通的 FX Graph**。輸入是存下來的值加上 `tangents_1`（上游梯度，這裡是 loss 對自己的梯度，純量 1），輸出對齊 forward 的輸入：`x` 不需要梯度所以是 `None`，`w` 的梯度是 `mm_1`。它跟 forward 一樣交給 Inductor 編譯，backward 也享受 fusion。
- **relu 的 backward 不是查表得來的**。圖裡是 `le` + `where`：小於等於 0 的位置梯度歸零。AOTAutograd 是拿 FakeTensor 把 forward 重新執行一遍、讓 autograd 引擎在上面跑出 backward，再把整個過程 trace 下來，所以微分規則來自 PyTorch 本來的 derivatives 定義，不是 AOTAutograd 自己寫的。

對照組：同一個函式包在 `torch.no_grad()` 裡編，`aot_graphs` 只印一張 Forward graph，沒有 `le` 也沒有 `permute`，輸出只有 `(sum_1,)`。要不要展開 backward，是看輸入的 `requires_grad` 和 grad mode 決定的，這也是 Day 6 `GLOBAL_STATE grad_mode` 那條 Guard 存在的原因。

## 順便換了一種語言

比較 Dynamo 圖和 AOT 圖，同一行程式的長相：

```
Dynamo:  matmul = l_x_ @ l_w_          （torch 層，使用者寫什麼就是什麼）
AOT:     mm = torch.ops.aten.mm.default(primals_1, primals_2)   （ATen 層）
```

AOTAutograd 拿 FakeTensor 重跑 forward 時，每個 op 都會經過 PyTorch 的 dispatcher，落到 ATen 層的正式名字。順帶發生的還有兩件事，各佔一天：in-place 和 view 被改寫成純函數式（Functionalization，明天），高階 op 被拆成更小的基本運算（Decomposition，後天）。所以 Inductor 拿到的圖，跟你寫的 Python 已經隔了三層轉換，但每一層都留著 log 可以對照。

## 為什麼要 ahead-of-time

Eager 的 autograd 是 define-by-run：forward 跑到哪、tape 記到哪，backward 按 tape 回放。好處是彈性，代價是 backward 永遠是一連串分開的 kernel，沒有全局可看、沒有 fusion 可做。AOT 展開之後：

- backward 是一張完整的圖，Inductor 能融合它，訓練的加速一半來自這裡。
- forward 和 backward 之間「存什麼」變成可以全局規劃的決策，而不是 autograd 引擎的預設行為（存所有需要的中間值）。省記憶體的 activation checkpointing 在這個框架下變成 partitioner 的一個策略。
- 展開發生在編譯期，用的是 FakeTensor，不碰真資料，成本一次付清。

## 結語

AOTAutograd 把 Dynamo 的一張 torch 層 forward 圖，變成兩張 ATen 層的圖：forward 多輸出一批要保存的中間值，backward 拿著它們和上游梯度算出對輸入的梯度，兩張各自交給 Inductor。微分不是它自己算的，是讓 autograd 引擎在 FakeTensor 上跑一遍再錄下來。

明天拆第一層轉換：Functionalization。`x.add_(1)`、view 這些會就地改記憶體的操作，是怎麼被改寫成純函數式、又保證語意不變的。那我們明天見！

## 參考資料

- [torch/_functorch/aot_autograd.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/aot_autograd.py)
- [functorch：aot_autograd 文件](https://pytorch.org/functorch/stable/aot_autograd.html)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 4 節）
