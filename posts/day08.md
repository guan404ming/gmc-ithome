# Day 8 | OutputGraph：散落的產出怎麼收成一張圖

## 前言

前四天各講一條生產線：節點（Day 4）、包裝（Day 5）、Guard（Day 6）、修改帳（Day 7）。今天講倉庫。`InstructionTranslator` 是筆，[`OutputGraph`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py) 是紙：一個 frame 的一次編譯只有一個 OutputGraph，所有產出都寫在它上面。

正文開始！

![OutputGraph 收集所有產出，compile_subgraph 一次收攏](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day08/output_graph.png)

*圖一：OutputGraph 是倉庫。左邊三條生產線逐條累積（節點、輸入、Guard、修改帳），RETURN 或 Graph Break 時 `compile_subgraph` 收攏：交給後端、把 `__compiled_fn` 塞進 globals、Guard 編成 C++ 樹。*

## 一張圖與它的周邊

| 成員 | 裝什麼 |
|---|---|
| `graph`（經 `SubgraphTracer` 寫入） | 正在長大的 fx.Graph |
| `graphargs` | 圖的輸入清單，每個都帶著 Source |
| `side_effects` | Day 7 的修改帳本 |
| guards（經 TracingContext 累積） | Day 6 的前提集合 |
| `install_global` 暫存區 | 等著塞進 frame globals 的東西，編譯結果就放這 |

## 輸入不是宣告出來的，是用到才登記

Dynamo 不看函式簽名決定圖的輸入。一個 Tensor 要等到真的被用上，才呼叫 `create_graph_input` 建 placeholder、登記一筆 GraphArg。實際驗證：

```python
bias = torch.randn(4, device="cuda")

def f(x, y, unused):
    return (x @ y + bias).relu()

torch.compile(f)(torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda"), torch.randn(9, device="cuda"))
```

`graph_code` 印出來：

```python
def forward(self, L_x_: "f32[4, 4][4, 1]cuda:0", L_y_: "f32[4, 4][4, 1]cuda:0", L_bias_: "f32[4][1]cuda:0"):
    matmul = L_x_ @ L_y_
    add = matmul + L_bias_
    relu = add.relu()
    return (relu,)
```

三個觀察：

- **`unused` 不在圖裡**。傳了但沒用到的參數不會被登記；收圖前 `remove_unused_graphargs` 還會再掃一輪，把中途變成死代碼的輸入拔掉。
- **`bias` 也是輸入**。它不是參數，是外面抓進來的 Tensor，但 Tensor 的值永遠當輸入而不是常數（Day 5 的押注原則），placeholder 的名字就是它的 Source。
- **中間值用完立刻 `= None`**：提早歸還引用，讓記憶體早點釋放。輸出永遠是 tuple，就算只有一個值。

`nn.Module` 的參數和 buffer 另有通道：`register_attr_or_module` 把它們掛進圖，Day 5 說「權重變成圖的輸入」的具體機關在這裡。

另外，平常只有一層 `SubgraphTracer` 在寫節點，但它其實是一疊：遇到 `torch.cond`、activation checkpoint 這類「參數是函式」的 higher-order op，分支必須自成一張子圖，就 push 一個新的 tracer，分支用到外層的值就地 lift 成子圖輸入。FX 本身沒有這種巢狀管理，SubgraphTracer 這層包裝很大一部分是為它存在的。

## compile_subgraph：收圖的瞬間

時機只有兩種：RETURN（整個 frame 翻完了）或 Graph Break（翻不下去了）。收圖是一連串動作：

1. **算活性**：symbolic stack 和 locals 裡哪些值在這之後還會被用到，它們得成為圖的輸出，不然斷點之後接不上。
2. **side_effects 結帳**（Day 7）。
3. 把活值接上 output 節點，`remove_unused_graphargs` 清掉沒用到的輸入。
4. **`call_user_compiler`**：把 GraphModule 交給 backend（inductor、eager、或你自訂的），拿回一個可呼叫的函式。後端炸了會被包成 `BackendCompilerFailed` 丟出來。這就是 Day 2 說「第三站可以換」的接口。
5. **`install_global`**：編譯結果以 `__compiled_fn_1` 這種名字塞進 frame 的 globals。實際看得到：

```python
>>> [k for k in g.__globals__ if k.startswith("__compiled_fn")]
['__compiled_fn_1_6d16fdd3_...', '__compiled_fn_4_998ecab5_...']
```

新 bytecode 之後只要一條 `LOAD_GLOBAL` 就叫得到它。

一個省錢細節：圖是空的（這段程式碼根本沒有 Tensor 運算）就直接跳過後端。用一個會計數的自訂 backend 驗證：

```python
def no_tensor(x):
    return len([1, 2, 3])

torch.compile(no_tensor, backend=counting_backend)(torch.randn(4))
# backend called: 0 times
```

## 結語

OutputGraph 是一個 frame 一次編譯的收集點：節點寫進 fx.Graph、輸入按 Source 用到才登記、Guard 和修改帳掛在旁邊。RETURN 或 Graph Break 時 `compile_subgraph` 收攏一切：算活性、結帳、清輸入、交給後端、把 `__compiled_fn` 塞進 globals。

`__compiled_fn_1` 已經躺在 globals 裡，但 CPython 不會自己知道怎麼用它。還缺最後一步：生一段新的 bytecode，把「載入、擺參數、呼叫、拆輸出、重播帳本、return」寫出來。明天看 PyCodegen 和它底下的 bytecode 工具箱。那我們明天見！

## 參考資料

- [torch/_dynamo/output_graph.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py)
- [Dynamo Deep-Dive：OutputGraph（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.fx 文件](https://pytorch.org/docs/stable/fx.html)
