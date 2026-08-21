# Day 8 | OutputGraph：散落的產出怎麼收成一張圖

## 前言

前四天各講一條生產線：節點（Day 4）、包裝（Day 5）、Guard（Day 6）、修改帳（Day 7）。每條線都在產出東西，但我們一直沒問一個問題：這些產出寫到哪裡去了？節點說「往圖上加」，是往哪張圖上加？Guard 說「丟進集合」，是誰的集合？今天講倉庫。`InstructionTranslator` 是筆，[`OutputGraph`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py) 是紙：一個 frame 的一次編譯只有一個 OutputGraph，所有產出都寫在它上面。

原始碼裡它的 docstring 第一句話就把定位講死了：「Wrapper class to hold outputs of InstructionTranslator」。今天沿著這句話拆三件事：產出是怎麼一筆一筆寫進去的、輸入為什麼是「用到才登記」、以及 RETURN 或 Graph Break 的那一瞬間，`compile_subgraph` 怎麼把一切收攏成一張 FX Graph 交出去。

正文開始！

![散落的節點、輸入、Guard 與修改帳逐一飛進 OutputGraph，compile_subgraph 收攏成一張 FX Graph 並交給後端](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day08/output_graph.gif)

*圖一：OutputGraph 是倉庫。翻譯期間節點、輸入、Guard、修改帳逐筆匯入；RETURN 或 Graph Break 時 `compile_subgraph` 一次收攏：接上 output 節點、清掉沒用的輸入、交給後端、把 `__compiled_fn` 塞進 globals。*

## 一個 frame，一個倉庫

先把「一對一」講清楚，因為它解釋了前幾天的一個現象。docstring 接著說：OutputGraph 與被處理的 frame 一對一；當使用者的程式呼叫另一個函式，Dynamo 開的 `InliningInstructionTranslator` 會**繼續寫進 root translator 的同一個 OutputGraph**。這就是 Day 5 看到「`helper` 被 inline 之後，整條呼叫鏈攤平成一張圖」的機關：筆可以換好幾支（每 inline 一層就多一台 translator），紙從頭到尾只有一張。

這張紙上有什麼？看 `__init__` 就知道倉庫開張時擺了哪些貨架：

| 成員 | 裝什麼 | 誰來寫 |
|---|---|---|
| `graph`（經 `SubgraphTracer` 寫入） | 正在長大的 fx.Graph | 每一個進圖的 Tensor 運算 |
| `graphargs` | 圖的輸入清單，每個都帶著 Source | 值第一次被用到時 |
| `side_effects` | Day 7 的修改帳本，`SideEffects(self)` 就是在這裡建的 | 每一筆 Python 層修改 |
| `guards`（轉手到 TracingContext） | Day 6 的前提集合 | `VariableBuilder` 每包一個值 |
| `nn_modules` | 被追蹤到的 module、參數、buffer | `register_attr_or_module` |
| `installed_globals` 暫存區 | 等著塞進 frame globals 的東西，編譯結果就放這 | `install_global` |
| `output_instructions` | 收圖後生成的新 bytecode | `compile_subgraph`（明天的主角） |

有兩個細節值得停一下。

第一，`guards` 和 `nn_modules` 其實不是 OutputGraph 自己的欄位，而是 property，轉手到 `self.tracing_context`。`__init__` 裡先建了 `ShapeEnv` 和 `FakeTensorMode`，再用它們建一個 [`TracingContext`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_guards.py)，Guard 真正住在 `tracing_context.guards_context.dynamo_guards` 裡。為什麼要多這一層？因為 Guard 的產地不只 Dynamo：之後 AOTAutograd 和 Symbolic Shapes 也會往裡面添前提，TracingContext 是整條編譯管線共用的隨身包，OutputGraph 只是它在 Dynamo 這一段的持有者。

第二，有些 Guard 不是翻譯中長出來的，是倉庫開張那一刻就裝上的。`__init__` 的最後一步呼叫 `init_ambient_guards()`，一口氣裝上 `GRAD_MODE`、`DEFAULT_DEVICE`、`DETERMINISTIC_ALGORITHMS`、`TORCH_FUNCTION_STATE`、`SHAPE_ENV` 這些「環境」前提。還記得 Day 6 讀 Guard 樹時，最前面那幾行不來自任何參數的 `GLOBAL_STATE`、`DEFAULT_DEVICE` 嗎？出處就是這裡：每張圖天生就押了「全域環境跟編譯當下一樣」這一注，一行使用者程式碼都還沒翻就押好了。

## 節點怎麼寫上去：create_proxy 這條路

Day 4 說 `BuiltinVariable` 發現兩個運算元是 Tensor，就「往圖上加一個 `mul` 節點」。具體的路是：variable 層呼叫 [`wrap_fx_proxy`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/builder.py)，它請 `OutputGraph.create_proxy` 造節點，而 OutputGraph 又只是轉手：

```python
def create_proxy(self, *args, **kwargs):
    return self.current_tracer.create_proxy(*args, **kwargs)
```

真正動筆的是 `SubgraphTracer`，它是 `fx.Tracer` 的子類別，就定義在同一個檔案下半部。FX 本身的 `create_proxy` 只負責「在圖上造一個節點、回一個 proxy」，SubgraphTracer 覆寫它，在造完節點之後多做幾件 Dynamo 才需要的事：把當下正在翻譯的那條 bytecode 的原始碼位置記進 `node.meta`、記下 `nn_module_stack`（這個運算發生在哪個 module 的 forward 裡）、記下 `source_fn_stack`（是哪個使用者層級的函式產生的）。你在 `graph_code` 輸出裡看到的那行註解：

```python
# File: /root/output_graph.py:15 in f, code: return (x @ y + bias).relu()
```

就是這時候寫進 meta、印圖時再讀出來的。圖不只是節點的集合，每個節點都帶著「我從你的哪一行程式碼來」的出生證明，之後 Graph Break 訊息、profiler 歸因、AOTAutograd 的 stack trace 保留，全都吃這份 meta。

另外每個節點還掛著一個 `example_value`：一顆 FakeTensor，只有 shape、dtype、device，沒有數值。這是 Day 3 說「符號執行」的物質基礎，也是圖上每個值印得出 `f32[4, 4][4, 1]cuda:0` 這種標註的原因：形狀資訊一路都在，值從頭到尾不在。

## 輸入不是宣告出來的，是用到才登記

節點講完了，講輸入。Dynamo 不看函式簽名決定圖的輸入。一個 Tensor 要等到真的被用上，才呼叫 `create_graph_input` 建 placeholder、登記一筆 `GraphArg`。實際驗證：

```python
bias = torch.randn(4, device="cuda")

def f(x, y, unused):
    return (x @ y + bias).relu()

torch.compile(f)(torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda"), torch.randn(9, device="cuda"))
```

`graph_code` 印出來：

```python
def forward(self, L_x_: "f32[4, 4][4, 1]cuda:0", L_y_: "f32[4, 4][4, 1]cuda:0", L_bias_: "f32[4][1]cuda:0"):
    matmul: "f32[4, 4][4, 1]cuda:0" = l_x_ @ l_y_;  l_x_ = l_y_ = None
    add: "f32[4, 4][4, 1]cuda:0" = matmul + l_bias_;  matmul = l_bias_ = None
    relu: "f32[4, 4][4, 1]cuda:0" = add.relu();  add = None
    return (relu,)
```

三個觀察：

- **`unused` 不在圖裡**。傳了但沒用到的參數不會被登記；收圖前 `remove_unused_graphargs` 還會再掃一輪，把中途變成死代碼的輸入拔掉。
- **`bias` 也是輸入**。它不是參數，是外面抓進來的 Tensor。`LOAD_DEREF` 載入它的那一刻，`VariableBuilder` 把它包成 `TensorVariable`，順手 lift 成 root graph 的輸入，placeholder 的名字 `L_bias_` 就是它的 Source。Tensor 的值永遠當輸入而不是常數，這是 Day 5 的押注原則在這裡的體現。
- **中間值用完立刻 `= None`**：提早歸還引用，讓記憶體早點釋放。輸出永遠是 tuple，就算只有一個值。

`create_graph_input` 的實作有個乾淨的地方：placeholder 一律插在圖的最前面（沿著上一個 placeholder 往後接），然後把 `GraphArg` 掛在 `node.meta["grapharg"]` 上。所以 OutputGraph 的 `graphargs` property 不是另外維護的一份名單：

```python
@property
def placeholders(self) -> list[fx.Node]:
    return self.graph.find_nodes(op="placeholder")

@property
def graphargs(self) -> list[GraphArg]:
    return [node.meta["grapharg"] for node in self.placeholders]
```

輸入清單就是圖裡的 placeholder 自己，單一事實來源，登記和圖永遠不會不同步。每筆 `GraphArg` 帶著 Source 和 fake tensor：Source 給明天的 PyCodegen 用（生出把這個值推上 stack 的 bytecode），fake tensor 給後端當 example input。

`nn.Module` 的參數和 buffer 另有通道：`register_attr_or_module` 把它們掛進 `nn_modules`、以 `get_attr` 或輸入的形式進圖，Day 5 說「權重變成圖的輸入」的具體機關在這裡。

## SubgraphTracer 其實是一疊

上面 `create_proxy` 轉手時用的是 `self.current_tracer`，複數的暗示很明顯：

```python
@property
def root_tracer(self):
    return self.tracers[0]

@property
def current_tracer(self):
    return self.tracers[-1]
```

平常這疊只有一層，root tracer 從頭寫到尾。但遇到 `torch.cond`、activation checkpoint 這類「參數是函式」的 higher-order op，分支必須自成一張子圖：Dynamo 就 push 一個新的 SubgraphTracer，接下來的節點全寫進子圖，翻完 pop 回來，子圖以 submodule 的身分掛回主圖。

麻煩的是自由變數。子圖裡用到外層的值，FX 的世界觀裡這是不合法的（一張圖只能用自己的 placeholder），所以 SubgraphTracer 的 `create_proxy` 在寫節點前多一步：發現參數是外層的 proxy，就呼叫 `maybe_lift_tracked_freevar_to_input`，把它就地 lift 成子圖的輸入，而且是遞迴的，巢狀幾層就一路往上提幾層，直到碰到真正持有它的那層為止。FX 本身沒有這種巢狀管理，SubgraphTracer 這層包裝很大一部分是為它存在的。

## compile_subgraph：收圖的瞬間

倉庫收貨收到什麼時候？時機只有兩種：RETURN（整個 frame 翻完了）或 Graph Break（翻不下去了）。兩條路殊途同歸，都走進 `compile_subgraph`，它的 docstring 把要做的事列得很白：呼叫編好的子圖、補做 side effect、生成 stack 和 locals 的重建碼、存回 locals。展開來是一連串動作：

1. **算活性**：symbolic stack 和 locals 裡哪些值在這之後還會被用到（`_get_stack_values_to_restore`），它們得成為圖的輸出，不然斷點之後接不上。RETURN 時這很簡單，就是回傳值；Graph Break 時才是重頭戲，翻到一半的中間狀態全要保住。
2. **side_effects 結帳**（Day 7）：`codegen_suffix` 請帳本把每筆修改的重播碼生出來。
3. **收圖**：把活值接上 `output` 節點，`remove_unused_graphargs` 清掉沒用到的輸入，`_make_graph_module` 把 fx.Graph 包成 GraphModule。
4. **`call_user_compiler`**：把 GraphModule 交給 backend（inductor、eager、或你自訂的），拿回一個可呼叫的函式。這一步被 `restore_global_state()` 包著，確保後端是在「編譯當下的全域狀態」下工作的；後端炸了會被包成 `BackendCompilerFailed` 丟出來。這就是 Day 2 說「第三站可以換」的接口。
5. **`install_global`**：編譯結果塞進 frame 的 globals。名字來自 `unique_id("__compiled_fn", with_uuid=True)`，所以長成 `__compiled_fn_1_6d16fdd3_...` 這樣，帶 uuid 是為了讓不同 `torch.compile` 實例互不踩腳。實際看得到：

```python
>>> [k for k in g.__globals__ if k.startswith("__compiled_fn")]
['__compiled_fn_1_6d16fdd3_...', '__compiled_fn_4_998ecab5_...']
```

新 bytecode 之後只要一條 `LOAD_GLOBAL` 就叫得到它。這裡還有一個對稱的細節：`install_global_unsafe` 塞東西進 globals 時會順手註冊一個 `CleanupHook`，這張圖將來被淘汰時，塞進去的東西也會被撿走，倉庫不留垃圾。

兩個省錢細節也在這條路上。其一，收圖有一條快速道：stack 上全是普通的 `TensorVariable`、帳本是空的、沒有要重建的複雜結構，就直接「呼叫圖、`UNPACK_SEQUENCE` 攤開回傳值」完事，連暫存變數 `graph_out_0` 都省了。其二，圖是空的就不叫後端：

```python
if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
    output.extend(self.compile_and_call_fx_graph(...))
```

這段程式碼根本沒有 Tensor 運算的話，`compile_and_call_fx_graph` 整個跳過。用一個會計數的自訂 backend 驗證：

```python
def no_tensor(x):
    return len([1, 2, 3])

torch.compile(no_tensor, backend=counting_backend)(torch.randn(4))
# backend called: 0 times
```

後端一次都沒被叫到。純 Python 的 frame 被 Dynamo 翻完、發現無圖可交，就默默放行，這也是為什麼 `torch.compile` 套在不含 Tensor 運算的函式上幾乎無感。

最後一步，`compile_and_call_fx_graph` 的收尾是一行伏筆：

```python
cg = PyCodegen(self.root_tx)
cg.make_call_generated_code(name)
return cg.get_instructions()
```

拿回編譯結果還不夠，得有人把「載入 `__compiled_fn_1`、把參數照 Source 推上 stack、呼叫、拆開回傳的 tuple」這段新 bytecode 寫出來。寫的人叫 PyCodegen，這是明天的主角。

## 結語

OutputGraph 是一個 frame 一次編譯的收集點：節點經 SubgraphTracer 寫進 fx.Graph 並帶上出生證明、輸入按 Source 用到才登記且清單就是 placeholder 本身、Guard 住在 TracingContext 裡且環境前提出生就裝好、修改帳掛在旁邊。RETURN 或 Graph Break 時 `compile_subgraph` 收攏一切：算活性、結帳、清輸入、交給後端、把 `__compiled_fn` 塞進 globals，空圖則直接放行。

`__compiled_fn_1` 已經躺在 globals 裡，但 CPython 不會自己知道怎麼用它。還缺最後一步：生一段新的 bytecode，把「載入、擺參數、呼叫、拆輸出、重播帳本、return」寫出來。明天看 PyCodegen 和它底下的 bytecode 工具箱。那我們明天見！

## 參考資料

- [torch/_dynamo/output_graph.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py)
- [torch/_dynamo/variables/builder.py：GraphArg 與 wrap_fx_proxy（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/builder.py)
- [torch/_guards.py：TracingContext（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_guards.py)
- [Dynamo Deep-Dive：OutputGraph（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.fx 文件](https://pytorch.org/docs/stable/fx.html)
