# Day 8 | TorchDynamo 的中央倉庫：OutputGraph

## 前言

前四天我們講了 TorchDynamo 四條重要的生產線，分別是 InstructionTranslator（Day 4）、VariableTracker（Day 5）、Guard（Day 6）、SideEffects（Day 7）。每條產線都會產出東西，不過有一個問題我們一直沒問：這些產出到底寫到哪裡去了？InstructionTranslator 說「往圖上加」，是往哪張圖上加？Guard 說「丟進集合」，又是誰的集合？今天就來介紹這個負責收貨的中央倉庫。簡單來說，`InstructionTranslator` 是筆，[`OutputGraph`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py) 就是那張紙。一個 frame 的一次編譯只會有一個 OutputGraph，所有產出都會被寫在它上面。

原始碼裡它的 docstring 第一句話就是「Wrapper class to hold outputs of InstructionTranslator」它基本上把它自己定位講得非常清楚。那今天我們就沿著這句話來拆三件事，產出是怎麼一筆一筆寫進去的、input 為什麼是「用到才登記」、以及 RETURN 或 Graph Break 的那一瞬間，`compile_subgraph` 是怎麼把一切收攏成一張 FX Graph 交出去的。

正文開始！

## 一個 frame，一個倉庫

先來把「一對一」這件事講清楚，因為它其實解釋了前幾天看到的一個現象。當使用者的程式呼叫另一個函式，Dynamo 開的 `InliningInstructionTranslator` 會**繼續寫進 root translator 的同一個 OutputGraph**。這就是 Day 5 在 VariableTracker 看到「`helper` 被 inline 之後，整條呼叫鏈攤平成一張圖」的機關。筆可以換好幾支（每 inline 一層就多一台 translator），紙從頭到尾只有一張。

那這張紙上到底有什麼呢？看一下它的 `__init__` 就知道倉庫開張的時候擺了哪些貨架。


| 成員                             | 裝什麼                                   | 誰來寫                       |
| ------------------------------ | ------------------------------------- | ------------------------- |
| `graph`（經 `SubgraphTracer` 寫入） | 正在長大的 fx.Graph                        | 每一個進圖的 Tensor 運算          |
| `graphargs`                    | 圖的 input 清單，每個都帶著 Source                   | 值第一次被用到時                  |
| `side_effects`                 | Day 7 的帳本，`SideEffects(self)` 就是在這裡建的 | 每一筆 Python 層修改            |
| `guards`（轉手到 TracingContext）   | Day 6 的押注集合                           | `VariableBuilder` 每包一個值   |
| `nn_modules`                   | 被追蹤到的 module、參數、buffer                | `register_attr_or_module` |
| `installed_globals` 暫存區        | 等著塞進 frame globals 的東西，編譯結果就放這        | `install_global`          |
| `output_instructions`          | 收圖後生成的新 bytecode                      | `compile_subgraph`（明天的主角） |


其中有兩個細節：

1. Guard 其實不是存在 OutputGraph 自己身上，而是放在一個叫 [`TracingContext`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_guards.py) 的共用容器裡。因為 Guard 的產地不只 Dynamo，之後 AOTAutograd 和 Symbolic Shapes 也會往裡面添前提，TracingContext 是整條編譯管線共用的隨身包，OutputGraph 只是它在 Dynamo 這一段的持有者。
2. 有些 Guard 不是翻譯中長出來的，而是倉庫開張那一刻就裝上的。`__init__` 收尾的 `init_ambient_guards()` 會裝上 `GRAD_MODE`、`DEFAULT_DEVICE` 這類「環境」前提。還記得 Day 6 讀 Guard 樹時，最前面那幾行不來自任何參數的 `GLOBAL_STATE` 嗎？出處就是這裡。每張圖天生就押了「全域環境跟編譯當下一樣」這一注，一行使用者程式碼都還沒翻就押好了。

## node 是怎麼被寫進圖裡的

Day 4 說 `BuiltinVariable` 發現兩個運算元是 Tensor，就「往圖上加一個 `mul` node」。這條路最後會走到 OutputGraph 手上的 `SubgraphTracer`，它是 `fx.Tracer` 的子類別，真正動筆的人就是它。FX 本身的 `create_proxy` 只負責「在圖上造一個 node、回一個 proxy」，SubgraphTracer 則在造完 node 之後多做幾件 Dynamo 才需要的事，像是把當下正在翻譯的那條 bytecode 的原始碼位置記進 `node.meta`、記下這個運算發生在哪個 module 的 forward 裡。你在 `graph_code` 輸出裡看到的那行註解長得像下面這樣。

```python
# File: /root/output_graph.py:15 in f, code: return (x @ y + bias).relu()
```

它就是這時候寫進 meta、印圖時再讀出來的。所以圖不只是 node 的集合，每個 node 都帶著「我從你的哪一行程式碼來」的出生證明，之後 Graph Break 訊息、profiler 歸因、AOTAutograd 的 stack trace 保留，全都吃這份 meta。另外，每個 node 還掛著一個 `example_value`，一顆只有 shape、dtype、device、沒有數值的 FakeTensor。這就是 Day 3 說「符號執行」的基礎，也是圖上每個值印得出 `f32[4, 4][4, 1]cuda:0` 這種標註的原因。形狀資訊一路都在，值從頭到尾沒有存。

## input 是用到的時候才登記的

node 講完了，接著來講 input。Dynamo 不看 function signature 決定圖的 input，一個 Tensor 要等到真的被用上，才呼叫 `create_graph_input` 建 placeholder、登記一筆 `GraphArg`。拿一小段程式驗證看看。

```python
bias = torch.randn(4, device="cuda")

def f(x, y, unused):
    return (x @ y + bias).relu()

torch.compile(f)(torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda"), torch.randn(9, device="cuda"))
```

`graph_code` 印出來如下。

```python
def forward(self, L_x_: "f32[4, 4][4, 1]cuda:0", L_y_: "f32[4, 4][4, 1]cuda:0", L_bias_: "f32[4][1]cuda:0"):
    matmul: "f32[4, 4][4, 1]cuda:0" = l_x_ @ l_y_;  l_x_ = l_y_ = None
    add: "f32[4, 4][4, 1]cuda:0" = matmul + l_bias_;  matmul = l_bias_ = None
    relu: "f32[4, 4][4, 1]cuda:0" = add.relu();  add = None
    return (relu,)
```

有三個地方值得放大來看。

- `unused` **不在圖裡**。傳了但沒用到的參數不會被登記，收圖前 `remove_unused_graphargs` 還會再掃一輪，把中途變成死代碼的 input 拔掉。
- `bias` **也是 input**。它不是參數，是外面抓進來的 Tensor。`LOAD_DEREF` 載入它的那一刻，`VariableBuilder` 把它包成 `TensorVariable`，順手 lift 成 root graph 的 input，placeholder 的名字 `L_bias_` 就是它的 Source。Tensor 的值永遠當 input 而不是常數，這就是 Day 5 那個押注原則在這裡的體現。
- **中間值用完立刻 `= None`**，提早歸還引用，讓記憶體早點釋放。輸出永遠是 tuple，就算只有一個值。

這裡還有個蠻乾淨的設計。OutputGraph 沒有另外維護一份 input 名單，input 清單就是圖裡的 placeholder 自己，單一事實來源，登記和圖永遠不會不同步。每筆 `GraphArg` 都帶著 Source 和 fake tensor。Source 給明天的 PyCodegen 用（生出把這個值推上 stack 的 bytecode），fake tensor 則給後端當 example input。

至於 `nn.Module` 的參數和 buffer，則另有通道，由 `register_attr_or_module` 把它們掛進 `nn_modules`、以 `get_attr` 或 input 的形式進圖，Day 5 說「權重變成圖的 input」的具體機關就在這裡。

## SubgraphTracer 其實是一個 stack？

OutputGraph 裡動筆的 tracer 其實是一個 stack。平常這個 stack 只有一層，root tracer 從頭寫到尾。但遇到 `torch.cond`、activation checkpoint 這類「參數是函式」的 higher-order op，分支必須自成一張子圖，Dynamo 就 push 一個新的 SubgraphTracer，接下來的 node 全寫進子圖，翻完 pop 回來，子圖以 submodule 的身分掛回主圖。

比較麻煩的是 free variable。子圖裡用到外層的值，在 FX 的世界觀裡這是不合法的（一張圖只能用自己的 placeholder），所以 SubgraphTracer 的 `create_proxy` 在寫 node 前多一步。發現參數是外層的 proxy，就呼叫 `maybe_lift_tracked_freevar_to_input`，把它就地 lift 成子圖的 input，而且是遞迴的，巢狀幾層就一路往上提幾層，直到碰到真正持有它的那層為止。FX 本身沒有這種巢狀管理，SubgraphTracer 這層包裝很大一部分就是為它存在的。

## compile_subgraph 收圖時做了什麼

那倉庫收貨要收到什麼時候呢？時機只有兩種，RETURN（整個 frame 翻完了）或 Graph Break（翻不下去了）。兩條路殊途同歸，都走進 `compile_subgraph`：

1. **計算活性**：symbolic stack 和 locals 裡哪些值在這之後還會被用到（`_get_stack_values_to_restore`），它們得成為圖的輸出，不然斷點之後接不上。RETURN 時這很簡單，就是回傳值。Graph Break 時才是重頭戲，翻到一半的中間狀態全要保住。
2. **side_effects 結帳**（Day 7）：`codegen_suffix` 請帳本把每筆修改的 replay 的 bytecode生出來。
3. **收圖**：把還活著的值接上 `output` node，`remove_unused_graphargs` 清掉沒用到的 input，`_make_graph_module` 把 fx.Graph 包成 GraphModule。
4. `call_user_compiler`：把 GraphModule 交給 backend（inductor、eager、或你自訂的），拿回一個可呼叫的函式。這一步被 `restore_global_state()` 包著，確保後端是在「編譯當下的全域狀態」下工作的。後端炸了會被包成 `BackendCompilerFailed` 丟出來。這就是 Day 2 說「第三站可以換」的接口。
5. `install_global`：編譯結果以 `__compiled_fn_1_6d16fdd3_...` 這樣的名字塞進 frame 的 globals（帶 uuid 是為了讓不同 `torch.compile` 實例互不衝突），新 bytecode 之後只要一條 `LOAD_GLOBAL` 就叫得到它。

```python
>>> [k for k in g.__globals__ if k.startswith("__compiled_fn")]
['__compiled_fn_1_6d16fdd3_...', '__compiled_fn_4_998ecab5_...']
```

這條路上還有一個有趣的細節，只要這裡圖是空的就不呼叫後端。這段程式碼根本沒有 Tensor 運算的話，交給後端這一步整個跳過。我們可以用一個會計數的自訂 backend 來驗證。

```python
def no_tensor(x):
    return len([1, 2, 3])

torch.compile(no_tensor, backend=counting_backend)(torch.randn(4))
# backend called: 0 times
```

後端一次都沒被叫到。純 Python 的 frame 被 Dynamo 翻完、發現無圖可交，就默默放行，這也是為什麼 `torch.compile` 套在不含 Tensor 運算的函式上會幾乎無感。

最後留一個伏筆。拿回編譯結果還不夠，得有人把「載入 `__compiled_fn_1`、把參數照 Source 推上 stack、呼叫、拆開回傳的 tuple」這段新 bytecode 寫出來。而負責寫的人就叫 PyCodegen，也就是明天的主角。

下面的動畫簡單的把今天整條「逐筆進貨、一次收攏」的流程走了一遍：

![散落的 node、input、Guard 與修改帳逐一飛進 OutputGraph，compile_subgraph 收攏成一張 FX Graph 並交給後端](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day08/output_graph.gif)

*圖一：OutputGraph 是倉庫。翻譯期間 node、input、Guard、修改帳逐筆匯入。RETURN 或 Graph Break 時 `compile_subgraph` 一次收攏，接上 output node、清掉沒用的 input、交給後端、把 `__compiled_fn` 塞進 globals。*

## 結語

OutputGraph 就是一個 frame 一次編譯的收集點。node 經 SubgraphTracer 寫進 fx.Graph 並帶上出生證明、input 按 Source 用到才登記且清單就是 placeholder 本身、Guard 住在 TracingContext 裡且環境前提出生就裝好、修改帳則掛在旁邊。RETURN 或 Graph Break 時 `compile_subgraph` 收攏一切，算活性、結帳、清 input、交給後端、把 `__compiled_fn` 塞進 globals，空圖則直接放行。

到這裡 `__compiled_fn_1` 已經躺在 globals 裡了，但 CPython 不會自己知道怎麼用它。所以還缺最後一步，生一段新的 bytecode，把「載入、擺參數、呼叫、拆輸出、replay 帳本、return」寫出來。明天我們就來看看 PyCodegen 和它底下的 bytecode 工具箱。那我們明天見！

## 參考資料

- [torch/_dynamo/output_graph.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py)
- [torch/_dynamo/variables/builder.py：GraphArg 與 wrap_fx_proxy（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/builder.py)
- [torch/_guards.py：TracingContext（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_guards.py)
- [Dynamo Deep-Dive：OutputGraph（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.fx 文件](https://pytorch.org/docs/stable/fx.html)

