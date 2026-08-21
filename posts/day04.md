# Day 4 | 走進 TorchDynamo 的心臟：InstructionTranslator

## 前言

昨天講到 Dynamo 把 `eval_frame` 指標換掉、接手你的 frame 之後，會「符號式地執行 bytecode」，然後就快轉過去了。今天我們就來補上這個洞。

原始碼的位置在 [`torch/_dynamo/symbolic_convert.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/symbolic_convert.py)，總共四千多行，PyTorch 團隊自己在文件裡說這是 「Dynamo 的心臟」。而裡面住著一個叫 `InstructionTranslator` 的 class，做的事簡單來說，就是把 CPython 的直譯器在 Python 層重寫一遍，只是跑在上面的不是真值，而是符號。

正文開始！

## 進門前的一站：convert_frame

先接上昨天的結尾。C 層的 frame hook 發現這個 frame 沒有可用的快取，會呼叫 Python 端的 callback，這個 callback 住在 [`convert_frame.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/convert_frame.py)。它做幾件行政工作：檢查這個 frame 是不是該碰（有些檔案在 skip 名單上）、檢查重編次數有沒有超過上限（超過就放棄，改用 eager 跑），然後呼叫 `_compile()`，在裡面一個叫 `transform()` 的函式建出 `InstructionTranslator`，真正的翻譯才開始。

所以層次是：C 層攔 frame -\> `convert_frame` 把關 -\> `InstructionTranslator` 動手。

## 複習：CPython 是一台 stack machine

要理解 Dynamo 怎麼模擬 CPython，先要記得 CPython 自己怎麼跑。它是一台 stack machine：每個 frame 有一個 value stack，每條 bytecode 都是對這個 stack 的操作。拿一個小函式來看：

```python
def f(x, y):
    z = x * y
    return z + 1
```

Python 3.12 下 `dis.dis(f)` 印出來是：

     17   2 LOAD_FAST     0 (x)
          4 LOAD_FAST     1 (y)
          6 BINARY_OP     5 (*)
         10 STORE_FAST    2 (z)
     18  12 LOAD_FAST     2 (z)
         14 LOAD_CONST    1 (1)
         16 BINARY_OP     0 (+)
         20 RETURN_VALUE

`LOAD_FAST x` 把區域變數 `x` 推上 stack；`BINARY_OP 5 (*)` 彈出兩個值、相乘、把結果推回去；`STORE_FAST z` 彈出頂端存進區域變數 `z`。八條指令，一台機器，一個 stack，一張區域變數表。

一個小提醒：bytecode 是跟著 Python 版本走的。3.10 以前乘法是 `BINARY_MULTIPLY`、加法是 `BINARY_ADD`，3.11 把它們合併成一條帶參數的 `BINARY_OP`；函式呼叫在 3.11 之後也從 `CALL_FUNCTION` 變成 `PUSH_NULL` 加 `CALL`。這代表 Dynamo 得對每一個支援的 Python 版本維護一套對應，也是為什麼新版 Python 剛出時 `torch.compile` 常常要等幾個月才跟上。

## InstructionTranslator：同一台機器，換成符號跑

`InstructionTranslator` 把這台機器原樣搬過來，換掉兩個核心零件：

- `stack`：一個 Python list，裡面裝的不是真值，而是 `VariableTracker`，每個 Python 值的符號替身（這是明天的主角，目前可以先把它當一個容器來理解）。
- `symbolic_locals`：一個 dict，變數名對到 `VariableTracker`，模擬 frame 的區域變數表。

指令怎麼執行？class 上每個 opcode 會對應一個同名 method：`LOAD_FAST` 指令由 `LOAD_FAST` 這個 method 處理，`STORE_FAST` 由 `STORE_FAST` 處理。PyTorch 在這裡用了 metaclass：class 定義完成的瞬間，自動把這些方法收進一張 `dispatch_table`：

```python
class BytecodeDistpatchTableMeta(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        dispatch_table = {
            op: getattr(cls, opname, functools.partial(_missing, opname))
            for opname, op in dis.opmap.items()
        }
        cls.dispatch_table = [dispatch_table.get(i) for i in range(2**8)]
```

（原始碼裡就拼成 `Distpatch`，不是我打錯。可以去發 PR 了，笑）沒有對應方法的 opcode 會塞一個 `_missing`，被呼叫到就丟出「Missing bytecode handler」的 graph break。實際數一下，PyTorch 2.8 在 Python 3.12 上這張表 256 格裡有 129 格是真的 handler。而 3.12 真實的 opcode 恰好就是 129 個（dis.opmap 裡另外 11 個是編號超過 255 的偽指令，只活在編譯器內部），也就是說 TorchDynamo 把每一條指令都接住了。

    entries: 256 | handlers: 129
      LOAD_FAST          -> InstructionTranslatorBase.LOAD_FAST
      STORE_FAST         -> InstructionTranslatorBase.STORE_FAST
      BINARY_OP          -> InstructionTranslatorBase.BINARY_OP
      CALL               -> InstructionTranslatorBase.CALL
      RETURN_VALUE       -> InstructionTranslator.RETURN_VALUE
      POP_JUMP_IF_FALSE  -> generic_jump..inner

主迴圈 `run()` 就是 `while self.step(): pass`，而 `step()` 的核心只有一行：

```python
self.dispatch_table[inst.opcode](self, inst)
```

取下一條指令、查表、呼叫 handler。每個 handler 做的事，就是把 CPython 對真值做的操作，翻譯成對符號做的操作。看兩個最簡單的：

```python
def LOAD_FAST(self, inst):
    name = inst.argval
    self.push(self.symbolic_locals[name].unwrap())

def STORE_FAST(self, inst):
    name = inst.argval
    loaded_vt = self.pop()
    loaded_vt.set_name_hint(name)
    self.symbolic_locals[name] = loaded_vt
```

跟 CPython 的 C 實作一模一樣的形狀，只是 push 和 pop 的東西從 `PyObject*` 變成了 `VariableTracker`。二元運算則統一用一個小工廠生出來：

```python
def stack_op(fn):
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)
    def impl(self, inst):
        self.push(fn_var.call_function(self, self.popn(nargs), {}))
    return impl

BINARY_MULTIPLY = stack_op(operator.mul)
BINARY_ADD = stack_op(operator.add)
```

彈出 n 個運算元，交給一個包著 `operator.mul` 的 `BuiltinVariable` 去「呼叫」，最後再把結果推回去 stack。3.11 之後的 `BINARY_OP` 只是多一層查表，依 `inst.arg` 轉到對應的 `stack_op`。至於「呼叫」在符號世界裡代表什麼，就是關鍵所在：如果兩個運算元都是 Tensor 的替身，`BuiltinVariable` 不會真的算，而是往 FX Graph 加一個 `mul` 節點，然後推回一個代表結果的新替身；如果兩個都是 Python 常數，它就直接算掉，推回一個常數替身。

順帶一提，`symbolic_convert.py` 裡其實有一個基底和兩個子類別。

- `InstructionTranslatorBase` 放所有 handler 和主迴圈
- `InstructionTranslator` 是最外層那個 frame 用的，多了「翻譯結束要收圖、生成新 bytecode」的收尾邏輯
- `InliningInstructionTranslator` 則是碰到呼叫你自己寫的函式時，開來鑽進被呼叫函式 bytecode 的那一台，翻完把回傳值接回上一層的 stack。

同一套 handler、同一張 `dispatch_table`，只是誰負責收尾不一樣。這也是為什麼呼叫自己的函式不會斷圖，明天講 `VariableTracker` 時會再看到它。

## 逐條走一遍

![InstructionTranslator 逐條走過 bytecode](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day04/translator.gif)

*圖一：InstructionTranslator 逐條走過 `f` 的八條 bytecode。左邊是指令，中間是符號 stack 與 `symbolic_locals`，右邊是 FX Graph。大部分指令只動 stack 和 locals，只有兩條 `BINARY_OP` 往圖上加了節點。*

`TORCH_LOGS="trace_bytecode"` 會把 `step()` 每一步印出來，附上當下的 stack。拿上面的 `f` 實際跑：

```python
torch._logging.set_logs(trace_bytecode=True)
torch.compile(f)(torch.randn(8, device="cuda"), torch.randn(8, device="cuda"))
```

    TRACE LOAD_FAST x        []
    TRACE LOAD_FAST y        [LazyVariableTracker()]
    TRACE BINARY_OP 5        [LazyVariableTracker(), LazyVariableTracker()]
    TRACE STORE_FAST z       [TensorVariable()]
    TRACE LOAD_FAST z        []
    TRACE LOAD_CONST 1       [TensorVariable()]
    TRACE BINARY_OP 0        [TensorVariable(), ConstantVariable(int: 1)]
    TRACE RETURN_VALUE None  [TensorVariable()]

每一行印的是「即將執行這條指令時」的 stack。對照整理成表，右邊補上圖裡多了什麼：

| 指令           | 執行後的 stack（頂端在右） | `symbolic_locals` | 圖裡多了什麼 |
|----------------|----------------------------|-------------------|--------------|
| `LOAD_FAST x`  | `[x]`                      |                   | 沒有         |
| `LOAD_FAST y`  | `[x, y]`                   |                   | 沒有         |
| `BINARY_OP *`  | `[Tensor(z)]`              |                   | `mul` 節點   |
| `STORE_FAST z` | `[]`                       | `z` 記下          | 沒有         |
| `LOAD_FAST z`  | `[Tensor(z)]`              |                   | 沒有         |
| `LOAD_CONST 1` | `[Tensor(z), Const(1)]`    |                   | 沒有         |
| `BINARY_OP +`  | `[Tensor(z+1)]`            |                   | `add` 節點   |
| `RETURN_VALUE` | 結束                       |                   | 收圖         |

一個小細節：頭兩個推上去的是 `LazyVariableTracker`，不是 `TensorVariable`。Dynamo 對 input 是 lazy 的，`LOAD_FAST` 只是把一個「等到有人真的要用再說」的殼推上去，直到 `BINARY_OP` 真的要對它做運算，才把它實體化成 `TensorVariable`、裝上 Guard。這是為了少做白工：一個函式收了十個參數但只碰其中兩個，另外八個就永遠不用建替身、不用裝 Guard。

最後 `graph_code` 印出的圖，就是這八步裡真正留下來的東西：

```python
def forward(self, L_x_: "f32[8][1]cuda:0", L_y_: "f32[8][1]cuda:0"):
    z: "f32[8][1]cuda:0" = L_x_ * L_y_
    add: "f32[8][1]cuda:0" = z + 1
    return (add,)
```

看出了嗎：八條指令裡只有兩條往圖上加了節點。`STORE_FAST` 是一行 dict 賦值，`LOAD_CONST` 是推一個常數替身，`LOAD_FAST` 是查表推值。真正進圖的只有碰到 Tensor 運算的那兩條 `BINARY_OP`。

## Python 被吸收，Tensor 被記下

這就是 Dynamo 抽取計算圖的本質：Python 的語意被直譯器當場吸收，Tensor 的語意被記錄下來延後執行。再看一個更明顯的例子：

```python
def g(x):
    scale = 2
    parts = []
    for i in range(3):
        parts.append(x * (scale + i))
    return sum(parts)
```

這裡有 int 運算、有 list、有 `for` 迴圈、有 `append`、有內建的 `sum`。Dynamo 走完之後吐出的圖是：

```python
def forward(self, L_x_: "f32[8][1]cuda:0"):
    element   = L_x_ * 2
    element_1 = L_x_ * 3
    element_2 = L_x_ * 4
    value   = 0 + element
    value_1 = value + element_1
    value_2 = value_1 + element_2
    return (value_2,)
```

`scale + i` 在翻譯時就地算成 2、3、4；`for` 迴圈在翻譯時就地展開成三份；`parts` 這個 list 從頭到尾活在符號世界裡，圖上根本沒有它；`sum` 被 Dynamo 用一份 Python 寫的 polyfill 拆成三個 `add`。留在圖上的，只有純粹的 Tensor 運算。這就是為什麼 Dynamo 交給後端的圖那麼乾淨，也是為什麼它能吃下大量看起來很「Python」的程式：只要那些 Python 東西不需要 Tensor 的真值，直譯器就能自己消化掉。

反過來，`if` 要選邊、`for` 要展開，前提都是條件和次數不依賴 Tensor 的真值。一旦依賴了，handler 就走不下去。

## Handler 舉手的地方，就是 Graph Break

這個設計也直接解釋了 Graph Break 從哪來。昨天看到的那條訊息，`attempted to jump with TensorVariable()`，現在可以讀懂了：`POP_JUMP_IF_FALSE` 對到的 handler 是 `generic_jump` 生出來的，它彈出 stack 頂端，看是什麼。是常數就直接決定跳不跳；是 Tensor 替身，它沒辦法只靠符號知道真假，於是呼叫 `unimplemented_v2()` 舉手，觸發 Graph Break。

還有一個東西值得看一下：generic_jump 裡面那串判斷。CPython 遇到 if 很簡單，彈出 stack 頂端、bool() 一下就知道往哪跳。但 Dynamo 的 stack 上沒有真值，只有符號替身，它得看頂端是哪種替身來決定：

- Python 常數：替身背後包的就是一個真值，直接 `bool()` 它，當場就可以選邊走，翻譯繼續。這就是 if n \> 0 這種純 Python 條件在翻譯期就被寫死的原因。也因為寫死了，這裡會留下一條 guard，下次 n 換成別的值，guard 失敗就重編。
- TensorVariable：答案藏在 tensor 的數據裡，而確切的值要等真正執行才會有，翻譯期無論如何都拿不到。所以這就沒有別的辦法，直接呼叫 `jump_graph_break`，graph break。這就是「tensor 上的 if 會斷圖」這個最常見的原因。
- NNModuleVariable 或 list、dict 這類容器：對容器問真假，其實是問「你是不是空的」。容器裡有幾個元素，符號替身一直都在追蹤，翻譯期就知道答案，繼續。所以 if my_list: 這種寫法可以安全通過。\
  使用者自訂物件：照 Python 的規則找它的 `__bool__` 或 `__len__`，把那個方法本身也符號執行一遍，看返回的是誰。翻出來是常數就回到第一種情況，繼續；翻出來還是 Tensor 就回到第二種，一樣斷開。等於多繞一層，最後還是落回前兩條的判斷。

整串判斷其實只在問一件事：這個分支的答案，翻譯期拿得到嗎？拿得到就選邊繼續，拿不到就斷圖，把選擇權還給真正的執行。

所以簡單來說，Graph Break 不是查表失敗。`dispatch_table` 對幾乎每個 opcode 都有同名 handler，查表這一步不會落空。真正斷開的時間點在 handler 執行中：它接下指令、看了運算元，發現這個操作沒辦法用符號值走下去，才主動認輸。這就是為什麼你讀 Graph Break 訊息時，看到的永遠是「哪個操作、為什麼走不下去」，而不是「哪條指令不認識」。至於舉手之後怎麼收拾殘局、續行函式怎麼生、後面怎麼接回來，留到 Graph Break 那一篇再仔細說說。

## 結語

`InstructionTranslator` 是一台用 Python 寫的 CPython：同樣的 stack machine、同樣的指令集，`dispatch_table` 一條指令對一個 handler，`run()` 就是 `while step()`。差別只在 stack 上放的是 `VariableTracker` 而不是真值。

大部分 handler 只動 stack 和 `symbolic_locals`，Python 的邏輯在翻譯當下就被吸收掉：int 算掉、迴圈展開、list 留在符號世界。只有碰到 Tensor 運算的那幾條，才往 FX Graph 加節點。這是 Dynamo 圖那麼乾淨的原因，也是它能吃下大量 Python 的原因。

而 handler 做到一半發現只靠符號走不下去、主動舉手的那一刻，就是 Graph Break。

今天一直把 stack 裡裝的東西叫「符號替身」，目前只是含糊的帶過。明天就會來仔細鑽研它：`VariableTracker`。Dynamo 幫每一種 Python 值都準備了一種包法，Tensor、常數、list、函式、`nn.Module` 各有各的類別，各自決定「被呼叫時怎麼辦、被取屬性時怎麼辦」。理解了這套型別系統，你也會理解兩件事：為什麼呼叫自己寫的函式不會斷圖（它被 inline 進來了），以及 Python 的 int 是怎麼被 bake 進圖裡變成常數的。那我們明天見！

## 參考資料

- [torch/_dynamo/symbolic_convert.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/symbolic_convert.py)
- [torch/_dynamo/convert_frame.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/convert_frame.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [`dis` 模組文件：bytecode 指令語意](https://docs.python.org/3/library/dis.html)
- [torch.compile 的 logging 選項](https://pytorch.org/docs/stable/logging.html)
