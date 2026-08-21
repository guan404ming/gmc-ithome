# Day 5 | 變數的替身使者？ VariableTracker 登場

## 前言

昨天 `InstructionTranslator` 的 stack 上面放的東西，一直被含糊地叫「符號替身」。今天準備來把它講清楚。原始碼位置在 [`torch/_dynamo/variables/`](https://github.com/pytorch/pytorch/tree/v2.8.0/torch/_dynamo/variables) 底下，Dynamo 幫追蹤時可能碰到的每一種 Python 值都準備了一個包裝類別，統稱 `VariableTracker`。這套型別系統，就是 Dynamo 對 Python 來說到底有多動態的一個完整的回答。

搭配它的還有一個小東西叫 `Source`，記的是「這個值在 runtime 要怎麼拿到」。理解這兩個概念，你就會懂三件昨天留下的懸案：Python 的 int 為什麼會被 bake 進圖變成常數、呼叫自己寫的函式為什麼不會 Graph Break、以及 Guard 到底是從哪裡長出來的。

正文開始！

## 為什麼需要包起來？

Dynamo 在翻譯 bytecode 的時候手上沒有真值，但指令的行為偏偏會取決於值的型別：同樣一條 `CALL`，呼叫的是 `torch.sin` 還是你自己寫的函式，處理方式天差地遠；同樣一條 `BINARY_OP *`，兩邊是 Tensor 還是 int，一個要進圖而另一個則需要當場算掉。所以每個值都需要一個「知道自己是什麼」的替身，handler 才能問它：你被乘的時候怎麼辦？你被呼叫的時候怎麼辦？你被取 attribute 的時候怎麼辦？

這些問題就是 `VariableTracker` 的介面，定義在 [`variables/base.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/base.py)：


| 方法                                    | 回答的問題                   |
| ------------------------------------- | ----------------------- |
| `call_function(tx, args, kwargs)`     | 我被呼叫時怎麼辦                |
| `call_method(tx, name, args, kwargs)` | 我的方法被呼叫時怎麼辦             |
| `var_getattr(tx, name)`               | 我被取屬性時怎麼辦               |
| `as_proxy()`                          | 給出進圖用的 FX proxy         |
| `as_python_constant()`                | 如果我其實是個常數，把真值交出來        |
| `reconstruct(codegen)`                | 改寫後的 bytecode 要怎麼把我重建出來 |


每個子類別用自己的方式回答。昨天看到的 `stack_op(operator.mul)` 就是把運算元交給一個 `BuiltinVariable`，由它去問兩邊的替身「你是什麼」，再決定要往圖上加節點還是直接算。

## 家族總覽

`variables/` 底下有幾十個檔案、上百個類別，但實務上最常撞到的就這幾個，它們的行為差異基本上就可以被視為 Dynamo 行為的縮影：


| Python 值           | 包成                                                 | 關鍵行為                  |
| ------------------ | -------------------------------------------------- | --------------------- |
| Tensor             | `TensorVariable`                                   | 帶著 FX proxy，運算會往圖裡加節點 |
| int、float、str、None | `ConstantVariable`                                 | 直接以常數身分被 bake 進圖，不佔輸入 |
| list、tuple         | `ListVariable`、`TupleVariable`                     | 元素各自被包，操作在符號世界模擬      |
| dict               | `ConstDictVariable`                                | 同上，key 查找在翻譯期完成       |
| 你寫的函式              | `UserFunctionVariable`                             | 被呼叫時 inline 進來繼續翻譯    |
| `torch.*` 函式       | `TorchInGraphFunctionVariable`                     | 被呼叫時往圖裡加節點            |
| `nn.Module`        | `NNModuleVariable`、`UnspecializedNNModuleVariable` | 屬性鏈被追蹤，參數變成圖的輸入       |
| 其他物件               | `UserDefinedObjectVariable`                        | 逐屬性追蹤，能走多遠走多遠         |
| 還沒被碰過的值            | `LazyVariableTracker`                              | 先掛個殼，真的被用到才實體化        |


這裡有兩個沒出現在下面實驗裡，不過那算是一定會碰到的，這邊還是先特別拉出來簡單講一下。

- `TorchInGraphFunctionVariable` 包的是 `torch.sin`、`torch.matmul` 這類 PyTorch 自己的函式，它的 `call_function` 就是往圖上加一個節點，Day 3 那個 `torch.sin(x)` 就是走這條路進圖的。
- `NNModuleVariable` 包的是你的 `nn.Module`：追蹤 `self.linear(x)` 時，Dynamo 順著屬性鏈找到 `self.linear.weight`，把它登記成圖的輸入而不是常數，所以權重更新不會觸發重編。

接下來這邊我們就來拿一個把這些都用上的函式去實際跑跑看會發生什麼事：

```python
class Config:
    scale = 2

cfg = Config()

def helper(t):
    return t * 3

def f(x, n, items, cfg=cfg):
    y = helper(x)
    return y * n + items[0] + cfg.scale

torch._logging.set_logs(trace_bytecode=True)
torch.compile(f)(x, 3, [1, 2])
```

`trace_bytecode` 印出來的 stack，每一格就是一個 `VariableTracker`（節錄）：

```
TRACE LOAD_DEREF helper     [NullVariable]
TRACE LOAD_FAST x           [NullVariable, LazyVariableTracker()]
TRACE CALL 1                [NullVariable, LazyVariableTracker(), LazyVariableTracker()]
TRACE RESUME 0              []                        <- 進到 helper 裡面了
TRACE LOAD_FAST t           []
TRACE LOAD_CONST 3          [TensorVariable()]
TRACE BINARY_OP 5           [TensorVariable(), ConstantVariable(int: 3)]
TRACE RETURN_VALUE None     [TensorVariable()]        <- helper 翻完，回到 f
TRACE STORE_FAST y          [TensorVariable()]
...
TRACE LOAD_FAST n           [TensorVariable()]
TRACE BINARY_OP 5           [TensorVariable(), LazyVariableTracker()]
TRACE LOAD_FAST items       [TensorVariable()]
TRACE LOAD_CONST 0          [TensorVariable(), LazyVariableTracker()]
TRACE BINARY_SUBSCR None    [TensorVariable(), LazyVariableTracker(), ConstantVariable(int: 0)]
TRACE LOAD_FAST cfg         [TensorVariable()]
TRACE LOAD_ATTR scale       [TensorVariable(), LazyVariableTracker()]
```

這邊有三件事值得放大來一起看看。

## ConstantVariable：Python int 被 bake 進圖

`f` 收了三個非 Tensor 的參數：`n=3`、`items=[1, 2]`、`cfg.scale=2`。看 Dynamo 吐出的圖：

```python
def forward(self, L_x_: "f32[8][1]cuda:0"):
    # File: vt.py:21 in helper, code: return t * 3
    y = L_x_ * 3
    # File: vt.py:25 in f, code: return y * n + items[0] + cfg.scale
    mul_1 = y * 3
    add = mul_1 + 1
    add_1 = add + 2
    return (add_1,)
```

圖的輸入只有 `L_x_` 一個。`n` 不見了，變成 `y * 3` 裡的那個 3；`items[0]` 不見了，變成 `+ 1`；`cfg.scale` 不見了，變成 `+ 2`。這就是 `ConstantVariable` 的行為：**Python 純量不當輸入，直接以常數身分寫進圖。**

圖因此可以被更好的最佳化，常數可以摺疊、可以特化，Inductor 生 kernel 時 `* 3` 是一個立即數（Immediate）而不是一次記憶體讀取。不過這邊的代價就會是這張圖只對 `n == 3`、`items[0] == 1`、`cfg.scale == 2` 這些條件成立下才會成立。那這邊由誰來記這筆帳呢？答案就是 Guard，下面會看到。

而 Tensor 的數值反過來，則會永遠留作 runtime 輸入，Dynamo 只特化 shape 和 dtype。這算是一個關於「變化頻率」的賭注：Tensor 的數值每個 batch 都有可能不同，bake 進圖的話等於每次呼叫都重編，划不來；int 純量常直接決定生成程式碼的長相（迴圈邊界、索引、分支），bake 進去收益大，而 Dynamo 賭它通常不會變。賭輸了就是一次 Recompile。同一個位置變太多次，Dynamo 會自動把它升級成 Symbolic Integer 停損，簡單來說就是標記這個變數是個改改怪，詳細解釋的話會留到 Symbolic Shapes 那篇。

## UserFunctionVariable：呼叫函式不是斷點，而是入口

再看 trace 裡 `CALL 1` 之後那幾行：stack 突然清空、出現 `RESUME`、`LOAD_FAST t`，這是 `helper` 的 bytecode。Dynamo 碰到呼叫你自己寫的函式，不會斷圖，而是開一個 `InliningInstructionTranslator`（昨天那台大機器的子類別，同一份 `symbolic_convert.py`），鑽進被呼叫函式的 bytecode 繼續翻譯，翻完把回傳值接回原本的 stack。所以圖上 `y = L_x_ * 3` 那行的來源註解是 `in helper`，不是 `in f`：不過最後整條呼叫鏈被會攤平成一張圖。

至於哪些函式該鑽進去、哪些該跳過（例如 numpy 的內部、標準函式庫的某些部分），由 [`trace_rules.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/trace_rules.py) 的名單決定。跳過的那些，就是可能 Graph Break 的地方。

順帶一提 `LazyVariableTracker`：`x`、`n`、`items`、`cfg` 剛被 `LOAD_FAST` 推上 stack 時全都是用它來包起來。主要是因為包裝本身其實也有成本（要建 Guard、要查型別、要遞迴包元素），不過在一開始很難判斷一個從沒被碰過的值不值得展開，所以先掛個懶惰的殼，真的被用到那一刻才實體化。`x` 是在進到 `helper` 裡碰到 `BINARY_OP` 才變成 `TensorVariable` 的。這邊的編譯時間就是這樣一點一點省下來的。

## Source：這個值是從哪裡來的

`VariableTracker` 還有一個蠻關鍵欄位 `source`，主要是用來回答另一個問題：這個值在 runtime 要怎麼拿到。定義在 [`source.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/source.py)，用起來像這樣：

```python
from torch._dynamo.source import AttrSource, GetItemSource, GlobalSource, LocalSource

LocalSource("x").name()                          # L['x']
AttrSource(GlobalSource("cfg"), "scale").name()  # G['cfg'].scale
GetItemSource(LocalSource("items"), 0).name()    # L['items'][0]
```

根節點只有兩種：`LocalSource`（frame 的區域變數，印成 `L[...]`）和 `GlobalSource`（全域，印成 `G[...]`），後面用 `AttrSource`、`GetItemSource` 一路鏈下去，任何值的取得路徑都能去做表示。這條鏈主要會有兩個用途，正好對應 Dynamo 的兩個出口：

- `source.make_guard(...)`：生成 Guard。`cfg.scale` 被 bake 成常數 2，前提「`L['cfg'].scale == 2`」就得記下來，下次呼叫時檢查。
- `source.reconstruct(codegen)`：生成 bytecode。改寫後的 bytecode 要把圖的輸入準備好，靠的就是發出 `LOAD_FAST cfg`、`LOAD_ATTR scale` 這樣的指令把值從這條我們剛剛在讀取的時候建立的鏈一個一個取出來。

`TORCH_LOGS="guards"` 印出來的東西，就是每個 Source 對應的 Guard（節錄）：

```
+- GuardManager: source=L['n']
| +- EQUALS_MATCH: L['n'] == 3
+- GuardManager: source=L['x']
| +- TENSOR_MATCH: check_tensor(L['x'], Tensor, ..., torch.float32, device=0, size=[8], stride=[1])
+- GuardManager: source=L['cfg']
| +- TYPE_MATCH: ___check_type_id(L['cfg'], 94810117705760)
| +- GuardManager: source=L['cfg'].scale
| | +- EQUALS_MATCH: L['cfg'].scale == 2
+- GuardManager: source=L['items']
| +- TYPE_MATCH: ___check_type_id(L['items'], 22635973059264)
| +- LENGTH_CHECK: len(L['items']) == 2
| +- GuardManager: source=L['items'][0]
| | +- EQUALS_MATCH: L['items'][0] == 1
+- GuardManager: source=L['helper']
| +- GuardManager: source=L['helper'].__code__
| | +- ID_MATCH: ___check_obj_id(L['helper'].__code__, 22635932715296)
```

每一行左邊的 `source=` 就是 Source 的 `name()`。看得出前面記下來的帳全在這裡：`n`、`items[0]`、`cfg.scale` 三個被 bake 的常數各有一條 `EQUALS_MATCH`；`x` 因為是 Tensor 所以只關注 dtype、shape、stride，而不去管值（`TENSOR_MATCH`）；被 inline 的 `helper` 則是會關注它的 `__code__` 物件 id，你重新定義 `helper` 它就會發現。實際改一下值：

```
g = torch.compile(f)
g(x, 3, [1, 2])
g(x, 4, [1, 2])          # Recompiling ... - 0/0: n == 3
cfg.scale = 5
g(x, 4, [1, 2])          # Recompiling ... - 0/1: cfg.scale == 2
```

`TORCH_LOGS="recompiles"` 會告訴你是哪條 Guard 沒過。這就是 bake 常數的代價被兌現的時刻。

## 兩個工廠：有沒有 Source，待遇不同

包裝的入口是 `VariableTracker.build(tx, value, source=...)`，`base.py` 裡就這幾行：

```python
@staticmethod
def build(tx, value, source=None):
    if source is None:
        return builder.SourcelessBuilder.create(tx, value)
    else:
        return variables.LazyVariableTracker.create(value, source)
```

按有無 Source 分流到兩個工廠，都在 [`variables/builder.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/builder.py)：

- `VariableBuilder`（經由 `LazyVariableTracker` 延後呼叫）：有 Source 的值，也就是從外面世界進來的：參數、全域、屬性。這些值下次呼叫可能變，所以包裝的同時要裝 Guard。
- `SourcelessBuilder`：追蹤中途生出來的值，例如翻譯期算出的 `scale + i`、剛建的空 list。它們不從外面來，下次呼叫會被一模一樣地重新生出來，不需要 Guard。

這邊其實就可以看出來：**Guard 關注的是圖跟外面世界的邊界，不是圖的內部發生的事。**  
  
所有最後整個流程大概就會長下面這樣：  
  
![每個進來的值被包成 VariableTracker，再決定進圖、bake 或 inline](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day05/variable_tracker.gif)

*圖一：`f(x, n, items, cfg)` *的每個輸入怎麼被包起來的。左邊是進來的值和它的 Source，中間是對應的* `VariableTracker` *與它回答問題的方式，右邊是結果：Tensor 進圖當輸入，int、list 元素、物件屬性被 bake 成常數，自己寫的函式被 inline，每一列都附帶一條 Guard。*

## 結語

`VariableTracker` 是 Dynamo 的型別系統：每一種 Python 值都有一個知道「自己被呼叫、被取屬性、被運算時該怎麼辦」的替身使者。Tensor 進圖，純量 bake 成常數，你寫的函式被 inline 攤平，其他物件逐屬性追蹤。`Source` 是每個替身身上的地址，記著它在 runtime 怎麼拿，一頭生出 Guard，一頭生出重建它的 bytecode。有 Source 的值走 `VariableBuilder` 並裝 Guard，追蹤中途生出的值就單純走 `SourcelessBuilder`不會裝上 Guard。

`cfg.scale` 也就是常數被 bake 成常數的代價，是一個「`L['cfg'].scale` 必須還是 2」的前提。今天我們看到這個前提從哪裡長出來，明天就會來看看它是長什麼樣子、怎麼樣被檢查：Guard 有哪幾種、`TORCH_LOGS=guards` 那棵樹的每一行怎麼讀，以及為了讓每次呼叫的檢查夠快，Dynamo 這邊做了什麼酷酷的騷操作。那一樣我們明天見！

## 參考資料

- [torch/_dynamo/variables/base.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/base.py)
- [torch/_dynamo/variables/builder.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/builder.py)
- [torch/_dynamo/source.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/source.py)
- [torch/_dynamo/trace_rules.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/trace_rules.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)

