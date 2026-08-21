# Day 10 | Graph Break：斷在哪裡，怎麼接回來

## 前言

Day 4 結尾說過一句話：Graph Break 不是查表失敗，是 handler 做到一半舉手。前面幾天把「乖乖翻完」的路走通了：攔截（Day 3）、翻譯（Day 4）、替身（Day 5）、驗票（Day 6）、記帳（Day 7）、收圖（Day 8）、寫碼（Day 9）。今天回頭走那條沒走的岔路，看舉手之後發生的每一件事：什麼樣的程式碼會踩到斷點、舉手在原始碼裡是一個什麼動作、一個函式怎麼被切成三段、resume function 怎麼做到「從函式中間開始跑」這種合法 Python 寫不出來的事、以及為什麼深層函式裡的一個 `print` 會劈開最外層的圖。

正文開始！

![函式時間軸在 print 斷開：前段編成圖一、print 掉進 eager、resume function 又被攔截編成圖二，再縫回一條執行路徑](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day10/graph_break.gif)

*圖一：`f` 的 bytecode 時間軸翻譯到 `print` 舉手之後被切成三段：前段收成圖一交給 Inductor、`print` 那條指令掉回 eager、剩下包成 `__resume_at_32_3` 再被 eval hook 攔截編成圖二，最後執行路徑把三段重新縫成一條。*

## 哪些程式碼會踩到斷點

先問一個問題：到底什麼會斷？Day 4 講過判斷發生的位置：`dispatch_table` 對幾乎每條 opcode 都有同名 handler，查表這一步不會落空；真正斷開的時機，是 handler 接下指令、看了運算元，發現這個操作沒辦法只靠符號值走下去。所以「會不會斷」不是指令說了算，是運算元說了算。同一條 `CALL`，呼叫 `torch.sin` 進圖、呼叫自己寫的函式被 inline、呼叫 `print` 就斷。實務上最常撞到的幾類：

| 類別 | 例子 | 為什麼走不下去 |
| --- | --- | --- |
| 依賴資料的控制流 | `if x.sum() > 0:` | 要有真值才知道往哪跳，符號 Tensor 沒有真假 |
| 把 Tensor 值抽成 Python 純量 | `.item()`、`int(t)`、`.tolist()` | 值要離開圖回到 Python 世界，預設不追 |
| 有 side effect 的 builtin | `print`、`input` | 效果發生在真實世界，圖裡建不了模 |
| 沒有符號模型的 C 函式 | 第三方套件的 C extension | 進不去 bytecode，也沒有對應的 handler |

第一類 Day 3 親眼看過：`generic_jump` 彈出 stack 頂端發現是 `TensorVariable`，丟出 `attempted to jump with TensorVariable()`。第二類有個開關 `torch._dynamo.config.capture_scalar_outputs`，打開之後 `.item()` 不再斷，而是變成一個沒有具體值的符號，代價留到明天 Symbolic Shapes 再算。第三類就是今天的主角。第四類則對應 Day 5 講過的 `trace_rules.py` 名單：名單上說跳過的函式，呼叫它就是潛在的斷點。

## 舉手的姿勢：Unsupported 例外

handler 舉手不是回傳一個錯誤碼，是丟一個例外。[`exc.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/exc.py) 裡定義了 `Unsupported`，所有 handler 統一經過 `unimplemented_v2()` 把它丟出來，而且丟的時候要交四樣東西：`gb_type`（分類）、`context`（現場）、`explanation`（解釋）、`hints`（修法建議）。這就是為什麼你看到的每一條 graph break 訊息都長同一個樣子，分類、解釋、提示一應俱全，等一下的實驗會看到實例。

丟出來之後誰接？[`symbolic_convert.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/symbolic_convert.py) 裡有個 decorator 叫 `break_graph_if_unsupported`，包住 `CALL` 這類最容易出事的 handler。接住 `Unsupported` 之後分兩條路：`fullgraph=True` 的話不收拾，原樣往上丟，變成使用者直接看到的錯誤；否則進入 graph break 的收拾流程。

收拾流程有個麻煩：例外飛出來的時候，翻譯常常做到一半。`CALL` 往被 inline 的函式裡鑽到一半才發現翻不了，這時 stack 疊了一半、Day 7 的帳本記了一半，狀態是髒的，直接在這裡切圖會切出錯的東西。Dynamo 的解法是砍掉重練：丟出 `RestartAnalysis` 重翻一遍，第一遍爆炸前 `SpeculationLog` 已經記下「走到第 N 條指令會失敗」，第二遍走到 N 就不往裡鑽，先把圖收乾淨、把斷點擺在乾淨的指令邊界上。兩遍分析花的是編譯期的時間，換到的是斷點永遠落在狀態一致的地方。這件事在 `dynamo.explain` 的 Compile Times 裡看得到痕跡：等下的實驗只編了一次，卻同時出現 `compile_attempt_0` 和 `compile_attempt_1` 兩筆。

## 斷點做三件事

收拾流程本身做三件事。翻譯進行到某條指令、`Unsupported` 被接住之後，Dynamo 不放棄整個函式，它：

1. **收前半段**：到斷點為止的節點照 Day 8 的 `compile_subgraph` 收成一張圖，該結的帳（Day 7）結清，交給後端編譯。
2. **讓那條指令回 eager**：生成的 bytecode 裡，斷點指令原樣保留，讓 CPython 自己跑。
3. **把剩下的包成 resume function**：斷點之後的 bytecode 被包成一個新函式，用 Day 9 的工具箱直接生出來。

關鍵在第三步的後續：resume function 也是函式，一被呼叫，Day 3 的 eval hook 照樣攔截它，於是斷點之後的程式碼編成第二張圖。一次 break 的結果是：兩張圖，中間夾一小段 eager。

## 動手看一次

```python
def f(x):
    x = x * 2
    print("mid")
    return x + 1
```

`TORCH_LOGS="graph_breaks,graph_code,bytecode"` 一次跑出完整現場。先看還沒動過手腳的原始 bytecode，`print` 那條 `CALL` 在 offset 24、它算完的下一條 `POP_TOP` 在 offset 32，這兩個數字待會都會再出現：

```
ORIGINAL BYTECODE f
 15           2 LOAD_FAST     0 (x)
              4 LOAD_CONST    1 (2)
              6 BINARY_OP     5 (*)
             10 STORE_FAST    0 (x)
 16          12 LOAD_GLOBAL   1 (NULL + print)
             22 LOAD_CONST    2 ('mid')
             24 CALL          1
             32 POP_TOP
 17          34 LOAD_FAST     0 (x)
             36 LOAD_CONST    3 (1)
             38 BINARY_OP     0 (+)
             42 RETURN_VALUE
```

翻譯走到 offset 24 的 `CALL`，handler 舉手，log 印出的正是 `unimplemented_v2` 那四件套：

```
Graph break in user code at graph_break.py:16
Graph Break Reason: Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print`
               with argument types ['str'] (has_kwargs False)
  Hint: Avoid calling builtin `print` with argument types ['str']. Consider
        using an equivalent alternative function/method to `print`.
  Hint: If you are attempting to call a logging function (e.g. `print`), you
        can try adding it to `torch._dynamo.config.reorderable_logging_functions`.
```

順帶記下第二條 Hint：如果只是想留 log，把 `print` 加進 `reorderable_logging_functions`，Dynamo 會把這類呼叫挪到圖跑完再執行，圖就不用斷。這是官方給的正規修法之一。

前半收成圖一，內容只有 `x * 2`：

```python
 ===== __compiled_fn_2 =====
def forward(self, L_x_: "f32[4][1]cuda:0"):
    x: "f32[4][1]cuda:0" = l_x_ * 2
    return (x,)
```

然後看改寫後的 bytecode 怎麼把三段接起來（節錄，profiler 標記已省略）：

```
MODIFIED BYTECODE f
   2 LOAD_GLOBAL   5 (NULL + __compiled_fn_2_...)   <- 圖一
  56 LOAD_FAST     0 (x)
 104 CALL          1
 112 STORE_FAST    1 (graph_out_0)
 116 LOAD_GLOBAL   2 (__builtins_dict___1)          <- 斷點指令回 eager
 126 LOAD_CONST    4 ('print')
 128 BINARY_SUBSCR
 132 LOAD_CONST    2 ('mid')
 134 LOAD_FAST     1 (graph_out_0)
 136 LOAD_CONST    5 (0)
 138 BINARY_SUBSCR
 142 STORE_FAST    0 (x)                            <- 圖一的輸出放回 x
 146 CALL          1
 154 LOAD_GLOBAL  13 (NULL + __resume_at_32_3_...)  <- 剩下包成 resume fn
 172 LOAD_FAST     0 (x)
 174 CALL          2
 182 RETURN_VALUE
```

三段的接縫都在這裡了：先呼叫 `__compiled_fn_2` 把前半算完、輸出放回區域變數 `x`；接著把 `print` 從 builtins 撈出來、CPython 真的呼叫它；最後呼叫 `__resume_at_32_3`，把 `print` 的回傳值和 `x` 一起傳進去。`__resume_at_32_3` 這個名字的意思是「從原函式第 32 個 byte 繼續」，正是上面 `POP_TOP` 的位置：斷點指令自己（`CALL print`）留在外面跑，它之後的第一條指令才是續集的起點。

## resume function：從函式中間開始跑

接著 log 裡出現 resume function 自己的 ORIGINAL BYTECODE，code object 的名字叫 `torch_dynamo_resume_in_f_at_16`（f 的第 16 行）：

```
ORIGINAL BYTECODE torch_dynamo_resume_in_f_at_16
 16           0 RESUME         0
              2 LOAD_FAST      0 (___stack0)
              4 JUMP_FORWARD  16 (to 38)
              6 RESUME         0
              8 LOAD_FAST      1 (x)
             ...（原函式的 bytecode 原樣跟在後面）
        >>   38 POP_TOP
 17          40 LOAD_FAST      1 (x)
             42 LOAD_CONST     3 (1)
             44 BINARY_OP      0 (+)
             48 RETURN_VALUE
```

它的參數表是斷點當下所有還活著的狀態：stack 上的值（`___stack0`，這裡是 `print` 的回傳值）加上之後還會被讀的 locals（`x`）。開場白把 `___stack0` 擺回 stack 原位，然後一條 `JUMP_FORWARD` 直接跳到 offset 38，也就是斷點之後的第一條指令，前面那段原函式的 bytecode 只是原樣帶著、永遠不會被執行。合法的 Python 寫不出「從函式第 32 個 byte 開始跑」這種函式，但 bytecode 層沒有這個限制，[`resume_execution.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/resume_execution.py) 就直接拼出來。同一個斷點的 resume function 只生成一次，之後重用。

然後劇本重演一遍：resume function 一被呼叫，eval hook 攔截它，`return x + 1` 編成圖二：

```python
 ===== __compiled_fn_5 =====
def forward(self, L_x_: "f32[4][1]cuda:0"):
    add: "f32[4][1]cuda:0" = l_x_ + 1
    return (add,)
```

`dynamo.explain(f)(x)` 的摘要印證整件事：

```
Graph Count: 2
Graph Break Count: 1
Op Count: 2
Ops per Graph:
  Ops 1: <built-in function mul>
  Ops 2: <built-in function add>
```

兩張圖各領一個 op，中間夾著那聲 `mid`。

## break 會傳染：inline 裡的斷點

被 inline 的函式（Day 5）內部發生 `Unsupported` 呢？Dynamo 不能在子函式裡斷：子函式沒有自己的 frame，斷在裡面沒有地方可以 resume。所以 inline 途中的斷點會讓整個 call 在 caller 端變成 break 點，一層層往上傳，直到真正有 frame 的邊界為止。

```python
def util(t):
    print("log")
    return t + 1

def big(x):
    y = x * 2
    z = util(y)
    return z * 3
```

翻譯 `big` 時，Dynamo 鑽進 `util` 撞到 `print`，於是 `z = util(y)` 這整條 `CALL` 在 `big` 裡變成斷點：前半收成圖一（`mul`），`util(y)` 改由 CPython 真的呼叫。但 `util` 自己是一個 frame，eval hook 又攔截它，在它自己的 frame 裡再斷一次（`print` 前面沒有任何 Tensor 運算，所以沒有圖）、再 resume（`t + 1` 收成圖二）。最後 `big` 的 resume function 把 `z * 3` 收成圖三。`explain` 的結果印證：

```
Graph Count: 3
Graph Break Count: 2
Op Count: 3
User Stack:
  <FrameSummary graph_break.py, line 39 in big>
  <FrameSummary graph_break.py, line 34 in util>
```

三張圖（`mul`、`add`、`mul` 各一張）、兩次 break，User Stack 指著兩層：斷點被回報在 `big` 的第 39 行，但兇手在 `util` 的第 34 行。Out Guards 也很有戲：`big` 那張圖多了一條 `L['util']` 的 `CLOSURE_MATCH`（inline 過的函式要守身分，Day 6 講過）；圖二守的是 `L['t']`（`util` 自己 frame 的參數）；圖三守的是 `L['___stack0']`，resume function 的參數表就這樣出現在 Guard 樹裡。

實務含義：深層 util 函式裡的一個 `print`，會把最外層大函式的圖劈碎。抓 break 時兇手常常不在報告的第一行，而在它呼叫的函式深處。

## 抓 break 的工具箱

Break 不會報錯，只會默默變慢，所以要主動抓：

- `torch.compile(f, fullgraph=True)`：把所有 break 升級成錯誤。前面說過 `break_graph_if_unsupported` 在這個模式下不收拾、直接往上丟，實跑就是一個 `Unsupported` 例外，訊息跟 log 裡的一模一樣：

  ```
  Unsupported : Failed to trace builtin operator
    Explanation: Dynamo does not know how to trace builtin operator `print`
                 with argument types ['str'] (has_kwargs False)
    Hint: Avoid calling builtin `print` with argument types ['str']. ...
  ```

  上線前用它掃一輪最省事，修到能跑等於保證整個函式一張圖。
- `torch._dynamo.explain(f)(*args)`：不改行為，列出每張圖、每個 break 的位置、原因和 User Stack，適合拿來看全貌。
- `TORCH_LOGS="graph_breaks"`：執行時逐個印，適合掛在長跑的 job 上。

抓到之後怎麼修，訊息裡的 Hint 通常已經給了方向：`print` 移出熱路徑或加進 `reorderable_logging_functions`；`.item()` 考慮 `capture_scalar_outputs`；依賴資料的 `if` 用 `torch.cond` 改寫（Day 3 的 Hint 就這麼說）；第三方 C 函式則把它挪出被編譯的函式，讓斷點斷在便宜的地方。

## 為什麼 break 貴

收個帳，一次 break 至少付四筆：圖被切小，跨不過斷點的 fusion 機會全沒了；中間那段 eager 本身慢；Day 7 的帳本被迫提前結算；resume function 還要多編一次（上面 `util` 的例子，一個 `print` 換來四次編譯：`big`、`util`、兩個 resume）。加上 Day 3 說過的，compiled function 執行期間每個 frame 都要繞進 Dynamo 判斷一次，圖越碎、繞的次數越多。所以效能調校的第一課永遠是：先數 break，再談其他。

## 結語

一次 break 把函式切成三段：前半編成圖一、斷點那條指令回 eager、剩下包成 resume function 再被攔截編成圖二。舉手是一個 `Unsupported` 例外，`break_graph_if_unsupported` 接住它、必要時 `RestartAnalysis` 重翻一遍，讓斷點永遠落在乾淨的指令邊界上。斷點必須落在有 frame 的地方，所以 inline 子函式裡的 break 會傳染到 caller 的那條 `CALL`，連鎖出一串圖和 resume function。

到今天，Dynamo 的主線完整了：攔截、翻譯、包裝、驗票、記帳、收圖、寫碼、斷了再接。還剩最後一塊拼圖：Day 6 那條 `EQUALS_MATCH: L['n'] == 3` 太窄了，值一變就重編。明天講 Symbolic Shapes：SymInt 怎麼把具體的 4 換成符號 s0、ShapeEnv 怎麼管理符號之間的約束，讓一張圖吃下所有 batch size。那我們明天見！

## 參考資料

- [torch/_dynamo/resume_execution.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/resume_execution.py)
- [torch/_dynamo/exc.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/exc.py)
- [torch/_dynamo/symbolic_convert.py：break_graph_if_unsupported、SpeculationLog（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/symbolic_convert.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.compile 疑難排解：Graph Breaks](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
