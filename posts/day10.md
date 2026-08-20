# Day 10 | Graph Break：斷在哪裡，怎麼接回來

## 前言

Day 4 說過，Graph Break 不是查表失敗，是 handler 做到一半舉手。前面幾天把「乖乖翻完」的路走通了，今天看舉手之後發生的每一件事：一個函式怎麼被切成三段、resume function 怎麼做到「從函式中間開始跑」、為什麼深層函式裡的一個 `print` 會劈開最外層的圖。

正文開始！

![一次 break 把函式切成三段](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day10/graph_break.gif)

*圖一：翻譯走到 `print` 舉手之後，`f` 被切成三段：前半收成圖一交給 Inductor、`print` 那條指令留在 eager、剩下包成 `__resume_at_32_3` 再被 eval hook 攔截編成圖二。*

## 斷點做三件事

翻譯進行到某條指令，handler 發現眼前的東西建不了模，丟出 `Unsupported`。Dynamo 不放棄整個函式，它做三件事：

1. **收前半段**：到斷點為止的節點照 Day 8 的 `compile_subgraph` 收成一張圖，該結的帳（Day 7）結清。
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

`TORCH_LOGS="graph_breaks,graph_code,bytecode"` 一次跑出完整現場。先是 break 的原因：

```
Graph break in user code at graph_break.py:16
Graph Break Reason: Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print` ...
```

前半收成圖一（只有 `x * 2`），然後看改寫後的 bytecode 怎麼把三段接起來（節錄）：

```
MODIFIED BYTECODE f
  LOAD_GLOBAL  __compiled_fn_2          <- 圖一
  LOAD_FAST    x
  CALL         1
  STORE_FAST   graph_out_0
  LOAD_GLOBAL  __builtins_dict___1      <- 斷點指令回 eager
  LOAD_CONST   'print'
  BINARY_SUBSCR
  LOAD_CONST   'mid'
  ...
  CALL         1
  LOAD_GLOBAL  __resume_at_32_3         <- 剩下的包成 resume function
  LOAD_FAST    x
  CALL         2
  RETURN_VALUE
```

`__resume_at_32_3` 的意思是「從原函式第 32 個 byte 繼續」。接著 log 裡出現它自己的 ORIGINAL BYTECODE，名字叫 `torch_dynamo_resume_in_f_at_16`，開頭很有意思：

```
  LOAD_FAST    ___stack0
  JUMP_FORWARD 16 (to 38)      <- 直接跳進函式中間
  ...（原函式的完整 bytecode 跟在後面）
```

它的參數表是斷點當下所有還活著的狀態（stack 上的值、之後還會被讀的 locals），開場白把這些擺回原位，然後一條 `JUMP_FORWARD` 跳到斷點之後那條指令。合法的 Python 寫不出「從函式第 32 個 byte 開始跑」，但 bytecode 層沒有這個限制，[`resume_execution.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/resume_execution.py) 直接拼出來。然後 eval hook 攔截它，`return x + 1` 編成圖二。同一個斷點的 resume function 只生成一次，之後重用。

`dynamo.explain(f)(x)` 的摘要印證：`Graph Count: 2`、`Graph Break Count: 1`。

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

`explain` 的結果是三張圖（`mul`、`add`、`mul` 各一張），User Stack 指著兩層：

```
User Stack:
  <FrameSummary graph_break.py, line 39 in big>
  <FrameSummary graph_break.py, line 34 in util>
```

實務含義：深層 util 函式裡的一個 `print`，會把最外層大函式的圖劈碎。抓 break 時兇手常常不在報告的第一行，而在它呼叫的函式深處。

順帶一提中途爆炸的情況：`CALL` 往函式裡鑽到一半才發現翻不了，這時 stack 疊了一半、帳本記了一半，狀態是髒的。Dynamo 的解法是砍掉重練：丟出 `RestartAnalysis` 重翻一遍，第一遍爆炸前 `SpeculationLog` 已記下「走到第 N 條指令會失敗」，第二遍走到 N 就不往裡鑽，先把圖收乾淨。兩遍分析花的是編譯期的時間，換到斷點永遠落在乾淨的指令邊界上。

## 抓 break 的工具箱

Break 不會報錯，只會默默變慢，所以要主動抓：

- `torch.compile(f, fullgraph=True)`：把所有 break 升級成錯誤。實跑會直接丟 `Unsupported`，訊息跟 log 裡的一樣帶著分類和修法提示。上線前用它掃一輪最省事。
- `torch._dynamo.explain(f)(*args)`：不改行為，列出每張圖、每個 break 的位置和原因。
- `TORCH_LOGS="graph_breaks"`：執行時逐個印。

## 為什麼 break 貴

收個帳，一次 break 至少付四筆：圖被切小，跨不過斷點的 fusion 機會全沒了；中間那段 eager 本身慢；帳本被迫提前結算；resume function 還要多編一次。加上 Day 3 說過的，break 後每個 frame 都還是會繞進 Dynamo 判斷一次。所以效能調校的第一課永遠是：先數 break，再談其他。

## 結語

一次 break 把函式切成三段：前半編成圖一、斷點那條指令回 eager、剩下包成 resume function 再被攔截編成圖二。斷點必須落在有 frame 的地方，所以 inline 子函式裡的 break 會傳染到 caller 的那條 `CALL`。

到今天，Dynamo 的主線完整了：攔截、翻譯、包裝、驗票、記帳、收圖、寫碼、斷了再接。還剩最後一塊拼圖：Day 6 那條 `EQUALS_MATCH: L['n'] == 3` 太窄了，值一變就重編。明天講 Symbolic Shapes：SymInt 怎麼把具體的 4 換成符號 s0、ShapeEnv 怎麼管理符號之間的約束，讓一張圖吃下所有 batch size。那我們明天見！

## 參考資料

- [torch/_dynamo/resume_execution.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/resume_execution.py)
- [torch/_dynamo/exc.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/exc.py)
- [torch/_dynamo/symbolic_convert.py：SpeculationLog（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/symbolic_convert.py)
- [torch.compile 疑難排解：Graph Breaks](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
