# Day 3 | Dynamo 憑什麼攔得住你的 Python？

## 前言

昨天畫完 `torch.compile` 的四站地圖，第一站是 Dynamo，我們說它會「在 Python 執行的當下攔截你的 bytecode」。這句話講得很輕鬆，但仔細想其實很奇怪：一個第三方套件，憑什麼能插進 CPython 的執行過程，在你的函式要跑之前先接手？它不是 monkey patch，也沒有改你的原始碼，那它到底站在哪裡？

今天就來回答這個問題。答案是 CPython 官方留的一個洞：PEP 523 的 frame evaluation API。搞懂這個洞，你就會知道 Dynamo 為什麼能吃下幾乎任何 Python、為什麼選擇讀最底層的 bytecode，以及為什麼它總在某些地方不得不斷開。

正文開始！

![CPython 執行一個 frame 的兩條路](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day03/eval_frame.gif)

*圖一：CPython 執行一個 frame 的兩條路。預設走 `_PyEval_EvalFrameDefault` 一條一條跑 bytecode；`torch.compile` 把 `eval_frame` 指標換掉後，同一個 frame 改送進 Dynamo，錄成 FX Graph、記下 Guards、改寫 bytecode，再交回 CPython 執行。*

## CPython 平常怎麼跑一個函式

Python 的每一次函式呼叫，CPython 都會建一個 frame。Frame 裝著這次呼叫的所有狀態：區域變數、value stack、目前執行到第幾條指令。函式的邏輯不是以原始碼的形式在跑，而是先被編譯成 bytecode 存在 code object 裡，CPython 再一條一條執行。

用內建的 `dis` 就能把 bytecode 攤開來看：

```python
import dis
import torch

def f(x):
    return torch.sin(x) + 1

dis.dis(f)
```

在 Python 3.12 上會印出這樣：

```
 18           4 LOAD_DEREF               1 (torch)
              6 LOAD_ATTR                1 (NULL|self + sin)
             26 LOAD_FAST                0 (x)
             28 CALL                     1
             36 LOAD_CONST               1 (1)
             38 BINARY_OP                0 (+)
             42 RETURN_VALUE
```

讀法很直覺：把 `torch` 推上 stack，取 `sin` 屬性，把 `x` 推上去，呼叫一次，推常數 `1`，做加法，回傳。CPython 拿到一個 frame，就交給一個 C 函式 `_PyEval_EvalFrameDefault` 去跑，它就是那個大 switch 迴圈，一條一條指令讀過去、執行、更新 stack。這是 Python 的心臟。

## PEP 523：把「怎麼跑一個 frame」變成可以替換的

[PEP 523](https://peps.python.org/pep-0523/)（Python 3.6 引入）做的事只有一件：在 interpreter state 上加一個函式指標 `eval_frame`。CPython 要執行 frame 時，不再寫死呼叫 `_PyEval_EvalFrameDefault`，而是呼叫這個指標指到的東西。預設它就指向 default，行為完全不變。

但這代表一個 C extension 可以把這個指標換成自己的函式。從那一刻起，每一個 frame 要執行之前，CPython 都會先問這個自訂 evaluator：「這個 frame，你要怎麼跑？」

這就是 Dynamo 的立足點。它不改你的原始碼，不包裝你的函式物件，它是在比函式更低的那一層，攔下 CPython 對每個 frame 的執行請求。你可以在 PyTorch 裡直接找到這個開關：

```python
>>> torch._C._dynamo.eval_frame.set_eval_frame
<built-in function set_eval_frame>
```

它的實作在 [`torch/csrc/dynamo/eval_frame.c`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/eval_frame.c)，Python 這一側的包裝在 [`torch/_dynamo/eval_frame.py`](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/eval_frame.py)。`torch.compile` 回傳的那個物件，被呼叫時做的第一件事就是把 `eval_frame` 換成 Dynamo 的版本，函式跑完再換回來。所以 Dynamo 只在你呼叫 compiled function 的期間接管，其他時候 CPython 照舊。

## Dynamo 接手之後做什麼

當你 `torch.compile(f)` 之後第一次呼叫，`f` 的 frame 送到 Dynamo 的自訂 evaluator，它做的大致是：

1. 拿到這個 frame 的 code object 和 bytecode。
2. 符號式地執行 bytecode：不是真的算數字，而是拿 FakeTensor 之類的符號值走過每一條指令，把「這裡做了一個 `sin`、那裡做了一個 `add`」記錄成一張 FX Graph。
3. 同時記下這次能成立的前提，也就是 Guard，例如 `x` 是某個 dtype、某個 shape。
4. 把這張圖交給後端編譯，然後改寫這個 frame 的 bytecode：把原本那段運算換成「呼叫編譯好的產物」。
5. 把結果快取在 code object 上，key 是 Guard。下次同一個函式再進來，Guard 過就直接用改寫過的 bytecode，不用重跑 Dynamo。

所以 Dynamo 的「即時」不是它跑在旁邊監看，而是它就站在 CPython 執行每個 frame 的必經之路上。

## 親眼看它改寫 bytecode

第 4 步說它會「改寫 bytecode」，這不是比喻，`TORCH_LOGS="bytecode"` 就能把改寫前後兩份都印出來：

```python
torch._logging.set_logs(bytecode=True)
torch.compile(f)(torch.randn(8, device="cuda"))
```

改寫前就是上面 `dis` 看到的那七條。改寫後長這樣（節錄）：

```
MODIFIED BYTECODE f bytecode.py line 17
   4 LOAD_GLOBAL              3 (NULL + __compiled_fn_1_f0966bae_...)
  ...
  58 LOAD_FAST                0 (x)
  ...
 106 CALL                     1
 114 STORE_FAST               1 (graph_out_0)
 116 LOAD_FAST                1 (graph_out_0)
 118 LOAD_CONST               2 (0)
 120 BINARY_SUBSCR
 126 RETURN_VALUE
```

原本的 `LOAD_ATTR sin`、`CALL`、`BINARY_OP +` 全都不見了，換成 `LOAD_GLOBAL __compiled_fn_1_...`：把編譯好的產物當一個全域函式載進來，把 `x` 當參數呼叫它，拿回傳 tuple 的第 0 個元素回傳。整個 `f` 的計算被折疊成「呼叫一個編好的函式」。中間省略的幾條是 PyTorch 2.8 加的 `record_pregraph_bytecode_enter/exit`，只是給 profiler 用的標記，可以忽略。

這是驗證「Dynamo 確實動在 bytecode 層」最直接的方式。它沒有動你的 `.py` 檔，也沒有換掉 `f` 這個函式物件，`f.__code__` 還是原本那份；它是在 frame 要執行的那一刻，遞給 CPython 另一份 code object。

## 為什麼是 bytecode，不是原始碼或 AST

一個很自然的問題：要分析程式，為什麼不去讀原始碼、或解析 AST，反而挑最底層、最難讀的 bytecode？

因為 bytecode 是唯一保證存在的形式。原始碼可能根本拿不到（`exec` 出來的、lambda、被 decorator 包過的、`.pyc` 直接載入的），AST 也一樣。但只要函式能被 CPython 執行，它就一定有 code object、一定有 bytecode。在 frame 這一層動手，Dynamo 就能吃下幾乎任何來源的 Python，不管它是誰寫的、怎麼生出來的。這也是昨天講的 TorchScript 做不到的事：`torch.jit.script` 要解析原始碼，一遇到拿不到原始碼、或用了它不支援的語法就死。

代價是 Dynamo 得自己實作一個 bytecode 的符號直譯器，把 CPython 幾百條指令的語意在 Python 層重寫一遍，而且每個 Python 版本的 bytecode 都不太一樣（3.11 之後改動特別大）。這是 Dynamo 裡最厚重的一塊，明天會打開來看。

## 「幾乎任何」還是留了「幾乎」

Frame eval hook 讓 Dynamo 有機會看到每一條 bytecode，但看得到不等於走得動。有些指令它沒辦法用符號值走下去，最典型的是依賴實際數值的控制流：

```python
def g(x):
    if x.sum() > 0:
        return x * 2
    return x + 1
```

`x.sum() > 0` 要落地成一個 Python bool 才知道往哪跳，但 Dynamo 手上是符號 Tensor，它不知道該走哪條分支。這時它不會硬猜，而是斷開。用 `TORCH_LOGS="graph_breaks"` 看：

```
Graph break in user code at bytecode.py:35
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`).
               Dynamo does not support tracing dynamic control flow.
  Hint: Use `torch.cond` to express dynamic control flow.
  Developer debug context: attempted to jump with TensorVariable()
```

`dynamo.explain(g)(x)` 的摘要則是 `Graph Count: 2`、`Graph Break Count: 1`：`if` 之前能捕獲的部分收成第一張圖，控制權交還 CPython 去真的跑那個 `if`，等分支確定後再從後面接著捕獲第二張圖。最後那行 debug context 很有意思，`attempted to jump with TensorVariable()`，它說的是 Dynamo 的符號直譯器走到一條跳躍指令（`POP_JUMP_IF_FALSE`），發現 stack 頂端是一個 Tensor 而不是能判斷真假的常數，於是放棄。這正好是明天的主題。

同樣會逼它斷開的還有：呼叫進到它沒有符號模型的 C 函式、`.item()` 這種要把 Tensor 值抽成 Python 純量的操作、`print` 這類有副作用的東西。Frame eval hook 給了 Dynamo「看到一切」的能力，但「一切都能符號化」是另一回事，這條界線就是 Dynamo 這幾篇的主軸。

## 結語

Dynamo 的攔截點在 frame，不在函式。PEP 523 在 CPython 的 interpreter state 上留了一個 `eval_frame` 函式指標，Dynamo 把它換成自己的，從此每一個 frame 要執行前，CPython 都會先問過它。這就是為什麼它不用改你的原始碼、也不用包裝你的函式物件，卻能「在 Python 執行的當下」接手。

選 bytecode 是為了通用性，不是炫技。原始碼和 AST 都可能拿不到，但只要能被 CPython 執行就一定有 bytecode，所以 Dynamo 不挑來源。代價是它得自己寫一個 bytecode 的符號直譯器，而它會 Graph Break，正是因為某些指令沒辦法只靠符號值走下去。

還有一件事值得記住：`eval_frame` 是換在 interpreter state 上的，不是只掛在某個函式上。所以在 compiled function 執行期間，路徑上每一個 frame 都會先繞進 Dynamo 判斷一次，就算它決定這段不編譯、照原樣交回 default evaluator，也已經多繞了一趟。這是為什麼一支充滿大量小函式、或一直在斷圖的程式，用 `torch.compile` 反而可能變慢。

明天打開那個符號直譯器的本體：`InstructionTranslator`。Dynamo 手上沒有真值，它要怎麼一條一條走過 `LOAD_FAST`、`CALL`、`BINARY_OP`，還維護一個和 CPython 一模一樣的 stack？我們會對照 `torch/_dynamo/symbolic_convert.py`，看它怎麼一條指令一個 handler，走到哪、圖就長到哪。那我們明天見！

## 參考資料

- [PEP 523: Adding a frame evaluation API to CPython](https://peps.python.org/pep-0523/)
- [`dis` 模組文件](https://docs.python.org/3/library/dis.html)
- [torch/csrc/dynamo/eval_frame.c](https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/eval_frame.c)
- [torch/_dynamo/eval_frame.py](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/eval_frame.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.compile 的 logging 選項](https://pytorch.org/docs/stable/logging.html)
