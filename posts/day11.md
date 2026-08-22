# Day 11 | 一張圖吃下所有 batch size？TorchDynamo 的伸縮量尺 Symbolic Shapes

## 前言

Day 5 說 Python int 被 bake 成常數是一場賭注，Day 6 說賭輸的代價是重編，結尾還埋了一個伏筆，`n` 變了一次之後，`x` 的 size Guard 從 `[4, 4]` 變成了 `[None, 4]`。把 shape 押死的 Guard 實在太窄，batch size 一變就得重編一次，這就是昨天說的最後一塊拼圖。

今天就來把這塊拼圖補上。Guard 原本用的是一把押死的量尺，量到 4 就是 4，差一格都不行。Symbolic Shapes 給了它一把伸縮量尺，把具體的 4 換成符號 `s0`，讓一張圖吃下所有 batch size。原始碼主要住在 [`torch/fx/experimental/symbolic_shapes.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py)，這是 PyTorch 裡最長的檔案之一，今天會把它的核心概念一次講完。

正文開始！

## static shape 哪裡不夠用

Day 6 看過，每個 Tensor 輸入都吃一條 `TENSOR_MATCH`，dtype、device、shape、stride 全部押住。shape 押死有很實際的好處。Inductor 生 kernel 時，迴圈邊界是具體數字，可以完全展開、可以按大小挑演算法，特化的圖就是比較好最佳化，這是 Day 5 講過的紅利。

但推論服務的 batch size 會跟著流量變、NLP 的序列長度每個 batch 都不同。如果每種 shape 都特化一張圖，Guard 樹越掛越長，撞上 `recompile_limit`（預設 8）之後整個 frame 退回 eager，編譯的收益直接歸零。

Dynamo 的解法不是把量尺丟掉，而是兩把都留著，用一套升級策略決定用哪把。

- **預設 static**，`assume_static_by_default` 預設為 `True`，第一次編譯所有 shape 按具體值特化，享受特化的紅利。
- **被逼才 dynamic**，同一個維度第二次出現不同的值，`automatic_dynamic_shapes`（預設也是 `True`）就把它升級成符號，這次編出來的圖從此吃下這個維度的所有值。

賭注還是照下，只是輸過一次的注不會再下第二次。接下來實際看升級流程跑起來的樣子。

## 實際看一次 automatic dynamic

拿最小的例子 `f(x, y) = x @ y`，`y` 固定 `(4, 8)`，`x` 的 batch 維一路換。

```python
def f(x, y):
    return x @ y

g = torch.compile(f)
y = torch.randn(4, 8, device="cuda")
g(torch.randn(4, 4, device="cuda"), y)      # call 1
g(torch.randn(8, 4, device="cuda"), y)      # call 2
g(torch.randn(16, 4, device="cuda"), y)     # call 3
g(torch.randn(100, 4, device="cuda"), y)    # call 4
```

第一次呼叫，`TORCH_LOGS="guards"` 印出來的 Guard 樹（節錄）就是 Day 6 熟悉的樣子，每一格都是具體數字。

    +- GuardManager: source=L['x']
    | +- TENSOR_MATCH: check_tensor(L['x'], ..., torch.float32, device=0,
    |                  requires_grad=False, size=[4, 4], stride=[4, 1])

變化發生在第二次呼叫。batch 從 4 換成 8，`TORCH_LOGS="recompiles"` 告訴我們票沒驗過。

    Recompiling function f ...
        triggered by the following guard failure(s):
        - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 8

到這裡都跟 Day 6 一樣，Guard 失敗、重編。但這次重編不是單純再來一遍。Dynamo 在 code object 上記著一本叫 `frame_state` 的小冊子（實作在 [`pgo.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/pgo.py) 的 `FrameStateSizeEntry`），記著每個輸入每個維度上次見過的值。第二次編譯前，[`variables/builder.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/builder.py) 的 `_automatic_dynamic` 逐維比對，發現「dim 0 上次 4 這次 8，它會變」，這個維度就不再特化，改發一個符號。看新那張圖的 Guard 樹，`x` 那條 `TENSOR_MATCH` 變了，長成下面這樣。

    | +- TENSOR_MATCH: check_tensor(L['x'], ..., torch.float32, device=0,
    |                  requires_grad=False, size=[None, 4], stride=[4, 1])
    +- LAMBDA_GUARD: 2 <= L['x'].size()[0]

size 第一格不再押死（`None`），代價是樹尾多了一條 `LAMBDA_GUARD` 的範圍約束。這條約束是從 ShapeEnv 生出來的，log 裡甚至能看到它被編成的 Python 函式。

    Python shape guard function:
    def guard(L):
        if not (2 <= L['x'].size()[0]):
            return False
        return True

接著第三、第四次呼叫，`(16, 4)`、`(100, 4)` 進來，`recompiles` 完全安靜，一張圖吃下所有 batch size。整個升級流程就是一齣三幕劇。

![shape 從 4 到 8 觸發 automatic dynamic 升級的三幕劇](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day11/automatic_dynamic.gif)

*圖一：automatic dynamic 的三幕劇。左邊是每次呼叫的輸入，中間是 `f.__code__` 上的 cache entry 與 `frame_state`，右邊是驗票結果。`(4, 4)` 第一次編譯全部特化，`frame_state` 記下 dim 0 = 4。`(8, 4)` 進來 `TENSOR_MATCH` 失敗，`frame_state` 比對出 dim 0 會變，第二張圖改押符號 `s0`，size 變成 `[None, 4]` 加一條 `2 <= s0`。此後 `(16, 4)`、`(100, 4)` 全部命中同一個 entry，不再重編。*

再補兩筆帳。第一，兩次編譯是這個機制的固定成本，第一張特化的圖幾乎注定被浪費掉。如果事先就知道某個維度一定會變，可以用 `torch._dynamo.mark_dynamic(x, 0)` 直接宣告，第一次編譯就用符號，省掉那次賭輸。`torch.compile(f, dynamic=True)` 是全部維度都這樣做的粗暴版，反過來 `dynamic=False` 則把升級機制整個關掉。第二，Guard 的驗票成本幾乎沒變。這次實驗裡兩張圖的 Guard eval latency 一個 61.67 us、一個 57.55 us，符號版少驗一個具體數字、多驗一條範圍檢查，整體打平。

## 為什麼是 2 <= s0？

上面那條 `2 <= L['x'].size()[0]` 看起來有點奇怪，明明只見過 4 和 8，為什麼下界是 2 而不是 4？log 裡這條 Guard 附的說明自己招了。

    2 <= L['x'].size()[0]  # (... the guard itself is not due user code but due
    to 0/1 specialization in the framework; to avoid specialization try
    torch._dynamo.mark_unbacked(tensor, dim))

這是框架的 **0/1 特化**（0/1 specialization）。size 0 和 1 在 PyTorch 的語意裡太特殊。size 0 是空 Tensor，很多 kernel 要走完全不同的路徑。size 1 會觸發 broadcasting，`(1, 4)` 乘 `(8, 4)` 和 `(8, 4)` 乘 `(8, 4)` 語意根本不同。如果允許一個符號涵蓋 0 和 1，ShapeEnv 每回答一個 shape 問題都得三向分裂「等於 0？等於 1？還是一般情況？」。所以 Dynamo 一律假設符號至少是 2，真的來了 0 或 1，就為它們各自特化一張專屬的圖。那條 Guard 的下界 2 不是從觀測值學來的，是符號誕生時就帶著的出生條件。

訊息裡提到的 `mark_unbacked` 是逃生口，把維度標成 unbacked 符號，連 0/1 特化都不做，代價是 ShapeEnv 推理時沒有任何具體值可以參考，這個概念下面馬上會碰到。

## SymInt 和 ShapeEnv 是什麼

維度換成符號之後，`x.shape[0]` 拿到的不再是 Python int，而是 `SymInt`。它的行為很像 int，但參與運算不會塌成數字，而是長出表達式。用 `dynamic=True` 加 `graph_code` 直接看。

```python
def h(x):
    b = x.shape[0]
    return x.reshape(b * 2, -1), b * 2
```

```python
def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27][s27, 1]cuda:0"):
    mul: "Sym(2*s77)" = s77 * 2
    reshape: "f32[2*s77, (s27//2)][(s27//2), 1]cuda:0" = L_x_.reshape(mul, -1)
    return (reshape, mul)
```

這張圖資訊量不小，逐項拆開。首先，符號本身成了圖的輸入，`s77` 和 `s27` 跟 `L_x_` 並列，runtime 真的會把兩個 int 傳進去，後端生的 kernel 拿它們當迴圈邊界。其次，`b * 2` 沒有被算掉，它是表達式 `Sym(2*s77)`，Day 4 那條「int 在翻譯期就地算掉」的規則對 SymInt 不成立，因為它根本沒有值可以算。最後看 reshape 的輸出 shape，`[2*s77, s27//2]` 這條表達式一路傳播下去，錯的形狀在翻譯期就會對不上。

這些符號全部住在同一個 `ShapeEnv` 裡，每次編譯一個 frame 就配一個，它做的事有三件。

- **發符號**：每個動態維度領一個 `s` 開頭的符號。同一次編譯裡相同的 size 會領到同一個符號（duck sizing），「這兩維相等」就天然地表達在圖裡。
- **傳播表達式**：`SymInt` 的每個運算由 [`sym_node.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/sym_node.py) 的 `SymNode` 轉交 sympy 建表達式，shape 推導全程符號化。
- **收集約束**：翻譯中每問一個 shape 問題就多記一條約束，翻譯結束時 `produce_guards` 把它們變成 Guard，上面那條 `2 <= s0` 就是這樣來的。

還有一個小配件叫 **hint**，每個符號記著它第一次被看到時的具體值（`s77` 的 hint 是 4）。ShapeEnv 推不出來的問題，就拿 hint 押注走一邊，這正是下一節的主題。

## if 碰上符號會發生什麼

Day 4 講過，`if` 能不能翻譯過去，取決於「這個分支的答案，翻譯期拿得到嗎」。符號進場之後，這個問題有了三種答案。

**第一種，問 shape，拿得到。** `if x.shape[0] > 10` 這種條件，ShapeEnv 翻譯期就能處理，先從既有約束推真假，推不出來就拿 hint 押注走一邊，押注的瞬間生成一條 Guard。實跑 `dynamic=True` 先餵 size 4 再餵 20。

```python
def k(x):
    if x.shape[0] > 10:
        return x * 2
    return x + 1
```

    Recompiling function k ...
        triggered by the following guard failure(s):
        - 0/0: 2 <= x.size()[0] <= 10  # if x.shape[0] > 10:

第一次 hint 是 4，押 `False` 那邊走，Guard 記下 `s0 <= 10`，跟 0/1 特化的下界合成 `2 <= s0 <= 10`。size 20 進來 Guard 失敗、特化第二張圖，兩張並存。這是 Day 6 守恆定律的符號版，只是特化單位從一個值放寬成一個區間。

**第二種，問 Tensor 的值，拿不到。** `if x.sum() > 0` 的答案住在 GPU 記憶體裡，翻譯期根本不存在，ShapeEnv 再聰明也無從押注，預設只能 Graph Break（Day 3 那個 `attempted to jump with TensorVariable()`）。真的要把資料相依的分支留在圖裡，得改寫成 `torch.cond`。

**第三種，問資料決定的 shape，連 hint 都沒有。** 更深的坑是 `x.item()`、`x.nonzero()` 這類輸出取決於資料的操作，它們產生的符號沒有 hint（叫 unbacked SymInt，慣例用 `u0` 命名），推不出來時連押注都不行。實跑 `int(x.sum().item())`，預設直接 break，訊息還附了修法。

    Reason: Unsupported Tensor.item() call with capture_scalar_outputs=False
    Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` ... to
          include these operations in the captured graph.

打開 `capture_scalar_outputs` 之後，Dynamo 會發一個 unbacked 符號讓翻譯走下去。之後若又碰到用 `u0` 判斷的分支，就會炸出 `GuardOnDataDependentSymNode` 這類錯誤，這時有兩個工具。`torch._check(u0 >= 0)` 把你知道但 ShapeEnv 不知道的事實餵給它當約束。ShapeEnv 自己則有一套 size-oblivious 推理（`guard_size_oblivious`），對當 size 用的 unbacked 符號直接沿用 0/1 特化的假設當 `>= 2` 處理，很多分支就不用真的問值了。

## mark_dynamic 還能當偵錯工具

`mark_dynamic` 除了省掉第一次賭輸，還有一個容易被忽略的用途。假設程式碼裡藏著 `if x.shape[0] == 4`。automatic dynamic 把 dim 0 換成符號後，翻譯走到這個 if，ShapeEnv 拿 hint 押注、生成 Guard `s0 == 4`，符號當場又被押死回 4。結果是每個沒見過的 batch size 都重編一次，撞上 `recompile_limit` 後退回 eager，全程沒有錯誤訊息，只有越跑越慢。這種「符號被默默押死」的情況，`mark_dynamic` 可以把它變成大聲的失敗，實跑一次就知道。

```python
def bad(x):
    if x.shape[0] == 4:
        return x * 2
    return x + 1

x4 = torch.randn(4, 5, device="cuda")
torch._dynamo.mark_dynamic(x4, 0)
torch.compile(bad)(x4)
```

    ConstraintViolationError: Constraints violated (L['x'].size()[0])!
      - You marked L['x'].size()[0] as dynamic but your code specialized
        it to be a constant (4). If you're using mark_dynamic, either
        remove it or use maybe_mark_dynamic.

    User stack:
      File "symbolic.py", line 65, in bad
        if x.shape[0] == 4:

你宣告了「這維是動態的」，翻譯途中卻冒出一條想把它押死成 4 的約束，兩者矛盾，`produce_guards` 直接報錯，訊息連兇手那一行都指出來了。這不是 bug 是功能，逼你面對「這段程式碼不支援動態」的事實。訊息裡的 `maybe_mark_dynamic` 是溫和版，同樣優先用符號，但被特化時不報錯。順帶一提，偵錯還有 `TORCH_LOGS="+dynamic"`，ShapeEnv 每發一個符號、每收一條約束、每次特化都會印出來。

## dynamic shape 的代價

講了這麼多好處，最後得結個帳，符號不是免費的。

- **後端最佳化變弱**：動態圖裡少了具體數字，Inductor 不能按 shape 挑 kernel、不能完全展開迴圈，動態版 kernel 通常比特化版慢一些。
- **編譯變慢**：ShapeEnv 的 sympy 推理、約束化簡、Guard 生成都要時間，動態維度越多這筆帳越大。
- **推理有極限**：sympy 面對整除、取模這類表達式常常推不動，推不動就得押注加 Guard，Guard 多了又回到重編的老路。

所以「預設 static、被逼才 dynamic、0 和 1 永遠特化」整套都是刻意的折衷，只為證實會變的維度付符號的成本，其他維度繼續享受特化的紅利。這跟 Day 1 的哲學一脈相承，賭注能押多準押多準，賭輸了有止損機制兜底。

## 結語

Dynamo 的機制到今天就完整了。攔截、翻譯、包裝、驗票、記帳、收圖、寫碼、斷了再接，最後一塊拼圖是賭輸了換符號，押死的量尺換成 SymInt 的伸縮量尺，一張圖從此吃下所有 batch size。

從明天起進入第二站 AOTAutograd。Dynamo 交出來的圖只有 forward，訓練還需要 backward，而且圖裡還藏著 in-place 和 view 這些後端不想看到的東西。明天先看全景，講為什麼 backward 也要 ahead-of-time 地展開。那我們明天見！

## 參考資料

- [torch/fx/experimental/symbolic_shapes.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py)
- [torch/fx/experimental/sym_node.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/sym_node.py)
- [torch/_dynamo/pgo.py 的 FrameStateSizeEntry（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/pgo.py)
- [torch/_dynamo/variables/builder.py 的 _automatic_dynamic（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/builder.py)
- [Dynamic Shapes（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
- [The dynamic shapes manual（Google Doc，PyTorch 團隊）](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs)
