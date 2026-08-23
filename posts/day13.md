# Day 13 | 讓 in-place 消失的潔癖抄寫員 Functionalization

## 前言

昨天說 AOTAutograd 拿 FakeTensor 重跑 forward、讓 autograd 引擎展開 backward 的路上，還順手做了兩層轉換。今天就來拆第一層的 Functionalization。

先講一下今天的比喻。想像一個有潔癖的抄寫員。你交給他一份手稿，上面滿是「把第三行塗掉改成這樣」「這一段直接畫線刪掉」的就地修改指示。他從頭到尾不在你的原稿上動一筆，每個修改都謄到一張新的紙上，帳本記著「目前最新的版本是哪一張」。等整份文件定稿，他才一次性把最終版謄回你的原稿。原稿的主人看到的結果跟自己動手改一模一樣，但抄寫的過程中沒有任何一張紙被塗改過。

Functionalization 做的就是這件事，只是手稿換成了 Tensor。`x.add_(1)` 就地改記憶體、`y = x.view(2, 8)` 讓兩個名字共享同一塊儲存，改 `y` 等於改 `x`。這些 mutation 和 aliasing 對後端是毒藥，而 Functionalization 把它們全部謄寫成純函數式的版本，語意留到邊界一次結清。

正文開始！

## 為什麼 in-place 對編譯器這麼麻煩

先把問題講清楚。Inductor 這類後端要做的事是重排和融合，把幾個 pointwise 運算揉進同一個 kernel、把運算順序調整成對記憶體最友善的樣子。這一切成立的前提是「值不會在背後被改掉」，圖上一個節點的輸出，不管誰在什麼時候讀，讀到的都是同一個值。

In-place 直接毀掉這個前提。`add_` 執行之後，所有指向同一塊記憶體的名字通通變了值，而「哪些名字指向同一塊記憶體」這件事，圖上根本看不出來。`y = x.view(2, 8)` 之後，`y` 和 `x` 是兩個節點，儲存卻是同一塊。這時候 `y.add_(1)` 改的不只是 `y`，連 `x` 也一起變了。後端如果想把某個讀 `x` 的運算往前搬，搬過了 `add_` 這條線，答案就錯了。要安全地重排，每個後端都得自己做一套完整的 aliasing 分析，而這種分析出了名的難寫對。

那能不能叫使用者別寫 in-place？不行。PyTorch 的 API 裡幾乎每個 op 都有帶底線的就地版本，`optimizer.step()` 更新參數靠的全是 in-place，這是 eager 世界的一等公民，編譯器只能自己想辦法。

先跟 SideEffects 劃個界線。它管的是 Python 層的修改，`self.counter += 1`、往 list 裡 `append`，這些東西進不了圖，所以記帳後用 bytecode replay。但 `add_` 是合法的 Tensor 運算，它進得了圖，Dynamo 也真的把它原樣放進圖裡。問題出在後端不喜歡它，所以得在 Dynamo 和 Inductor 之間有人負責把它「翻譯」掉，這個人就是 AOTAutograd 裡的 Functionalization。

## Functionalization 的兩條規則

Functionalization 的規則其實只有兩條。

1. **In-place 換成 out-of-place**：`add_` 變 `add`，結果是一顆新 Tensor，帳本記下「原本那個名字的最新值，現在是這顆新 Tensor」，之後所有讀舊名字的地方，一律改讀新值。
2. **View 用「重放」維持一致**：透過 view 改資料時，把修改反映回 base（把新值 view 回 base 的形狀），再從 base 重新長出 view。base 和 view 永遠指著同一份最新內容，alias 語意靠重放保住，而不是靠真的共享記憶體。

兩條規則合起來，圖內部就完全沒有 mutation 了，每個節點的值一旦產生就不再改變，是教科書定義的 SSA 形式。剩下的唯一問題是邊界，這個下面會講。

## 實際測 view 加 in-place

拿一個把 view 和 in-place 都用上的函式來實測，一樣跑在 Modal 的 L40S 上，`torch 2.8.0+cu128`。這個函式透過 view 就地改了 `x`，最後回傳依賴被改過的 `x`。

```python
def f(x):
    y = x.view(2, 8)
    y.add_(1)
    y.relu_()
    return x * 2

x = torch.randn(4, 4, device="cuda")
torch._logging.set_logs(graph_code=True, aot_graphs=True)
torch.compile(f)(x)
```

先看 Dynamo 吐出的圖（`graph_code`）。

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4][4, 1]cuda:0"):
        l_x_ = L_x_
        y: "f32[2, 8][8, 1]cuda:0" = l_x_.view(2, 8)
        add_: "f32[2, 8][8, 1]cuda:0" = y.add_(1);  add_ = None
        relu_: "f32[2, 8][8, 1]cuda:0" = y.relu_();  y = relu_ = None
        mul: "f32[4, 4][4, 1]cuda:0" = l_x_ * 2;  l_x_ = None
        return (mul,)
```

Dynamo 忠實保留了使用者的寫法，`add_`、`relu_` 都還在，view 也還是 `l_x_.view(2, 8)`。這是刻意的分工，Dynamo 只負責把 Python 攔下來、抓成圖，正規化的髒活集中交給下一站。同一個函式經過 AOTAutograd 之後（`aot_graphs`），長成了另一副模樣。

```python
def forward(self, arg0_1: "f32[4, 4][4, 1]cuda:0"):
    view = torch.ops.aten.view.default(arg0_1, [2, 8])
    add = torch.ops.aten.add.Tensor(view, 1);  view = None          # add_ -> add
    view_1 = torch.ops.aten.view.default(add, [4, 4]);  add = None      # 寫回 base
    view_2 = torch.ops.aten.view.default(view_1, [2, 8]);  view_1 = None  # 重新長出 view
    relu = torch.ops.aten.relu.default(view_2);  view_2 = None      # relu_ -> relu
    view_3 = torch.ops.aten.view.default(relu, [4, 4]);  relu = None
    mul = torch.ops.aten.mul.Tensor(view_3, 2)
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_3)            # 唯一倖存的 mutation
    return (mul,)
```

把這張圖攤開細讀，兩條規則都找得到。

- `y.add_(1)` 變成了三行。`add` 是規則一，out-of-place 算出新值。`view_1`、`view_2` 是規則二，把 2x8 形狀的新值 view 回 4x4 的 base，再從 base 重新長出 2x8 的 view。修改在 base 和 view 兩種形狀之間搬運了一趟，兩個名字看到的內容從此一致。
- `y.relu_()` 同樣是 `relu` 加一條 `view_3` 寫回 base。
- 最關鍵的是 `mul` 那行。`x * 2` 用的是 `view_3`，不是原始輸入 `arg0_1`。抄寫員的帳本記著「`x` 的最新值是 `view_3`」，所以讀 `x` 的地方自動改讀新值。這就是「重新綁定」，舊名字不再出現，所有引用都指向最新的那份。

整張圖除了最後那條 `copy_`，內部完全純函數式，後端可以放心亂序。整個過程用動畫走一遍。

![in-place instruction 逐條被改寫成純函數，修改集中到圖尾端的 copy_](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day13/functionalization.gif)

*圖一：抄寫員的工作流程。左邊是使用者寫的 in-place 程式，被處理到的 instruction 逐條劃掉、標上改寫後的名字。中間是 FunctionalTensor 的帳本，`x` 和 `y` 兩個名字接在同一條 storage 線上，每次就地改都讓帳本的最新值換一格。右邊是 functionalized 的圖一行一行長出來，`add_` 變成 `add` 加兩條 view 重放，結尾補上寫回輸入的 `copy_`。*

## 邊界上那條 copy_

最後那條 `copy_` 是唯一的例外，也是抄寫員「把定稿謄回原稿」的那一步。`x` 是從外面傳進來的，呼叫者手上還握著它，函式跑完後呼叫者期待 `x` 已經被改過，這是 PyTorch 的語意，不能丟。所以 Functionalization 把「對輸入的修改」集中成圖尾端的一條 `copy_`，圖內部照純函數算，最後一步把最終結果一次寫回輸入的記憶體。

一個更小的例子看得更清楚。`g` 直接就地改參數。

```python
def g(x):
    x.mul_(2)
    return x + 1
```

AOT 圖長成下面這樣。

```python
def forward(self, arg0_1: "f32[4][1]cuda:0"):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 1)
    copy_ = torch.ops.aten.copy_.default(arg0_1, mul)
    return (add,)
```

`mul_` 變 `mul`，`x + 1` 改吃 `mul`，對輸入的修改壓縮成結尾一條 `copy_`。還有一個容易錯過的細節。`aot_graphs` 的 log 在圖前面印了一大串 `ViewAndMutationMeta`，裡面把這件事寫成了 metadata，像是這個輸入 `mutates_data=True`、整張圖 `keep_input_mutations=True`。AOTAutograd 對每個輸入輸出都記著「它有沒有被改、是不是別人的 alias」，這份 meta 決定了 mutation 是像這樣留在圖尾端，還是搬到圖外由 wrapper 補做。

語意有沒有真的保住？eager 和 compiled 各跑一次 `f`，比較輸出、也比較被改過的輸入。

    outputs equal: True | inputs equal after mutation: True

輸出相等，連輸入被就地改掉的結果也相等。呼叫者完全感覺不到中間發生過一場大改寫。

回頭一看，這跟 SideEffects 的結構一模一樣。SideEffects 把 Python 層的修改記帳、圖跑完用 bytecode replay。Functionalization 把 Tensor 層的修改謄寫、圖尾端一條 `copy_` 寫回。兩層各管一段，語意都靠「最後一刻結算」保住。

## 原始碼裡它在哪

對照 v2.8.0 的原始碼，這套機制分兩層。

Python 這層的入口是 [`torch/_subclasses/functional_tensor.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_subclasses/functional_tensor.py)。`FunctionalTensor` 是一個 Tensor subclass，把真正的 Tensor 包在裡面。`FunctionalTensorMode` 掛在 dispatch 路徑上，AOTAutograd 追蹤時每一個 ATen op 都會先經過它。真正的帳本則在 C++，[`aten/src/ATen/FunctionalTensorWrapper.cpp`](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/FunctionalTensorWrapper.cpp) 裡的 wrapper 存著一個 `value_`，指向「這個名字目前的最新值」。碰到 in-place op，dispatcher 透過 `Functionalize` 這個 dispatch key 把它導到對應的 out-of-place kernel，算出新 Tensor 之後呼叫 `replace_()` 把 `value_` 換過去，這就是動畫裡「帳本的最新值換一格」的那一步。

View 的部分更講究一點。每做一次 view，wrapper 會記下一筆 `ViewMeta`，裡面有一對正向和反向的轉換（從 base 長出 view、把 view 的修改折回 base）。透過 view 寫入時，就靠這對轉換把修改傳回 base、再把所有 alias 重新生成，這就是圖裡那些成對 view 的來源，PyTorch 把它叫 view replay。

AOTAutograd 這一側，追蹤前會先跑一遍 [`collect_metadata_analysis.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/collect_metadata_analysis.py) 裡的 `run_functionalized_fw_and_collect_metadata`，把函式在 functionalization 底下用 FakeTensor 跑一次，收集出上面看到的 `ViewAndMutationMeta`。哪個輸入被改過、哪個輸出其實是輸入的 alias，全部先問清楚，再決定圖要長什麼樣、wrapper 要在圖外補做哪些事。

## 為什麼不留給後端自己處理

理論上 Inductor 也可以自己分析 aliasing，但代價是每個後端都要重新實作一遍這套非常難寫對的分析，像是 view of view、跨 view 的寫入順序、輸入輸出互為 alias，每一項都是坑。在 AOTAutograd 集中做一次，後端拿到的圖保證純函數式，這就是「正規化」的意義，把千奇百怪的合法寫法收斂成一種標準形狀，後面的每一層都只需要處理標準形狀。這跟 Day 12 講的「順便換成 ATen 語言」是同一個哲學，正規化做得越徹底，後端越好寫。

代價也有，view 重放讓圖變長，上面 `f` 那張圖用四條 view 換掉了兩條 in-place。不過這些 view 是免費的 metadata 操作，不碰資料本體，Inductor 大多能在 lowering 時吸收掉。有趣的是，Inductor 在確認安全之後，甚至會在生成程式碼時把一些 buffer 原地重用，等於把 functionalization 拆掉的 in-place 又賺回來，只是這次是編譯器自己決定的、保證正確的 in-place，而不是使用者手寫的那種沒人敢動的 in-place。

## 結語

Functionalization 是 AOTAutograd 的潔癖抄寫員。in-place 換成 out-of-place、名字重新綁定到最新值，view 用重放維持 alias 的一致性，圖內部變成純函數式。對輸入的修改集中成圖尾端的 `copy_`，語意在邊界上一次結清。帳本記在 `FunctionalTensorWrapper` 的 `value_` 和 `ViewMeta` 裡，該不該留 `copy_` 由 `ViewAndMutationMeta` 說了算。後端從此不用懂 aliasing。

圖現在是純的了，但還是「大」的，LayerNorm、GELU 這些高階 op 一個頂十幾個基本運算。明天拆第二層轉換 Decomposition，看兩千多個 ATen op 怎麼收斂到後端只需要面對的幾百個。那我們明天見！

## 參考資料

- [Functionalization in PyTorch: Everything You Wanted To Know（dev-discuss）](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965)
- [torch/_subclasses/functional_tensor.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_subclasses/functional_tensor.py)
- [aten/src/ATen/FunctionalTensorWrapper.cpp（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/FunctionalTensorWrapper.cpp)
- [torch/_functorch/_aot_autograd/collect_metadata_analysis.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/collect_metadata_analysis.py)
- [torch/_functorch/_aot_autograd/schemas.py（ViewAndMutationMeta）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/schemas.py)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 4 節）
