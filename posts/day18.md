# Day 18 | Loop-level IR 不是一棵樹，是一條函式

## 前言

昨天把 Inductor 的流水線整個總覽了一遍，從接下 ATen 圖到吐出 kernel，中間排著 lowering、scheduling、codegen 幾站。今天走進第一站 Lowering，看每個 ATen node 是怎麼被翻譯成 Inductor 自己的 IR 的。FX Graph 是圖的語言，node 連 node，而 kernel 是迴圈的語言，一格一格算，Lowering 就是跨過這道語言鴻溝的那一步。

這層 IR 有個蠻反直覺的設計。多數編譯器的 IR 是資料結構，節點加欄位，組成一棵樹或一張圖，你可以走訪它、比對它、改寫它。Inductor 的 loop-level IR 卻是一條函式。一個 node 被 lower 之後，留下來的東西是「給我一個 index，我告訴你這一格輸出怎麼算」的 Python callable，官方把這種風格叫 define-by-run IR。今天會先看查表這個入口，再把這條函式的長相講清楚，接著用 debug 輸出實際看 Pointwise 和 Reduction 兩種 IR，最後回答為什麼這種設計讓融合幾乎是免費的。

正文開始！

## 入口是一張表

Inductor 拿到的圖，經過 Day 13 和 Day 14 的整頓，只剩下純函數式的 ATen node。接手的類別叫 `GraphLowering`，定義在 [`graph.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/graph.py)，它本質上是一個 FX interpreter，把圖按拓撲序走一遍，每碰到一個 call_function node 就去查一個全域的 dict。這個 dict 名字就叫 `lowerings`，住在 [`lowering.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/lowering.py) 開頭。查到了，就呼叫查到的函式，把 node 換成 IR。查不到，就走 fallback，把原本的 eager kernel 包起來原樣呼叫。

今天的實驗都跑在 CPU 上（torch 2.8.0），先看看這張表的規模，還有幾個代表性的 op 在不在裡面。

```
lowerings entries: 1713
aten.add.Tensor                  in lowerings
aten.relu.default                in lowerings
aten.sum.default                 in lowerings
aten.mm.default                  in lowerings
aten.convolution.default         in lowerings
aten._cdist_forward.default      in lowerings (fallback)
lowerings[aten.relu.default] -> torch._inductor.lowering.make_pointwise.<locals>.inner
lowerings[aten.sum.default] -> torch._inductor.lowering.sum_
```

1713 條 entry（以 overload 計）。每一條的 value 都是普通的 Python 函式，relu 對到的是 `make_pointwise` 做出來的函式，sum 對到的是一個叫 `sum_` 的函式。註冊方式跟 Day 14 的 decomposition table 如出一轍，就是 decorator。

```python
relu = register_pointwise(aten.relu)

@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    ...
    fn = make_reduction("sum", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)
```

`register_lowering` 還會順手處理 broadcast 和 type promotion，所以每條規則只要專心描述計算本身。表裡也藏著幾種不同的命運。`aten.mm` 和 `aten.convolution` 在表裡，但它們的 lowering 不長迴圈，而是走 matmul template 那條路，交給預先寫好的高效模板去 autotune，Day 14 說的「戰略 op 不拆」就是在這裡接關的。`aten._cdist_forward` 也在表裡，但同時被登記在 fallbacks 名單，它的 value 是一層 wrapper，執行時直接呼叫 eager kernel，對 Inductor 來說是一個不透明的節點，融合的手伸不進去，但至少語意保住了。

## 查表換到的是一條函式

那 lowering 函式回傳的 IR 長什麼樣？以 pointwise op 為例，`make_pointwise` 的核心拿掉細節之後大概是這樣。

```python
def inner(*inputs):
    loaders = [x.make_loader() for x in inputs]
    ranges = inputs[0].get_size()

    def inner_fn(index):
        return fn(*[load(index) for load in loaders])

    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=ranges)
```

一個 Pointwise 物件身上重要的就三樣。`ranges` 記輸出多大，`dtype` 記型別，`inner_fn` 是一條函式，吃一組 index，回傳「這一格的值怎麼算」。注意它不真的算出數字，body 裡呼叫的 `ops.load`、`ops.add` 都是符號操作，之後誰拿著這條函式執行，才決定這段描述變成 Triton code 還是 C++ code。這就是 define-by-run 的意思，IR 的內容不是被資料結構存下來的，而是執行這條 Python 函式的過程跑出來的。

這個設計還有一層聰明之處。`ops.load`、`ops.add` 這些呼叫的實際意義不是寫死的，它們會被轉發給當下掛著的 handler。debug 時掛一個印字串的 handler，把 inner_fn 跑一遍，就得到人類可讀的 body。分析時掛一個計數的 handler，跑一遍就知道這條鏈讀了幾個 buffer。到了 codegen，掛的是 Triton 或 C++ 的 handler，同一條函式跑出來的就是 kernel 原始碼。IR 只寫一份，解讀方式隨掛上的 handler 換，等一下實驗印出來的 loop body 就是這麼來的。

另一個主角 Reduction 多帶兩樣東西，記著哪幾個軸要被收掉、用什麼方式收（sum、max、prod）。兩個類別都繼承自 [`ir.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/ir.py) 裡的 `Loops`，這層 IR 的絕大多數運算就落在這兩大類，一類是每格獨立算，一類是一群格子收成一格。

順帶交代包在外面的兩層殼。lowering 函式實際回傳的是 `TensorBox(StorageBox(Pointwise(...)))`，TensorBox 對應一個 tensor 名字，StorageBox 對應一塊儲存，兩層 box 的功能就是讓內容物可以換。之後某個時刻這顆 Pointwise 需要落地成真的 buffer 時，box 裡面會被原地換成 `ComputedBuffer`，拿著 box 的下游全都不用改。點到為止，知道有這兩層殼就好。

## 把 IR 實際印出來

講了半天長相，不如直接印出來看。`TORCH_COMPILE_DEBUG=1` 會讓 Inductor 把每張圖的中間產物 dump 到 `torch_compile_debug/` 底下，其中 `ir_pre_fusion.txt` 記的是 Scheduler 融合前的 IR 狀態。拿一個 pointwise 和 reduction 都有的函式來編。

```python
def f(x):
    y = (x * 2 + 1).relu()
    return y, y.sum(dim=1)

torch.compile(f)(torch.randn(4, 8))
```

`y` 被 return，必須實際存在，所以這張圖有兩個 IR node。第一個是 pointwise，它的 loop body 被完整印了出來（節錄）。

```python
class op0_loop_body:
    var_ranges = {p0: 32}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(2.0, torch.float32)
        mul = ops.mul(load, constant)
        constant_1 = ops.constant(1.0, torch.float32)
        add = ops.add(mul, constant_1)
        relu = ops.relu(add)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, relu, None)
        return store
```

這就是 inner_fn 被印出來的樣子。`p0` 是 loop var，範圍 32，4x8 的輸出被攤平成一維。讀進 `arg0_1` 之後乘 2、加 1、relu，最後寫進 `buf0`。值得注意的是 `x * 2`、`+ 1`、`.relu()` 在 FX Graph 上是三個 node，在這裡已經是同一條函式裡的三行，這件事下一節回來講。第二個 node 是 reduction。

```python
class op1_loop_body:
    var_ranges = {p0: 4, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_1, reduction)
        return store_reduction
```

跟 pointwise 的差異全在形狀上。loop var 變成兩組，`p0` 是留下來的軸，`p1` 是要被收掉的軸，讀的時候用 `8*p0 + p1` 掃過整列，寫的時候只用 `p0`，32 個格子收成 4 個。結尾也從 `ops.store` 換成 `ops.reduction` 加 `ops.store_reduction`。兩段擺在一起，Pointwise 和 Reduction 的分界就很清楚，前者輸入輸出一樣大，後者天生就是多進一出。

順帶一提，`ir_pre_fusion.txt` 裡除了 body 還印了每個 node 的讀寫依賴和 users。像 `buf0` 的 users 列著 op1 和 OUTPUT 兩個，意思是它同時被 reduction 讀、也要交還給呼叫者。這些依賴資訊今天先放著，明天 Scheduler 決定誰跟誰融的時候，靠的全是它們。

## 融合是內聯的副作用

現在把函式改一個地方，`y` 不再被 return。

```python
def g(x):
    return (x * 2 + 1).relu().sum(dim=1)
```

同一條運算鏈，`ir_pre_fusion.txt` 裡卻只剩一個 node。

```python
class op0_loop_body:
    var_ranges = {p0: 4, p1: 8}
    index0 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(2.0, torch.float32)
        mul = ops.mul(load, constant)
        constant_1 = ops.constant(1.0, torch.float32)
        add = ops.add(mul, constant_1)
        relu = ops.relu(add)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', relu)
        ...
```

mul、add、relu 直接出現在 reduction 的 body 裡，中間 buffer 一顆都沒有，整條鏈只讀一次 `arg0_1`、只寫一次 `buf0`。再強調一次檔名，這是 pre_fusion，Scheduler 根本還沒上場，融合就已經發生了。三個 ATen node 沒有經過任何配對分析，自然而然地縮成了一個。

機制藏在前面 `make_pointwise` 那行 `make_loader()` 裡。sum 的 lowering 跟上游要一個 loader，而 Pointwise 的 `make_loader` 寫得再直白不過：

```python
class Pointwise(Loops):
    def make_loader(self):
        ...
        return self.inner_fn
```

上游還沒落地成 buffer 的話，下游拿到的 loader 就是上游的 inner_fn 本身。呼叫它，等於把上游整條計算抄進自己的 body。函式呼叫函式，天生就能組合，融合在這層 IR 裡不是「分析兩個 kernel 能不能合併」的難題，而是內聯的副作用。如果 IR 是資料結構樹，做同一件事得改圖、接邊、重寫 index 映射，每一步都是程式碼和 bug。整個過程用動畫走一遍。

![ATen node 逐個查表變成 inner_fn，pointwise 鏈內聯進 reduction 的 body](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day18/lowering.gif)

*圖一：lowering 的全程。ATen 圖上的 node 逐個查 lowering table，`aten.add` 變成一條 inner_fn，relu 的 inner_fn 直接呼叫它、內聯成一條，sum 的 Reduction 再把整段 pointwise 鏈吸進自己的 body，realize 之後就是實驗裡那個 loop body。*

回頭看 `f` 版本，`buf0` 是誰決定要落地的？是 `StorageBox.realize()`。`y` 要被 return 給呼叫者，不能只是一段描述，於是 box 裡的 Pointwise 被換成 ComputedBuffer、登記進圖，這才有了 `buf0` 這個名字。

當然，內聯不是永遠划算的。一顆被三個下游讀到的 Pointwise，內聯進去等於同一段計算被抄三份、算三遍，便宜的加減乘除無所謂，一長串鏈就虧了。所以除了輸出必須 realize，Inductor 還有一組 heuristic，被太多下游讀到、inner_fn 累積得太大，都會讓一個節點提前落地。落地就是在畫 kernel 邊界的第一刀，第二刀由 Scheduler 來畫。

## 跟 Decomposition 的分工

Day 14 的 decomposition 和今天的 lowering 都在翻譯 op，分工其實劃得很乾淨。decomposition 是 ATen 語言內部的改寫，gelu 拆成 mul、erf、add，拆完還是 ATen node，發生在 AOTAutograd 那一層。lowering 是換語言，把 ATen node 換成 loop-level IR，發生在圖進到 Inductor 之後。先拆再 lower，lowering table 只需要覆蓋拆剩的基本詞彙。

這也解釋了 Day 14 留下的一個小謎。Inductor 的 decomposition 表特地把 `aten.sum` 排除掉，旁邊註明 inductor lowers this directly。現在答案揭曉，sum 在 lowering 這層有自己的 `make_reduction` 路線，直接長成 Reduction node，比先拆成別的 op 再翻譯來得乾淨。兩張表是協調過的，decomposition 拆到 lowering table 接得住的粒度就收手。

而這 1713 條 entry 也分成三種待遇，正好對應三種 op 的性格。走 loop IR 的（add、relu、sum）可以被融合，走 template 的（mm、convolution）去做 autotune，fallback 的原樣執行。一張表，就是這個後端能力範圍的完整清單。哪天你要幫 Inductor 加一個 op 的支援，第一件事就是往這張表裡塞一條規則。

## 結語

Lowering 是 Inductor 的第一站。入口是一張 dict，ATen op 對到 Python 函式，查到就把 node 換成 IR。這層 IR 不是資料結構樹，而是一條 inner_fn，給我 index，我告訴你這格怎麼算，Pointwise 和 Reduction 兩大類撐起絕大多數運算，外面包著 TensorBox 和 StorageBox 兩層可以換內容物的殼。define-by-run 的可組合性讓融合變成內聯的副作用，還沒 realize 的計算自動被下游吸收，realize 的時機就是 kernel 邊界的第一刀。

不過今天看到的融合都是順便發生的，上游剛好只有一個下游、形狀又剛好對得上。真實的圖裡誰跟誰融、融了划不划算、迴圈順序要不要重排，需要一個真的排程器來拍板。明天來看 Scheduler，它會把今天這些 loop body 接過去，配對、融合、排順序，把 kernel 的邊界正式定下來。那我們明天見！

## 參考資料

- [torch/_inductor/lowering.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/lowering.py)
- [torch/_inductor/ir.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/ir.py)
- [torch/_inductor/graph.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/graph.py)
- [TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes（dev-discuss）](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）
