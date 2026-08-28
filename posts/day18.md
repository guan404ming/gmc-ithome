# Day 18 | Loop-level IR 不是樹，而是函式

## 前言

昨天總覽了 Inductor 的 pipeline，從接下 ATen 圖到吐出 kernel，中間要經過 lowering、scheduling、codegen 幾站。今天走進第一站 Lowering，看每個 ATen node 怎麼被翻譯成 Inductor 自己的 IR。FX Graph 用 node 描述一張圖，kernel 卻用迴圈一格一格地算。Lowering 就是跨過這道鴻溝的那一步。

這層 IR 的設計蠻反直覺的。多數編譯器的 IR 是資料結構，節點加上欄位，組成一棵樹或一張圖。Inductor 的 loop-level IR 卻是一條函式。一個 node 被 lower 之後，留下的是「給我一個 index，我告訴你這格怎麼算」的 Python callable，官方把這種風格叫 define-by-run IR。今天先看查表的入口，再看 Pointwise 和 Reduction 兩種 IR，最後回答為什麼這種設計讓 fusion 幾乎不用額外動手。

正文開始！

## 入口是一張表

Inductor 拿到的圖經過 Functionalization 和 Decomposition 整頓，只剩純函數式的 ATen node。接手的 `GraphLowering` 會把圖走一遍，每碰到一個 op，就去 `lowerings` 表裡查該怎麼翻譯。查到就把 node 換成 IR；查不到就走 fallback，原樣呼叫 eager kernel。一張表，就是 lowering 的入口。

今天的實驗都跑在 CPU 上（torch 2.8.0），先看這張表的規模：

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

1713 條 entry（以 overload 計），每條都對到一個 Python 函式。op 查到後會走三條路。add、relu、sum 這類 op 會變成 loop IR，之後還能 fusion。mm 和 convolution 這類「戰略 op」不拆成迴圈，而是交給高效模板去 autotune。Inductor 還不會處理的 op 則走 fallback，不能參與 fusion，但程式至少還是能跑。

## 查表換到的是一條函式

那 lowering 函式回傳的 IR 長什麼樣？以 pointwise op 為例，`make_pointwise` 的核心拿掉細節大概是這樣。

```python
def inner(*inputs):
    loaders = [x.make_loader() for x in inputs]
    ranges = inputs[0].get_size()

    def inner_fn(index):
        return fn(*[load(index) for load in loaders])

    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=ranges)
```

第一個主角 Pointwise 物件重要的就三樣。`ranges` 記輸出多大，`dtype` 記型別，`inner_fn` 則回答「給我這格的 index，該怎麼算？」它現在只描述計算，不會真的算出數字。到了 codegen，同一條描述才會變成 Triton 或 C++ code。這就是 define-by-run：IR 不是一棵固定的樹，而是跑過函式才展開。

這條函式也不只是給 codegen 用。debug 時跑一遍，它可以印成人類看得懂的 loop body；分析時跑一遍，可以數出它讀寫了哪些 buffer；codegen 時再把它變成 kernel。IR 只寫一份，不同階段用自己的方式解讀它。

另一個主角 Reduction 物件則是多記兩件事：哪幾個軸要被收掉，以及怎麼收（sum、max、prod）。絕大多數運算就落在這兩類：Pointwise 每格獨立算，Reduction 把一群格子收成一格。

這時的 IR 還只是一段描述。等到它必須成為中間 buffer，Inductor 才會讓它「落地」，這個動作叫 realize。等一下講 fusion 會用到。

## 把 IR 實際印出來

上面都是用說的這邊就來直接印出來看。`TORCH_COMPILE_DEBUG=1` 會讓 Inductor 把每張圖的中間產物 dump 到 `torch_compile_debug/` 底下，其中 `ir_pre_fusion.txt` 記的是 Scheduler fusion 前的 IR 狀態。拿一個 pointwise 和 reduction 都有的函式來編。

```python
def f(x):
    y = (x * 2 + 1).relu()
    return y, y.sum(dim=1)

torch.compile(f)(torch.randn(4, 8))
```

`y` 被 return，必須實際存在，所以這張圖有兩個 IR node。第一個是 pointwise。把 debug 輸出的樣板拿掉，它描述的其實就是下面這條迴圈。

```text
for p0 in 0..31:
    buf0[p0] = relu(arg0_1[p0] * 2 + 1)
```

`p0` 的範圍是 32，也就是把 4x8 的輸出攤平成一維。原本 FX Graph 上的乘、加、relu 是三個 node，到這裡已經變成同一條迴圈裡的一條算式。第二個 node 則是 reduction。

```text
for p0 in 0..3:
    buf1[p0] = sum(buf0[8*p0 + p1] for p1 in 0..7)
```

這次多了一組迴圈：`p0` 是留下的軸，`p1` 是要收掉的軸。32 個格子最後收成 4 個。兩段擺在一起就很清楚：pointwise 的輸入輸出一樣大，reduction 則是多進一出。

`ir_pre_fusion.txt` 還會列出每個 node 讀寫了誰、結果要交給誰。這些 dependent 的資訊今天先放著，明天 Scheduler 決定誰跟誰融，靠的就是它們。

## Fusion 是 inline 的副作用

現在來小改一個地方，這邊讓 `y` 不再被 return。

```python
def g(x):
    return (x * 2 + 1).relu().sum(dim=1)
```

雖然同一條運算鏈，不過 `ir_pre_fusion.txt` 只剩一個 node。

```text
for p0 in 0..3:
    buf0[p0] = sum(
        relu(arg0_1[8*p0 + p1] * 2 + 1)
        for p1 in 0..7
    )
```

乘、加、relu 直接進了 reduction 的迴圈，中間沒有 buffer。整條鏈只讀一次輸入、寫一次輸出。這還是 pre_fusion，Scheduler 沒上場，fusion 卻已經發生。

原因就在這層 IR 存的是函式。上游還沒 realize 時，下游拿到的不是 buffer，而是上游的 inner_fn。下游一呼叫它，就等於把上游整條計算抄進自己的 body。函式天生可以組合，所以 fusion 不需要先把兩顆 kernel 拆開重接，而是 inline 的副作用就能把他們先融合起來了。下面用動畫把整個過程走一遍。

![ATen node 逐個查表變成 inner_fn，pointwise 鏈 inline 進 reduction 的 body](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day18/lowering.gif)

*圖一：ATen node 逐個查表變成 inner_fn，還沒 realize 的 pointwise 鏈被下游吸收，最後變成同一條 loop body。*

回頭看 `f` 版本，`y` 還要 return 給呼叫者，不能單純只是一段描述，所以它必須 realize 成 `buf0`。當中間結果落地成 buffer，這條自動 inline 的鏈就在這裡斷開。

當然 inline 不是永遠划算。一顆 Pointwise 物件如果被三個下游讀取，inline 的話就等於把同一段計算算三遍。所以節點被太多下游讀取、或 inner_fn 已經太大時，Inductor 會讓它提前落地。落地是 kernel 邊界的第一刀，第二刀則由 Scheduler 來畫。

## 與 Decomposition 的分工

decomposition 和 lowering 都在翻譯 op，但分工很清楚。decomposition 在 ATen 語言內部改寫，例如把 gelu 拆成 mul、erf、add。拆完仍然是 ATen node，發生在 AOTAutograd 那一層。lowering 則是換語言，把 ATen node 換成 loop-level IR，發生在圖進入 Inductor 之後。先拆再 lower，lowering table 只要覆蓋拆剩的基本詞彙。

這也解釋了講 Decomposition 時留下的小謎題。Inductor 的 decomposition 表特地排除 `aten.sum`，旁邊註明 inductor lowers this directly。現在答案揭曉了：sum 在 lowering 這層有自己的 `make_reduction` 路線，可以直接長成 Reduction node，比先拆再翻譯乾淨。兩張表早就協調過，只要在 decomposition 拆到 lowering table 接得住的粒度就收手。

## 結語

Lowering 是 Inductor 的第一站。它先查表決定每個 ATen op 要走 loop IR、template 還是 fallback。其中 loop IR 不是一棵資料結構樹，而是一條回答「這格怎麼算」的 inner_fn。Pointwise 每格獨立算，Reduction 把一群格子收成一格。Python 函式的可組合性讓 fusion 自然發生，直到中間結果 realize 成 buffer，才畫下 kernel 邊界的第一刀。

不過今天的 fusion 都是順便發生的：上游剛好只有一個下游，形狀也剛好對得上。真實的圖裡哪些 node 要 fusion、划不划算、迴圈順序要不要重排，還是要由排程器拍板。明天來看 Scheduler，它會接手這些 loop body，配對、fusion、排順序，正式定下 kernel 邊界。那我們明天見！

## 參考資料

- [torch/_inductor/lowering.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/lowering.py)
- [torch/_inductor/ir.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/ir.py)
- [torch/_inductor/graph.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/graph.py)
- [TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes（dev-discuss）](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）

