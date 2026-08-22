# Day 14 | AOTAutograd 的樂高拆解師：Decomposition

## 前言

昨天 Functionalization 把圖洗成了純函數式，但圖還是「大」的。一個 LayerNorm 節點背後頂著十幾個基本運算，一個 GELU 藏著一條完整的數學式。而這背後其實還有一個更根本的問題。PyTorch 有超過兩千個 operator（光 `torch.ops.aten` 就登記了八百多個 op 家族），如果每個後端都要為每一個 op 寫一份 codegen，那任何新後端都不用做了，光把 op 清單看完就先陣亡。

Decomposition 的思路很像玩樂高。大多數看起來很複雜的 op，其實都是少數基本積木的組合。把組合拆開，後端只需要面對基本積木，而且拆完之後，後端還可以用自己的方式把積木拼回去，拼出來的東西常常比原本更快。今天會實際看一次拆解、翻開拆解表看它長什麼樣、看規則是怎麼註冊的、講清楚為什麼拆了不會變慢，最後看哪些 op 說什麼都不拆。

正文開始！

## op 其實分好幾層

在拆之前，先把「op 有幾層」這件事講清楚，因為 decomposition 拆的方向就是沿著這個階層往下走，整理成一張表。

| 層級 | 例子 | 規模 | 誰在用 |
|----|----|----|----|
| torch API | `F.gelu`、`nn.LayerNorm` | 2000+ | 使用者 |
| ATen | `aten.gelu`、`aten.native_layer_norm` | 827 個家族（實測） | dispatcher、autograd |
| Core ATen | `aten.add`、`aten.rsqrt`、`aten.var_mean` | 約 180 個 | 後端的共同詞彙表 |
| Prims | `prims.add`、`prims.convert_element_type` | 約 250 個 | 拆到底的極簡詞彙表 |

最上層是使用者寫的 torch API。往下一層是 ATen，也就是 Day 12 開始一直看到的 `torch.ops.aten.*`，dispatcher 和 autograd 都工作在這一層。再往下是 [Core ATen IR](https://pytorch.org/docs/stable/torch.compiler_ir.html)，官方從 ATen 裡挑出約 180 個 op 當作「後端至少要支援的最小集合」。最底層是 PrimTorch 專案定義的 [prims](https://github.com/pytorch/pytorch/tree/v2.8.0/torch/_prims)，約 250 個語意最單純的基本運算，連 type promotion、broadcast 都被拆成顯式的 op，是「拆到底」的目標詞彙表。

Decomposition 做的事，就是把上層的 op 用下層的 op 重寫。要拆到哪一層停，不是寫死的，是後端自己決定的。Inductor 大致停在 Core ATen 附近（下面會看到它的表），一個只想支援三十個基本 op 的玩具後端可以要求一路拆到 prims。

## 兩個 op，拆出十二行

拿兩個大家最熟的高階 op 來拆。

```python
ln = torch.nn.LayerNorm(8).cuda()

def f(x):
    return torch.nn.functional.gelu(ln(x))

torch._logging.set_logs(aot_graphs=True)
with torch.no_grad():
    torch.compile(f)(torch.randn(4, 8, device="cuda"))
```

LayerNorm 和 GELU，兩個 op。但 `aot_graphs` 印出來的圖裡它們都不見了。

```python
def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]", arg2_1: "f32[4, 8]"):
    # LayerNorm 變成：
    var_mean = torch.ops.aten.var_mean.correction(arg2_1, [1], correction=0, keepdim=True)
    add = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt = torch.ops.aten.rsqrt.default(add)
    sub = torch.ops.aten.sub.Tensor(arg2_1, getitem_1)
    mul = torch.ops.aten.mul.Tensor(sub, rsqrt)
    mul_1 = torch.ops.aten.mul.Tensor(mul, arg0_1)     # * weight
    add_1 = torch.ops.aten.add.Tensor(mul_1, arg1_1)   # + bias
    # GELU 變成：
    mul_2 = torch.ops.aten.mul.Tensor(add_1, 0.5)
    mul_3 = torch.ops.aten.mul.Tensor(add_1, 0.7071067811865476)
    erf = torch.ops.aten.erf.default(mul_3)
    add_2 = torch.ops.aten.add.Tensor(erf, 1)
    mul_4 = torch.ops.aten.mul.Tensor(mul_2, add_2)
    return (mul_4,)
```

逐段讀。LayerNorm 被拆成它的定義。`var_mean` 一次算出平均值和變異數，`add` 加上 `eps=1e-05` 防止除以零，`rsqrt` 取反平方根，`sub` 和 `mul` 完成標準化，最後兩行乘上 weight、加上 bias，正好對應 `nn.LayerNorm(8)` 的兩個參數 `arg0_1` 和 `arg1_1`。

GELU 則被拆成它的數學定義 `0.5 * x * (1 + erf(x / sqrt(2)))`。`mul_2` 是 `0.5 * x`，`mul_3` 那個神秘的 `0.7071067811865476` 就是 `1 / sqrt(2)`，接著 `erf`、`+ 1`，最後 `mul_4` 把兩半乘起來。一條數學式，五行基本運算。

整張圖數下來，兩個高階 op 變成了十二行，而且只用到 `var_mean`、`add`、`rsqrt`、`sub`、`mul`、`erf` 六種基本運算。這就是 decomposition 的效果，不管使用者用了多少花俏的 op，到了後端手上，詞彙表只剩下少數幾種。

另外交代一下，這裡包了 `torch.no_grad()`，所以 AOTAutograd 判定是 inference，只有一張 forward 圖。訓練模式下 backward 圖也會經過同一套拆解，這點下面講到規則的本質時會再回來。

## 拆了不會變慢嗎？

看到這裡應該會有個很自然的疑問。GELU 在 eager 下是一個手寫的 CUDA kernel，一次 launch 就做完，拆成五個 op 之後，難道不是變成五次 launch、五趟記憶體來回？如果真是這樣，decomposition 就是負優化了。

答案藏在 Day 2 就看過的東西裡。Inductor 最擅長把連續的 elementwise 運算融合回一個 kernel。拆出來的 `mul`、`erf`、`add` 全部都是 pointwise，正是最好融合的那一種，Day 2 那個 `triton_poi_fused_add_cos_mul_sin_tanh_0` 的 kernel 名字就是證據，五個 op、一次 `tl.load`、一次 `tl.store`。拆解拆出來的這十幾行，最後在 GPU 上根本不會是十幾個 kernel。

所以 decomposition 和 fusion 是一套組合拳，**拆解把高階 op 打散成基本運算，融合再把基本運算收攏成大 kernel**。效果等於「為每一種複合 op 手寫一個 fused kernel」，但成本完全不同，手寫路線要為每個 op、每種組合各寫一份。拆解加融合的路線只需要為六種基本運算寫 codegen，所有組合自動涵蓋。甚至連使用者自創的、PyTorch 從來沒見過的運算組合，都能被融合成一個不存在於任何函式庫裡的客製 kernel。這就是拆的底氣。拆不是把東西變碎，是把「怎麼拼」的決定權交給後端。

![高階 op 逐層炸開成基本運算，再被 Inductor 融合收攏](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day14/decomposition.gif)

*圖一：拆解加融合的完整旅程。左邊是 torch 層的高階 op，中間是查 decomposition table 之後的 ATen 圖，LayerNorm 先炸開成七行、GELU 再炸開成五行。`x @ w` 這種戰略 op 不在表裡，原樣穿過直達後端。右邊是 Inductor 的收攏，十個 pointwise 被融回一個 `triton_poi_fused_*` kernel，`var_mean` 走 reduction kernel，mm 交給 matmul template。*

## 拆解表長什麼樣

那「怎麼拆」是誰規定的？答案意外地樸素，就是一張 op 對到 Python 函式的映射表。在 [`torch/_decomp/__init__.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_decomp/__init__.py) 裡就是一個全域 dict，實測數一下規模。

```
aten ops registered:                     827
torch._decomp.decomposition_table:      1123
inductor decompositions:                1133
```

1123 條規則（比 827 多是因為表是以 overload 為單位登記的）。每一條規則就是「用其他 op 把這個 op 實作一遍」的普通 Python 函式，用一個 decorator 註冊進表裡。GELU 那條就寫在 [`torch/_refs/nn/functional/__init__.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_refs/nn/functional/__init__.py)，把包裝拿掉之後核心長成下面這樣。

```python
@register_decomposition(aten.gelu)
def gelu(a, approximate="none"):
    M_SQRT1_2 = 0.70710678118654752440
    ...
    kAlpha = M_SQRT1_2
    return a * 0.5 * (1 + torch.erf(a * kAlpha))
```

跟上面 AOT 圖裡那五行完全對得上，連 `0.7071` 的出處都找到了。LayerNorm 的規則也一樣，`@register_decomposition(aten.native_layer_norm)` 註冊在 [`torch/_refs/__init__.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_refs/__init__.py)，更多規則集中住在 [`torch/_decomp/decompositions.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_decomp/decompositions.py)。查表的時機就是 Day 12 講過的 trace 過程。AOTAutograd 拿 FakeTensor 重跑函式時，每碰到一個 op 先查表，表裡有，就改成呼叫那個 Python 函式，tracer 走進函式內部，錄下來的自然就是拆開後的基本運算。

「規則本身也是 PyTorch 程式」這個設計比看起來重要，因為它一次帶來三個好處。

- **可以再被 trace**：拆出來的還是 Tensor 運算，可以繼續拆、繼續變換，一路拆到 prims 也行。
- **可以被微分**：規則是普通的可微運算組成的，autograd 引擎直接就能對它求導，所以 backward 圖的拆解不用另外寫一套。
- **順便當 shape 推導用**：`register_decomposition` 的文件裡明寫，註冊的同時預設也會登記到 dispatcher 的 Meta key。也就是說同一份 Python 函式，既是拆解規則，也是 FakeTensor 推 shape 用的 meta 實作，一份程式碼三種用途。

## 每個後端挑自己的那份

表既然是字典，就可以換、可以加、可以刪。Inductor 就是這麼做的，翻開 [`torch/_inductor/decomposition.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/decomposition.py)，它的表是三步組出來的。

```python
inductor_decompositions = get_decompositions([
    ...
    aten.gelu,
    aten.native_layer_norm,
    ...
])
decompositions = {**core_aten_decompositions(), **inductor_decompositions}
remove_decompositions(decompositions, decomps_to_exclude)
```

第一步從官方的 Core ATen 拆解集合出發。第二步加上自己額外想拆的 op，上面實驗裡的 `aten.gelu` 和 `aten.native_layer_norm` 就在這份名單上。第三步再刪掉一批它「不想拆」的，`decomps_to_exclude` 的每一項後面都附著理由，例如 `aten.sum` 旁邊註明 inductor lowers this directly（Inductor 自己 lowering 更快，不需要先拆），`aten.baddbmm` 註明拆了會 upcast 到 fp32 有效能問題。這三行把 decomposition 的哲學講得很明白。**拆什麼、不拆什麼，是後端的效能決策，不是全域的真理**。這也是實測數字裡 Inductor 的表（1133）比預設表（1123）多、但兩者又不是包含關係的原因。

如果哪天你要寫自己的後端（這個系列最後真的會寫一個），拿到的第一個禮物就是這張表。`get_decompositions()` 挑你要的規則，不支援的 op 讓表幫你拆掉，你只需要實作剩下的基本運算。

## 哪些 op 不該拆

最後回頭看圖上兩個「倖存者」，它們是理解 decomposition 分寸感的關鍵。

第一個是 `var_mean`。它明明可以再拆（平均值是 sum 除以 n，變異數也是幾個 reduction 的組合），但 AOT 圖裡它留著。第二個更明顯，如果函式裡有 `x @ w`，圖上會是一條原封不動的 `mm`，decomposition 完全不碰它。

因為拆解是會丟資訊的。`mm` 一旦拆成三層迴圈的乘加，後端就再也認不出「這是矩陣乘法」，也就不知道該叫 cuBLAS、該套 Triton 的 matmul template、該用 tensor core。這些高度優化的實作只認得 `mm` 這個名字，不認得迴圈。所以計算密集、有專屬 kernel 的戰略 op 必須保持原樣，一路傳到後端，讓後端在 lowering 時各自決定它們的命運。

判斷標準大致是下面三條。

- **pointwise 和簡單 reduction 儘管拆**：反正會被融合回來，拆了只賺不賠。
- **matmul、conv 這類計算密集 op 絕不拆**：拆了就換不到專屬的高效實作，賠大了。
- **中間地帶看後端的表**：像 `var_mean` 這種，AOTAutograd 層先留著，Inductor 拿到後有自己的 lowering 決定怎麼處理，決策點越晚，後端的自由度越大。

所以 decomposition 不是「一次拆到底」，而是分層發生、每層都只拆到「下一層需要的粒度」。這跟這個系列一路看下來的哲學一致，每一層轉換都只做自己該做的事，把選擇權留給更懂的人。

## 結語

Decomposition 把兩千多個 op 收斂成少數基本運算。規則是普通的 Python 函式，可以再 trace、可以微分、還兼任 meta 實作。表是可以換的字典，Inductor 加一點、刪一點，玩具後端可以拆到 prims。拆解分層發生，pointwise 儘管拆、matmul 這類戰略 op 絕不拆。而拆的底氣來自融合，拆解加融合的組合拳等於免費得到所有組合的手寫 fused kernel。

配合昨天的 Functionalization，Inductor 拿到的圖現在既純又小，只有基本運算、沒有 aliasing、沒有 mutation。到此 forward 圖的正規化完成。但還有一個懸而未決的問題。Day 12 看到 forward 多輸出了 `le` 和 `permute` 給 backward 用，誰決定存這兩個而不是別的？存多了費記憶體，存少了 backward 要重算。明天講 joint graph 與 partitioner，forward 和 backward 其實是先畫成一張圖，再由一個 min-cut 演算法切開的。那我們明天見！

## 參考資料

- [torch/_decomp/__init__.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_decomp/__init__.py)
- [torch/_decomp/decompositions.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_decomp/decompositions.py)
- [torch/_refs/nn/functional/__init__.py 裡的 GELU 拆解規則（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_refs/nn/functional/__init__.py)
- [torch/_inductor/decomposition.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/decomposition.py)
- [torch/_prims（PrimTorch，v2.8.0）](https://github.com/pytorch/pytorch/tree/v2.8.0/torch/_prims)
- [Core ATen IR 與 Prims IR（PyTorch 官方文件的 IRs 頁）](https://pytorch.org/docs/stable/torch.compiler_ir.html)
- [PyTorch 2.0 發布公告的 PrimTorch 一節](https://pytorch.org/get-started/pytorch-2.0/)
