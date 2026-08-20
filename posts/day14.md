# Day 14 | Decomposition：兩千個 op 拆成幾百個

## 前言

AOTAutograd 的第二層轉換。PyTorch 有超過兩千個 operator（光 `torch.ops.aten` 就登記了八百多個 overload 家族），如果每個後端都要為每個 op 寫一份 codegen，任何新後端都不用做了。Decomposition 的思路是：大多數 op 其實是少數基本運算的組合，把組合拆開，後端只需要面對基本運算。

正文開始！

![高階 op 拆成基本運算，再由 Inductor 融合回去](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day14/decomposition.png)

*圖一：拆解加融合的組合拳。LayerNorm 與 GELU 按 decomposition table 拆成十幾種基本運算，matmul 這類戰略 op 不拆；拆出來的 pointwise 之後被 Inductor 融合回一個 kernel。*

## 實際看一次拆解

```python
ln = torch.nn.LayerNorm(8).cuda()

def f(x):
    return torch.nn.functional.gelu(ln(x))
```

兩個高階 op：LayerNorm 和 GELU。`aot_graphs` 印出來的圖裡它們都不見了：

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

LayerNorm 拆成 `var_mean / rsqrt / sub / mul / add`，GELU 拆成它的數學定義 `0.5 * x * (1 + erf(x / sqrt(2)))`，那個 `0.7071` 就是 `1/sqrt(2)`。圖上只剩十幾種最基本的 pointwise 和 reduction。

拆完看起來變慢了？GELU 從一個 kernel 變五個 op？不會，因為這五個 op 全是 elementwise，Day 2 就看過 Inductor 最擅長把連續的 elementwise 融合回一個 kernel。拆解加融合的組合拳，效果等於「為每個複合 op 手寫 fused kernel」，但一份基本 op 的 codegen 就涵蓋所有組合。這就是拆的底氣：拆是為了讓後端用自己的方式拼回去。

## 拆解表是一份 Python 字典

Decomposition 不是寫死的，就是一張 op 到 Python 函式的映射表。實測數字：

```
aten ops registered:                     827
torch._decomp.decomposition_table:      1123
inductor decompositions:                1133
```

每一條規則就是用其他 op 實作這個 op 的普通 Python 函式，例如 [`torch/_decomp/decompositions.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_decomp/decompositions.py) 裡的 GELU 就寫著上面那條數學式。因為規則本身也是 PyTorch 程式，拆出來的圖可以再被 trace、再被微分，backward 的 decomposition 不用另外寫。

選多少條規則是後端的自由：Inductor 帶自己的表（比預設多幾條、也覆寫幾條），一個只支援 32 個基本 op 的玩具後端可以要求拆到底。PrimTorch 專案定義了最底層的 prim ops（兩百多個），作為「拆到底」的目標詞彙表。

## 分層的智慧

值得注意 decomposition 是分層發生的，不是一次拆到底。AOT 圖裡 LayerNorm 拆了，但 `var_mean` 留著、`mm` 留著。因為拆解也會丟資訊：`mm` 拆成迴圈就再也認不出「這是矩陣乘法、該叫 cuBLAS 或 Triton matmul template」。所以有戰略價值的 op 保持原樣往下傳，直到 Inductor 的 lowering 階段再各自決定命運。`post_grad_graphs` log 可以看到 Inductor 接手後又拆了一輪。

判斷標準大致是：pointwise 和簡單 reduction 儘管拆（反正會被融合回來），計算密集的 matmul、conv 留著（它們有專屬的高效實作），其他看後端的表。

## 結語

Decomposition 把兩千個 op 收斂成少數基本運算：規則是普通的 Python 函式、表是可以換的字典、拆解分層發生、matmul 這類戰略 op 不拆。配合昨天的 Functionalization，Inductor 拿到的圖既純又小：只有基本運算、沒有 aliasing、沒有 mutation。

到此 forward 圖的正規化完成。但還有一個懸而未決的問題：Day 12 看到 forward 多輸出了 `le` 和 `permute` 給 backward 用，誰決定存這兩個而不是別的？存多了費記憶體，存少了 backward 要重算。明天講 joint graph 與 partitioner：forward 和 backward 其實是先畫成一張圖，再由一個 min-cut 演算法切開的。那我們明天見！

## 參考資料

- [torch/_decomp/decompositions.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_decomp/decompositions.py)
- [torch/_inductor/decomposition.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/decomposition.py)
- [torch/_prims：PrimTorch（v2.8.0）](https://github.com/pytorch/pytorch/tree/v2.8.0/torch/_prims)
