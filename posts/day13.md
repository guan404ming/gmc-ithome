# Day 13 | Functionalization：把 in-place 變不見

## 前言

昨天說 AOTAutograd 在展開 backward 的路上順便做了兩層轉換，今天拆第一層：Functionalization。

問題是這樣的：`x.add_(1)` 就地改記憶體、`y = x.view(2, 8)` 讓兩個名字共享同一塊儲存，改 `y` 等於改 `x`。這些 aliasing 和 mutation 對後端是毒藥：Inductor 要自由地重排和融合節點，前提是「值不會在背後被改掉」。Day 7 的 SideEffects 把 Python 層的修改擋在圖外，但 Tensor 自己的 in-place 是合法的圖內運算，得用另一套辦法。

正文開始！

![Functionalization 前後對照](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day13/functionalization.png)

*圖一：同一個函式的前後對照。左邊是使用者寫的 in-place 與 view；右邊是 functionalized 之後的純函數圖，`add_` 變 `add`、view 重放維持一致，對輸入的修改集中成圖尾端的一條 `copy_`。*

## 改寫的規則

Functionalization 的規則只有兩條：

1. **In-place 換成 out-of-place**：`add_` 變 `add`，結果是一顆新 Tensor，之後所有用到舊 Tensor 的地方改用新的。
2. **View 用「重放」維持一致**：透過 view 改資料時，把修改反映回 base，再從 base 重新長出 view。

看實例。這個函式透過 view 就地改了 `x`，最後回傳依賴被改過的 `x`：

```python
def f(x):
    y = x.view(2, 8)
    y.add_(1)
    y.relu_()
    return x * 2
```

Dynamo 的圖（`graph_code`）忠實保留使用者寫法，`add_`、`relu_` 都還在。經過 Functionalization 之後（`aot_graphs`）：

```python
def forward(self, arg0_1: "f32[4, 4]"):
    view = torch.ops.aten.view.default(arg0_1, [2, 8])
    add = torch.ops.aten.add.Tensor(view, 1);  view = None       # add_ -> add
    view_1 = torch.ops.aten.view.default(add, [4, 4]);  add = None   # 寫回 base
    view_2 = torch.ops.aten.view.default(view_1, [2, 8])             # 再長出 view
    relu = torch.ops.aten.relu.default(view_2)                   # relu_ -> relu
    view_3 = torch.ops.aten.view.default(relu, [4, 4])
    mul = torch.ops.aten.mul.Tensor(view_3, 2)
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_3)         # 唯一倖存的 mutation
    return (mul,)
```

整張圖讀下來：`add_` 和 `relu_` 都變成了純函數版本，中間穿插的 `view` 對是把修改在 base（4x4）和 view（2x8）兩種形狀之間搬運的重放；`x * 2` 用的是改完的 `view_3`，不是原始輸入。圖內部完全純函數式，後端可以放心亂序。

## 邊界上那條 copy_

最後那條 `copy_` 是唯一的例外。`x` 是從外面傳進來的，呼叫者手上還握著它，函式跑完後呼叫者期待 `x` 已經被改過，這是 PyTorch 的語意，不能丟。所以 Functionalization 把「對輸入的修改」集中成圖尾端的一條 `copy_`：圖內部照純函數算，最後一步把最終結果一次寫回輸入的記憶體。

跟 Day 7 的結構一模一樣：SideEffects 把 Python 層的修改記帳、圖跑完重播；Functionalization 把 Tensor 層的修改改寫、圖尾端 `copy_` 寫回。兩層各管一段，語意都靠「最後一刻結算」保住。實測也確認：eager 和 compiled 各跑一次，輸出相等、被改過的輸入也相等。

一個更小的例子看得更清楚。`g(x): x.mul_(2); return x + 1` 的 AOT 圖：

```python
def forward(self, arg0_1: "f32[4]"):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 1)
    copy_ = torch.ops.aten.copy_.default(arg0_1, mul)
    return (add,)
```

log 開頭的 `ViewAndMutationMeta` 還把這件事寫成了 metadata：`mutates_data=True, keep_input_mutations=True`。AOTAutograd 對每個輸入輸出都記著「它有沒有被改、是不是別人的 alias」，這份 meta 決定了 wrapper 在圖外要補做哪些事。

## 為什麼不留給後端自己處理

理論上 Inductor 也可以自己分析 aliasing，但代價是每個後端都要重新實作一遍這套非常難寫對的分析（view of view、跨 view 的寫入順序、輸入輸出互為 alias）。在 AOTAutograd 集中做一次，後端拿到的圖保證純函數式，這是「正規化」的意義：把千奇百怪的合法寫法收斂成一種標準形狀，後面的每一層都只需要處理標準形狀。

代價也有：view 重放讓圖變長（上面 4 條 view 換 2 條 in-place）。不過這些 view 是免費的 metadata 操作，Inductor 大多能在 lowering 時吸收掉。

## 結語

Functionalization 把 in-place 換成 out-of-place、用 view 重放維持 alias 的一致性，圖內部變成純函數式；對輸入的修改集中成圖尾端的 `copy_`，語意在邊界上一次結清。後端從此不用懂 aliasing。

圖現在是純的了，但還是「大」的：LayerNorm、GELU 這些高階 op 一個頂十幾個基本運算。明天拆第二層轉換：Decomposition，兩千多個 ATen op 怎麼收斂到後端只需要面對的幾百個。那我們明天見！

## 參考資料

- [Functionalization in PyTorch（開發者說明）](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965)
- [torch/_subclasses/functional_tensor.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_subclasses/functional_tensor.py)
- [AOTAutograd：input mutation 的處理（原始碼註解）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/schemas.py)
