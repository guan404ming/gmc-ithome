# Day 11 | Symbolic Shapes：讓一張圖吃下所有 batch size

## 前言

Day 5 說 int 被 bake 成常數是一場賭注，Day 6 說賭輸的代價是重編。今天講賭輸之後的止損：automatic dynamic 與 Symbolic Shapes，Dynamo 這部分的最後一塊拼圖。

正文開始！

![static 到 automatic dynamic 的三步](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day11/automatic_dynamic.png)

*圖一：automatic dynamic 的三步。第一次按 `(4, 4)` 特化；`(8, 4)` 進來 Guard 失敗，`frame_state` 發現 dim 0 會變，第二次編譯改押符號；此後 `(16, 4)`、`(100, 4)` 全走同一張圖，size 押 `[None, 4]` 加一條 `2 <= s0`。*

## 預設 static，被逼才 dynamic

Dynamo 預設 `assume_static_by_default`：第一次編譯，所有 shape 按具體值特化，`(4, 4)` 就是 `(4, 4)`。特化的圖好最佳化，這是 Day 5 講過的紅利。

變化發生在第二次。拿 `f(x, y) = x @ y` 實跑，batch 從 4 換成 8：

```
Recompiling function f ...
    - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 8
```

但這次重編不是單純再來一遍。`frame_state` 記著每個輸入每個維度上次見過的值，一比對，「dim 0 上次 4 這次 8，它會變」，第二次編譯就把這個維度換成符號。看新那張圖的 Guard：

```
TENSOR_MATCH: check_tensor(L['x'], ..., size=[None, 4], stride=[4, 1])
LAMBDA_GUARD: 2 <= L['x'].size()[0]
```

size 第一格不再押死（`None`），換來一條範圍約束。接著餵 `(16, 4)`、`(100, 4)`，`recompiles` 完全安靜：一張圖吃下所有 batch size。兩次編譯是這個機制的固定成本；知道某維一定會變，可以 `torch._dynamo.mark_dynamic(x, 0)` 直接宣告，第一次就用符號。

那條 `2 <= s0` 是 **0/1 特化**：size 0 和 1 太特殊（空 Tensor、broadcasting 規則都不同），符號一律假設至少是 2，真的來了 0 或 1 就各自特化一張圖。允許符號涵蓋 0 和 1，每個 shape 問題都得三向分裂，答案就不唯一了。

## SymInt 與 ShapeEnv

換成符號之後，`x.shape[0]` 拿到的不再是 int，是 `SymInt`。它參與運算不會塌成數字，而是長出表達式。用 `dynamic=True` 直接看：

```python
def h(x):
    b = x.shape[0]
    return x.reshape(b * 2, -1), b * 2
```

```python
def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27]"):
    mul: "Sym(2*s77)" = s77 * 2
    reshape: "f32[2*s77, (s27//2)]" = L_x_.reshape(mul, -1)
    return (reshape, mul)
```

符號本身成了圖的輸入（`Sym(s77)`），`b * 2` 是表達式 `2*s77`，reshape 的輸出 shape 是 `[2*s77, s27//2]`。所有符號都住在 `ShapeEnv`（[`symbolic_shapes.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py)）裡，它做三件事：發符號（同一個符號出現在多個輸入上，天然表達「這兩維相等」）、傳播表達式（錯的形狀翻譯期就對不上）、收集約束（每次問 shape 問題就多一條 Guard）。

## 當 if 碰上符號

兩種 `if` 的命運完全不同：

**問 shape**：`if x.shape[0] > 10`。ShapeEnv 翻譯期就能回答：推不出來就拿 hint（這個符號第一次被看到時的具體值）押注走一邊，押注的瞬間生成一條 Guard。實跑 `dynamic=True` 先餵 size 4 再餵 20：

```
Recompiling function k ...
    - 0/0: 2 <= x.size()[0] <= 10
```

第一張圖押了 `s0 <= 10`，size 20 進來 Guard 失敗、特化第二張，兩張並存。這是「圖有多特化，Guard 就有多少條」的符號版。

**問 Tensor 值**：`if x.sum() > 0`。答案住在 GPU 記憶體裡，翻譯期根本不存在，ShapeEnv 再聰明也無從押注，預設只能 Graph Break（Day 4 看過的 `TensorVariable` jump）。真的需要資料相依的分支留在圖裡，得改寫成 `torch.cond`。

更深的坑是 `x.item()`、`nonzero()` 這類輸出取決於資料的操作，產生的符號連 hint 都沒有（unbacked SymInt）。實跑 `int(x.sum().item())` 預設直接 break，訊息還附了修法：

```
Reason: Unsupported Tensor.item() call with capture_scalar_outputs=False
Hint: set torch._dynamo.config.capture_scalar_outputs = True
```

打開之後 Dynamo 會發一個 unbacked 符號讓翻譯走下去，配合 `torch._check(u0 >= 0)` 把你知道但它不知道的事實餵給 ShapeEnv。

## mark_dynamic 當偵錯工具

`mark_dynamic` 還有一個容易忽略的用途。假設程式碼裡藏著 `if x.shape[0] == 4`：automatic dynamic 把 dim 0 換成符號後，翻譯走到這個 if 生成 Guard `s0 == 4`，符號當場又被押死回 4，每個沒見過的 batch size 都重編一次，撞上 recompile limit 後退回 eager，全程沒有任何錯誤訊息。`mark_dynamic` 把這種沉默的失敗變成大聲的失敗，實跑：

```
ConstraintViolationError: Constraints violated (L['x'].size()[0])!
  - You marked L['x'].size()[0] as dynamic but your code specialized it
    to be a constant (4).
    if x.shape[0] == 4:
```

你宣告了「這維是動態的」，翻譯途中冒出一條想押死它的約束，兩者矛盾，直接報錯，訊息指向兇手那一行。這不是 bug 是功能：它逼你面對「這段程式碼不支援動態」的事實，而不是默默重編到死。

## 代價

符號不是免費的：動態圖少了具體數字，後端不能按 shape 挑 kernel、不能完全展開迴圈；ShapeEnv 的推理也讓編譯變慢。所以「預設 static、被逼才 dynamic、0 和 1 永遠特化」整套都是刻意的折衷：為證實會變的維度付符號的成本，其他維度繼續享受特化的紅利。

## 結語

Dynamo 的機制到此完整：攔截、翻譯、包裝、驗票、記帳、收圖、寫碼、斷了再接、賭輸了換符號。從明天起進入第二站 AOTAutograd：Dynamo 交出來的圖只有 forward，訓練還需要 backward，而且圖裡還藏著 in-place 和 view 這些後端不想看到的東西。明天先看全景：為什麼 backward 也要 ahead-of-time 地展開。那我們明天見！

## 參考資料

- [torch/fx/experimental/symbolic_shapes.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py)
- [Dynamic Shapes（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
- [The dynamic shapes manual（Google Doc，PyTorch 團隊）](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs)
