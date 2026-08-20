# Day 7 | self.counter += 1 怎麼過純函數的圖？SideEffects

## 前言

Guard 記「前提」、Source 記「出處」，今天講 Dynamo 的第三本帳：`SideEffects`，記「修改」。

到目前為止被追蹤的程式都很乖：算，然後 return。但真實的 Python 會改東西：`self.counter += 1`、往 list 裡 `append`、寫全域變數。這些修改進不了圖，因為圖是純函數式的；也丟不得，因為語意會錯。今天看 Dynamo 怎麼走出第三條路。

正文開始！

![每筆修改進帳本，圖跑完才重播](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day07/side_effects.gif)

*圖一：`forward` 的三行怎麼分家。`self.calls += 1` 和 `log.append` 進 SideEffects 帳本、不真的改；`x * 2` 進 FX Graph；生成的 bytecode 先呼叫純圖，再把帳本的最終狀態寫回真實世界。*

## 為什麼圖必須是純的

FX Graph 交給後端之後，後端要自由地重排、融合、刪除節點，Inductor 的整套最佳化全建立在這個自由上。一旦圖裡藏著「第三個節點會偷改全域變數」這種事，重排就不再安全。所以 Dynamo 給後端的承諾是：圖只算值，不碰世界。

但你的程式就是會碰世界。出路只有一條：翻譯期不真的改，先記下來；圖跑完，再補做。

## 帳本怎麼記

`SideEffects` 掛在 OutputGraph 上（明天的主角），實作在 [`side_effects.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/side_effects.py)。翻譯期間，每個被改過的 `VariableTracker` 帶一個 `mutation_type` 標記，兩個軸、四種：改的是值還是屬性、物件是外面帶進來的還是翻譯期間新生的：

| mutation_type | 例子 |
|---|---|
| `ValueMutationExisting` | 對傳進來的 list 做 `append` |
| `AttributeMutationExisting` | `self.counter += 1` |
| `ValueMutationNew` | 翻譯期間建的 list 又被改 |
| `AttributeMutationNew` | 翻譯期間建的物件又被 `setattr` |

Existing 與 New 的分野是生死線：existing 的修改一定要重播，因為外面的世界看得到這個物件；new 的物件如果沒逃出函式（沒被 return、沒被塞進既有結構），整筆帳直接勾銷（`prune_dead_object_new`），連重建都省了。

## 動手驗證

```python
log = []

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        log.append(self.calls)
        return x * 2

m = Model()
cm = torch.compile(m)
cm(torch.randn(4, device="cuda"))
cm(torch.randn(4, device="cuda"))
print(m.calls, log)   # 2 [1, 2]
```

`graph_code` 印出來的圖只有一個乘法：

```python
def forward(self, L_x_: "f32[4][1]cuda:0"):
    mul = L_x_ * 2
    return (mul,)
```

`calls` 和 `log` 在圖裡完全不存在，但執行結果跟 eager 一模一樣：計數器有加、list 有長。修改沒進圖，也沒丟。

那它去哪了？看 `TORCH_LOGS="bytecode"` 印的改寫後 bytecode（節錄），答案在 `RETURN_VALUE` 之前：

```
LOAD_GLOBAL  __compiled_fn_1        <- 先呼叫純圖
LOAD_FAST    x
CALL         1
STORE_FAST   graph_out_0
...
LOAD_ATTR    __setattr__            <- 重播 1：self.calls = 1 寫回
LOAD_FAST    self
LOAD_CONST   'calls'
LOAD_CONST   1
...
BUILD_LIST   1                      <- 重播 2：log[:] 接上新內容
LOAD_DEREF   log
...
STORE_SUBSCR
RETURN_VALUE
```

生成的 bytecode 先呼叫 `__compiled_fn_1` 拿到輸出，然後才用 `object.__setattr__` 把最終的 `calls` 寫回、用 slice 賦值把 `log` 的新內容補上。兩個要點：

- **重播的是最終狀態，不是過程**。迴圈裡 `append` 十次，重播一次補齊十個元素；`counter += 1` 三次，寫回一次最終值。整張圖的執行是原子的，外界只在圖跑完之後才回來看世界。
- **順序在圖之後**。圖在 GPU 上算多久，Python 層的世界就維持原樣多久。

## 翻譯期間，帳本是唯一真相

一個容易忽略的細節：修改被記帳之後，後續的讀取讀的是帳本，不是真實物件。

```python
def g(x, obj):
    obj.k = 7
    return x + obj.k
```

實跑的圖是 `add = L_x_ + 7`：`obj.k = 7` 當下真實物件沒被碰，但緊接著的 `obj.k` 讀到的是帳本裡的 7，直接被 bake 進圖。真實物件一直凍結到重播那一刻才更新。這是被兩個限制夾出來的設計：翻譯期真的去改物件，side effect 就提前洩漏，圖還沒跑世界就變了；但讀到舊值語意又錯。唯一的出路就是讓帳本成為權威。

反過來，沒逃出去的新物件連帳都不用結。`tmp = []; tmp.append(1); return x * 2` 改寫後的 bytecode 裡，`BUILD_LIST` 和 `append` 徹底消失，只剩一次 `__compiled_fn` 呼叫。

還有一個邊界要劃清：Tensor 的 in-place 運算（`x.add_(1)`）不歸這本帳管。它是 Tensor 運算，直接進圖，之後由 AOTAutograd 的 Functionalization 收拾。`SideEffects` 管的是 Python 層：屬性、容器、全域變數、closure 的 cell。

## 帳結不了就 break

遇到建不了模的修改，例如對 Dynamo 不認識的 C 擴充物件呼叫一個會改內部狀態的方法，翻譯只能舉手 Graph Break。而 break 本身也跟帳本有關：斷圖的瞬間，前半段的帳要先結清，把已記下的修改全部重播完，才能把控制權還給 CPython，因為真實世界必須是最新狀態。這是 Graph Break 貴的另一個原因：不只圖被切碎，帳本也被迫提前結算。

## 結語

圖只算值，不碰世界；修改在翻譯期進帳本，圖跑完由生成的 bytecode 一次重播最終狀態。Existing 的修改必須重播，沒逃出去的 new 物件整筆勾銷。翻譯期間帳本是唯一真相，真實物件凍結到重播那一刻。

節點在長、值被包著、Guard 在堆、修改在記帳，這些產出全部匯進同一個物件。明天看這個倉庫：OutputGraph，輸入怎麼「用到才登記」、以及 RETURN 或 Graph Break 的瞬間，`compile_subgraph` 怎麼把一切收攏成一張 FX Graph 交給後端。那我們明天見！

## 參考資料

- [torch/_dynamo/side_effects.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/side_effects.py)
- [torch/_dynamo/variables/base.py：MutationType（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/base.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
