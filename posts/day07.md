# Day 7 | TorchDynamo 的隨行記帳員：SideEffects

## 前言

經過前兩天的介紹，現在 Dynamo 手上已經有兩本帳了。Source 記「出處」，也就是每個值在 runtime 要怎麼拿；Guard 記「前提」，也就是這張圖成立的條件。那今天就來介紹第三本帳 `SideEffects`，它記的則是「修改」。

昨天結尾有提到，前兩天被追蹤的程式都很乖：算，然後 return。但真實的 Python 就是會改東西：`self.counter += 1`、往 list 裡 `append`、寫全域變數、動 closure 的 cell。這些修改進不了圖，因為圖是純函數式的；但也丟不得，因為語意會錯。Dynamo 走出的第三條路，是翻譯期一筆一筆記帳，圖跑完再由生成的 bytecode 把帳一次結清。我們會先看帳本內部長什麼樣、實際跑一個會改東西的 `forward` 去讀改寫後的 bytecode、再對照 [`side_effects.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/side_effects.py) 看重播的程式碼是怎麼生出來的，最後把它跟前兩本帳的關係接起來。

正文開始！

## 為什麼圖必須是純的？

FX Graph 交給後端之後，後端要自由地重排、融合、刪除節點，Inductor 的整套最佳化全建立在這個自由上。一旦圖裡藏著「第三個節點會偷改全域變數」這種事，重排就不再安全。改了順序，別人看到的世界就不一樣了。所以 Dynamo 給後端的承諾就一句，**圖只算值，不碰世界**。

不過你的程式就是會碰世界。而兩個極端都不行。把修改塞進圖，後端的自由沒了；把修改丟掉，程式的語意錯了。所以出路就只有一條，翻譯期不真的改，先記下來；圖跑完，再補做。這其實跟資料庫的 transaction 是同一個思路，所有寫入先進 log，commit 的瞬間一次生效，外界要嘛看到全部、要嘛什麼都沒看到。

## 打開帳本：三個欄位、四種標記

`SideEffects` 掛在 `OutputGraph` 上（明天的主角），一個 frame 的翻譯過程共用同一本。打開 [`side_effects.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/side_effects.py) 看它的核心欄位，其實只有三個：

- `id_to_variable`：一個 `dict[int, VariableTracker]`，用 `id(obj)` 當 key，登記「這個真實物件由哪個替身代表」。同一個物件不管從幾條路徑摸到，都對到同一個替身，這是 aliasing 不會出錯的關鍵。
- `store_attr_mutations`：`dict[VariableTracker, dict[str, VariableTracker]]`，記「哪個替身的哪個屬性，被改成了哪個新替身」。屬性寫入全部落在這裡，真實物件一根汗毛都不會動。
- `keepalive`：一個單純的 list，把被追蹤的真實物件抓著不放。它存在的理由很微妙。`id_to_variable` 的 key 是 `id()`，如果物件中途被 GC 回收、記憶體位址被新物件重用，帳就會記到別人頭上。

而每個被碰過的 `VariableTracker` 身上會帶一個 `mutation_type` 標記，定義在 [`variables/base.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/base.py)，兩個軸、四種：改的是值本身還是屬性、物件是外面帶進來的還是翻譯期間新生的：

| mutation_type | 例子 |
|---|---|
| `ValueMutationExisting` | 對傳進來的 list 做 `append` |
| `AttributeMutationExisting` | `self.counter += 1` |
| `ValueMutationNew` | 翻譯期間建的 list 又被改 |
| `AttributeMutationNew` | 翻譯期間建的物件又被 `setattr` |

其中 Existing 與 New 的分野是一條生死線。existing 的修改一定要重播，因為外面的世界看得到這個物件；new 的物件如果沒逃出函式（沒被 return、沒被塞進既有結構），整筆帳直接勾銷，連重建都省了。`ValueMutationExisting` 上還有一個 `is_modified` 旗標，`append` 那一刻才被翻成 True，讓結帳時可以快速濾出「真的被改過」的那些，而不是掃過所有被登記的物件。

## 一起來動手跑跑看！

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

`forward` 三行，一行改屬性、一行改 closure 裡的 list、一行算 Tensor。`graph_code` 印出來的圖只有一個乘法：

```python
def forward(self, L_x_: "f32[4][1]cuda:0"):
    mul = L_x_ * 2
    return (mul,)
```

`calls` 和 `log` 在圖裡完全不存在，但執行結果跟 eager 一模一樣，計數器有加、list 有長。修改沒進圖，也沒丟。

那它們到底去哪了呢？我們用 `TORCH_LOGS="bytecode"` 把改寫後的 bytecode 印出來看看（節錄），答案就藏在 `RETURN_VALUE` 之前：

```
LOAD_GLOBAL   __compiled_fn_1        <- 先呼叫純圖
LOAD_FAST     x
CALL          1
STORE_FAST    graph_out_0
...
LOAD_ATTR     __setattr__            <- 載入 object.__setattr__
LOAD_FAST     self
LOAD_CONST    'calls'
LOAD_CONST    1                      <- calls 的最終值，是常數
COPY          1
BUILD_LIST    1                      <- 用它組出 [1]
LOAD_DEREF    log
LOAD_CONST    None
LOAD_CONST    None
BUILD_SLICE   2
STORE_SUBSCR                         <- 重播 1：log[:] = [1]
CALL          3                      <- 重播 2：object.__setattr__(self, 'calls', 1)
POP_TOP
RETURN_VALUE
```

生成的 bytecode 先呼叫 `__compiled_fn_1` 拿到輸出，然後才開始結帳。仔細看結帳這段的形狀。它先把 `object.__setattr__`、`self`、`'calls'`、`1` 全部排上 stack，中間插進 `log[:] = [1]` 的 slice 賦值，最後才 `CALL 3` 完成屬性寫回。這種「先把所有新值排上 stack、再一筆一筆執行寫入」的兩段式結構待會看原始碼時會再出現。三個要點，我們先筆記起來：

- **重播的是最終狀態，不是過程**。迴圈裡 `append` 十次，重播一次補齊十個元素；`counter += 1` 三次，寫回一次最終值。整張圖的執行是原子的，外界只在圖跑完之後才回來看世界。
- **最終值以常數形式被寫死在 bytecode 裡**。`LOAD_CONST 1` 那個 1 不是算出來的，是翻譯期就知道的答案，直接烙進改寫後的 code object。
- **順序在圖之後**。圖在 GPU 上算多久，Python 層的世界就維持原樣多久。

把剛剛 `forward` 這三行在時間軸上的分家整個畫出來，就是下面這張圖：

![追蹤期間修改只進帳本，圖跑完由改寫後的 bytecode 逐筆重播回真實世界](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day07/side_effects.gif)

*圖一：`forward` 的三行在時間軸上怎麼分家。追蹤期 `self.calls += 1` 和 `log.append` 進 SideEffects 帳本、真實世界完全不動；`x * 2` 進 FX Graph；圖執行完，改寫後的 bytecode 才把帳本裡的最終狀態逐筆重播回真實世界。*

## 翻譯期間，帳本是唯一真相

有個細節容易被忽略。修改被記帳之後，後續的讀取讀的是帳本，不是真實物件。

```python
def g(x, obj):
    obj.k = 7
    return x + obj.k
```

實跑的圖是 `add = L_x_ + 7`。`obj.k = 7` 當下真實物件沒被碰，但緊接著的 `obj.k` 讀到的是帳本裡的 7，直接被 bake 進圖。真實物件一直凍結到重播那一刻才更新。

對照原始碼，這件事就是一對出入口。`STORE_ATTR` 的 handler 最後會走到 `side_effects.store_attr()`，一行 dict 賦值把新替身塞進 `store_attr_mutations[item][name]`；而讀屬性時 `load_attr()` 會先查同一個 dict，查到就直接回傳帳本裡的替身，查不到才去問真實物件。這其實是被兩個限制夾出來的設計。翻譯期真的去改物件，side effect 就提前洩漏，圖還沒跑世界就變了；但讀到舊值語意又錯。唯一的出路就是讓帳本成為權威，所有讀寫都先過它。

順手劃清一條邊界，Tensor 的 in-place 運算（`x.add_(1)`）不歸這本帳管。它是 Tensor 運算，直接進圖，之後由 AOTAutograd 的 Functionalization 收拾。`SideEffects` 管的是 Python 層：屬性、容器、全域變數、closure 的 cell。

## 重播碼的產地：codegen_suffix

翻譯結束（`RETURN` 或 Graph Break）時，`OutputGraph.compile_subgraph` 負責收攤，其中結帳的部分在 `codegen_suffix` 裡，順序是固定的三步：

1. `codegen_save_tempvars()`：先處理翻譯期間新生、又活著逃出去的物件。它們在真實世界還不存在，得先發出 `object.__new__(cls)` 這類指令把殼建出來、存進一個臨時區域變數，並把這個變數登記成它的 Source。先建殼再補屬性，是因為新物件之間可能互相引用，殼不先出來後面的帳沒地方寫。
2. 中間穿插 `codegen_hooks()` 和 stack 的還原，把該回傳的值排好。
3. `codegen_update_mutated()`：真正的結帳。走過帳本裡每個 `is_modified` 的替身，按型別生出對應的重播指令。

第三步裡每種型別都有自己的重播套路，直接對照原始碼整理成表：

| 被改的東西 | 重播方式 |
|---|---|
| list | `old[:] = new`，一次 `STORE_SUBSCR` 蓋掉全部內容 |
| dict | `old.update(new)`，有 key 被刪過才先 `old.clear()` |
| 物件屬性 | `STORE_ATTR`；物件自訂了 `__setattr__` 就改走 `object.__setattr__` 繞過它 |
| 全域變數 | `STORE_GLOBAL` |
| closure cell | `STORE_DEREF` |
| `random` 的狀態 | `random.setstate(...)` 把亂數種子狀態同步回去 |

幾個細節值得停下來多看兩眼。list 不是把 `append` 重做一遍，而是整個內容用 slice 賦值換掉，這樣不管中間經歷幾次 `append`、`pop`、`reverse`，重播永遠只有一筆，而且是對原本那個 list 物件動手，別人手上握著的 reference 依然有效。改屬性時特地繞過自訂 `__setattr__` 也是同個邏輯。那個 `__setattr__` 裡的邏輯在翻譯期已經被 inline 追蹤過、效果都記在帳上了，重播時再跑一次就會重複執行。

最後，`codegen_update_mutated` 把每筆帳拆成「準備新值」和「執行寫入」兩半。準備的部分順著生，寫入的部分收集在一個 `suffixes` list 裡，最後整個反序附加。這就是上面 bytecode 裡 `log[:] = [1]` 會插在 `__setattr__` 的參數和 `CALL 3` 中間的原因。兩筆帳的新值先全部排上 stack，寫入再按正確順序一筆一筆發生。

## 沒逃出去的帳，整筆勾銷

而在結帳之前其實還有一步過濾，叫 `prune_dead_object_new()`。它從「翻譯結束時還活著的東西」出發做可達性分析，根節點有三類：即將被 return 的值（`tx.stack`）、Graph Break 時要留給後半段的區域變數（`symbolic_locals`）、以及所有 existing 物件。從這些根一路走引用鏈，走得到的 new 物件才需要重建，走不到的整筆勾銷。

實際驗證，`tmp = []; tmp.append(1); return x * 2` 改寫後的 bytecode 裡，`BUILD_LIST` 和 `append` 徹底消失，只剩一次 `__compiled_fn` 呼叫：

```
RESUME        0
LOAD_GLOBAL   __compiled_fn_7
LOAD_FAST     x
CALL          1
STORE_FAST    graph_out_0
...
RETURN_VALUE
```

`tmp` 從頭到尾只活在符號世界，真實世界從來不知道它存在過。這跟 Day 4 看到「list 留在符號世界、圖上根本沒有它」是同一件事的兩面。讀的那面由 `InstructionTranslator` 就地吸收，寫的那面由 `SideEffects` 記帳後發現死掉、整筆撕掉。

## 把三本帳拼在一起

走到這一步，三本帳終於可以拼起來了。同一個值，讀和寫走的是不同的帳：

- **讀**：值從外面進來，帶著 Source；被讀的那一刻裝 Guard，讀到的內容可能被 bake 進圖。這是 Day 5、Day 6 的路。
- **寫**：寫入被 `SideEffects` 攔下記帳，真實物件凍結；後續的讀先查帳本。
- **重播**：`codegen_update_mutated` 生指令時，要先把「被改的那個物件」放上 stack 才能對它 `STORE_ATTR`，而找回這個物件靠的正是它的 Source，也就是發出 `LOAD_FAST self`、`LOAD_DEREF log` 這些指令的那條鏈。Source 一頭生 Guard、一頭生重建 bytecode，Day 5 埋的這句話，後半句真正的使用者就是 SideEffects。

也因為讀寫是兩本帳，`self.calls += 1` 這種計數器是個經典踩雷組合。讀的那半被當成常數 bake 進圖、裝了一條 `EQUALS_MATCH` 的 Guard，寫的那半又把它加一寫回。於是下次呼叫時 `calls` 已經變了，Guard 必失敗、必重編，計數器每跳一次就重編一次，直到 automatic dynamic 介入停損。想在編譯區域裡維護計數器，用 buffer（Tensor）而不是 Python int，讓它走進圖的那條路。

## 結不了的帳，只能舉手

帳本能記的，前提是 Dynamo 對這個修改建得了模。建不了模的，翻譯就只能舉手 Graph Break。從原始碼裡挑幾個具體的例子：

- 對 Dynamo 不認識的 C 擴充物件呼叫會改內部狀態的方法。它連物件裡有什麼都不知道，自然記不了帳。
- 對有 `maxlen` 的 `deque` 做修改。滿了會自動擠掉舊元素，重播「最終狀態」需要知道每筆操作的順序，但帳本只記結果不記過程，v2.8.0 直接放棄（`gb_type="Side effect on existing deque with limited maxlen"`）。
- 在 higher order operator（例如 `torch.cond` 的分支）裡改外層 scope 的變數。分支可能不被執行，重播與否變成 runtime 才知道的事，`check_allowed_side_effect` 會擋下來。
- 要重建一個沒跑完的 generator，但追蹤期間又有變數被改。Dynamo 得把 generator 耗盡才能重建它，這會多執行到不該執行的修改，只能報錯。

而 Graph Break 本身也跟帳本有關。斷圖的瞬間走的同樣是 `compile_subgraph` 這條收攤路，前半段的帳要先結清，把已記下的修改全部重播完，才能把控制權還給 CPython，因為接手的真實 Python 必須看到最新狀態。這是 Graph Break 貴的另一個原因，不只圖被切碎，帳本也被迫提前結算，原本可以合併成一筆的修改被拆成好幾次重播。

## 結語

今天的核心其實濃縮起來就一句，圖只算值，不碰世界；修改在翻譯期進帳本，圖跑完由生成的 bytecode 一次重播最終狀態。帳本內部是 `id_to_variable` 認人、`store_attr_mutations` 記帳、`keepalive` 防止物件被回收，四種 `mutation_type` 裡 Existing 的修改必須重播，沒逃出去的 New 物件由 `prune_dead_object_new` 整筆勾銷。翻譯期間帳本是唯一真相，真實物件凍結到重播那一刻；重播按型別各有套路，list 換內容、dict 做 update、屬性走 `STORE_ATTR`，找回物件靠的是 Source。

節點在長、值被包著、Guard 在堆、修改在記帳，這些產出其實全部匯進了同一個物件。明天就來看看這個倉庫 OutputGraph，我們會看輸入怎麼「用到才登記」、以及 RETURN 或 Graph Break 的瞬間，`compile_subgraph` 是怎麼把一切收攏成一張 FX Graph 交給後端的。那我們明天見！

## 參考資料

- [torch/_dynamo/side_effects.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/side_effects.py)
- [torch/_dynamo/variables/base.py：MutationType（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/base.py)
- [torch/_dynamo/output_graph.py：compile_subgraph 與 codegen_suffix（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py)
- [torch/_dynamo/codegen.py：PyCodegen（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/codegen.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
