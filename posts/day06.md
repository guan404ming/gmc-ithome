# Day 6 | Guard：這張圖什麼時候還能用？

## 前言

前兩天一直在欠帳。Day 4 說 `InstructionTranslator` 翻譯時「順手記下前提」，Day 5 說 bake 常數的代價「由 Guard 記帳」。今天來還。

Guard 是 Dynamo 整個快取機制的靈魂。它決定了一張編好的圖能不能被下一次呼叫重用，也是 `torch.compile` 很多「為什麼又變慢了」「為什麼一直在重編」的答案所在。今天用 `TORCH_LOGS="guards"` 逐行讀它裝了什麼、實際改幾個輸入看哪條會失敗、看一個函式怎麼同時掛好幾張圖，最後看它為什麼要被編成一棵 C++ 的樹。

正文開始！

![每次呼叫先在 cache entry 上驗票，全過才重用，全敗才重編](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day06/guards.gif)

*圖一：同一個 `f` 連續四次呼叫的驗票過程。左邊是這次呼叫的輸入，中間是 `f.__code__` 上掛的 cache entry 與各自的 Guard 樹，右邊是結果。新 Tensor 同 shape 同 dtype 全過；`n` 從 3 變 4 讓 `EQUALS_MATCH` 失敗、重編、新圖掛到最前面且改押符號整數；`no_grad` 讓每張圖的 `GLOBAL_STATE` 都失敗、再編第三張；最後 `f(x, 3)` 逐個驗到全過的那張，命中的 entry 被搬到最前面。*

## 前提為什麼躲不掉

Dynamo 編譯的不是你的函式，是你的函式在某一組具體輸入下的樣子。翻譯時它做了一堆押注：`x` 是 float32、在 CUDA 上、shape 是 `(4, 4)`；`n` 是 3，直接 bake 進圖；`if` 走了左邊，因為條件當時為真。每一注都讓圖更快，也讓圖更窄。

下次呼叫，這些押注還成立嗎？不檢查就重用，答案錯了都不知道。逐一檢查，就是 Guard。所以 Guard 不是防禦性的裝飾，它是「特化換速度」這筆交易裡負責記帳的那一方：圖有多特化，Guard 就有多少條。

## 從 Source 到 Guard

昨天講過每個外來值都有一條 Source 鏈，Guard 就長在這條鏈上。翻譯過程中，`VariableBuilder` 每包一個值，就會順手裝一條：

```python
install_guard(source.make_guard(GuardBuilder.TYPE_MATCH))
```

`GuardBuilder` 定義在 [`torch/_dynamo/guards.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/guards.py)，上面每個大寫方法就是一種檢查。最常見的幾種：

| Guard | 檢查什麼 | 典型來源 |
|---|---|---|
| `TYPE_MATCH` | 型別一樣（比 type 的 id） | 大多數物件 |
| `ID_MATCH` | 還是同一個物件（比 id） | 函式、模組這類「不該換人」的 |
| `EQUALS_MATCH` | 值相等 | 被 bake 成常數的 int、str |
| `TENSOR_MATCH` | dtype、device、shape、stride、requires_grad | 每一個 Tensor 輸入 |
| `SEQUENCE_LENGTH` | list、tuple 長度 | 被展開的容器 |
| `GLOBAL_STATE` | grad mode、autocast 等全域開關 | 每張圖都有 |

`TENSOR_MATCH` 值得注意：它守的不只是「是個 Tensor」，而是 dtype、device、shape、stride、requires_grad 一整組，唯獨不守數值。任何一項變了，這張圖的假設就塌了：Inductor 按 float32 生的 kernel 拿到 float64 就是錯的。

一條 Guard 在翻譯期其實只是一個很小的資料物件（`torch/_guards.py` 裡的 `Guard`），主要就兩個欄位：`originating_source`（守誰，也就是昨天的 Source）和 `create_fn`（怎麼守，也就是 `GuardBuilder` 上的哪個方法）。`install_guard()` 把它丟進當前 tracing context 的一個集合裡，翻譯過程中這個集合越長越大，翻完才一次性交給 `CheckFunctionManager` 去建那棵真正會執行的檢查樹。所以翻譯期的 Guard 是「宣告」，執行期的 Guard 才是「檢查」，兩者中間隔著一次編譯。

## 動手讀一次

```python
cfg_scale = 2

def f(x, n):
    return x * n * cfg_scale

x = torch.randn(4, 4, device="cuda")
torch._logging.set_logs(guards=True)
g = torch.compile(f)
g(x, 3)
```

在 L40S 上印出來的（整理過縮排）：

```
TREE_GUARD_MANAGER:
+- RootGuardManager
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['n'], accessed_by=FrameLocalsGuardAccessor(key='n', framelocals_idx=1)
| | +- EQUALS_MATCH: L['n'] == 3
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, ..., torch.float32, device=0,
|                     requires_grad=False, size=[4, 4], stride=[4, 1])
| +- GuardManager: source=L['cfg_scale'], accessed_by=FrameLocalsGuardAccessor(key='cfg_scale', framelocals_idx=2)
| | +- EQUALS_MATCH: L['cfg_scale'] == 2

Guard eval latency = 9.60 us
```

逐行讀：

- 前三行是每張圖都有的「環境」前提：預設 device 沒被改、grad mode 和 autocast 這些全域狀態跟編譯時一樣、沒有掛 `__torch_function__` mode。它們不來自你的任何一個參數，卻同樣能讓圖失效。
- `L['n']` 是 `LocalSource` 印出來的樣子，昨天的 Source 鏈在這裡現形。`n` 是 Python int，被 bake 成常數，所以是 `EQUALS_MATCH: L['n'] == 3`。
- `L['x']` 吃一條 `TENSOR_MATCH`，dtype、device、size、stride、requires_grad 全被押住。
- `cfg_scale` 在這個實驗裡是 closure 變數，所以也印成 `L[...]`；如果是模組層全域就會是 `G['cfg_scale'] == 2`。
- 最後一行 `Guard eval latency = 9.60 us`：整棵樹跑一遍不到 10 微秒。這個數字後面會回來。

每一條後面其實還帶著註解，指出它是因為哪一行使用者程式碼裝上的（`# return x * n * cfg_scale  # guards.py:16 in f`），除錯時很好用，這裡省略。

## 實際去踩

把 `TORCH_LOGS="recompiles"` 打開，然後故意換幾種輸入：

```python
g(torch.randn(4, 4, device="cuda"), 3)   # 同 shape、同 dtype、新數值
g(torch.randn(8, 4, device="cuda"), 3)   # shape 變了
g(x, 4)                                  # 被 bake 的 int 變了
g(x.double(), 3)                         # dtype 變了
with torch.no_grad():
    g(x, 3)                              # grad mode 變了
```

```
(第一次呼叫沒有任何輸出)
Recompiling function f ... - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 8
Recompiling function f ... - 0/1: n == 3
Recompiling function f ... - 0/2: tensor 'x' dtype mismatch. expected Float, actual Double
Recompiling function f ... - 0/3: GLOBAL_STATE changed: grad_mode
```

第一次呼叫換了一顆全新的隨機 Tensor，什麼都沒發生，因為 `TENSOR_MATCH` 不看數值。後面四次每一次都重編，而且 log 直接告訴你是哪一條 Guard 沒過。這是新手最常見的重編來源清單：把會變的東西當純量參數傳、shape 一直換、dtype 混用、`no_grad` 裡外交替呼叫同一個函式。

最後那個 `GLOBAL_STATE changed: grad_mode` 常讓人意外：我又沒改任何參數。但 grad mode 決定了 AOTAutograd 要不要一起 trace 出 backward 圖、要不要保存中間結果，開著 grad 編出來的圖和 `no_grad` 底下編的根本是兩個東西，所以它跟參數的 dtype 一樣是這張圖的前提。訓練迴圈裡 forward 開著 grad、驗證時包在 `no_grad` 裡呼叫同一個 model，就會看到兩張圖並存，這是正常的，不是 bug。

還有一個常見誤會：Guard 失敗的訊息 `0/1: n == 3` 前面那個 `0/1`，是「frame 0 的第 1 個 cache entry」。同一次呼叫可能列出好幾行，那是它把每一個 entry 都驗過一遍、每一個都失敗的紀錄，不是重編了好幾次。

一個伏筆：`n` 換成 4 觸發重編之後，如果你再去看新那張圖的 Guard，會發現 `L['n']` 那條從 `EQUALS_MATCH: L['n'] == 3` 變成了 `TYPE_MATCH`，`x` 的 size 也從 `[4, 4]` 變成 `[None, 4]`。Dynamo 注意到「這個位置的值會變」，第二次編譯就改用符號整數而不是再 bake 一個常數，這叫 automatic dynamic，Symbolic Shapes 那篇會講。

## 一個函式，多張圖

Guard 失敗不等於舊的圖被丟掉。上面五次重編之後，看一下 `f.__code__` 身上掛了什麼：

```python
entries = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(f.__code__)
print(len(entries))     # 5
```

五個 cache entry，每個是一組（Guard 樹、編譯好的 code）。呼叫進來時，逐個 entry 驗票，第一個全過的就用它；全部失敗，才輪到重新編譯，編完把新 entry 掛到最前面（`extra_state.cpp` 裡的 `emplace_front`），命中的 entry 也會被搬到最前面，讓下次先檢查最近用過的那個。所以再呼叫一次 `g(x, 3)`、`g(x, 4)`、`g(torch.randn(8, 4), 3)`，`recompiles` 完全安靜，entry 數還是 5，各自命中各自的圖。

Entry 有數量上限，`torch._dynamo.config.recompile_limit` 預設 8（舊名 `cache_size_limit`），超過就放棄編譯這個 frame、退回 eager 跑，這就是 Recompilation 爆炸那篇要處理的問題。

## 跟 Guard 相處的幾個開關

知道 Guard 從哪來、怎麼失敗之後，實務上常用的幾個開關就好理解了：

- `torch._dynamo.mark_dynamic(x, dim)`：一開始就告訴 Dynamo 這個維度會變，直接用符號整數，不要先 bake 成常數再等 automatic dynamic 來救。`torch.compile(f, dynamic=True)` 是全部維度都這樣做的粗暴版。
- `torch.compiler.set_stance("fail_on_recompile")`：把重編變成錯誤。上線前用它跑一次，任何沒預期到的 Guard 失敗都會直接炸出來，而不是默默變慢。旁邊還有 `"eager_on_recompile"`，Guard 失敗時退回 eager 而不重編，適合偶爾出現的奇怪輸入。
- `torch._dynamo.config.error_on_recompile = True`：更早以前的同一件事，效果類似。
- `TORCH_LOGS="recompiles"`：上面用過的，每次重編印出是哪條 Guard 沒過。它的加強版 `recompiles_verbose` 會把所有失敗的 entry 都列出來，而不只第一個。
- `torch._dynamo.config.guard_nn_modules`：預設 `True`，會對 `nn.Module` 的屬性裝 Guard。關掉能少一些檢查，代價是改了模型屬性 Dynamo 不會發現。

共通的原則只有一句：Guard 是翻譯時押注的帳單，開關都在調「押多少注」，不是在讓帳單消失。

## 為什麼要編成一棵 C++ 樹

Guard 檢查發生在每一次呼叫的熱路徑上。如果用 Python 迴圈逐條檢查，一張圖十幾條 Guard、一次 forward 幾百個 frame，光驗票就把編譯賺到的時間吐回去了。所以翻譯結束時，`CheckFunctionManager` 把整組 Guard 編成一棵 C++ 的 `GuardManager` 樹，實作在 [`torch/csrc/dynamo/guards.cpp`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/dynamo/guards.cpp)，六千多行就是在做這件事。上面那個 9.6 微秒就是它的成績。

樹的形狀照抄資料的存取路徑：根節點拿到 frame 的 locals，往下 `L['x']` 一支掛著 `TENSOR_MATCH`，`L['n']` 一支掛著 `EQUALS_MATCH`。log 裡的 `accessed_by=FrameLocalsGuardAccessor(key='n', framelocals_idx=1)` 就是在說「這個節點的值，從 frame locals 第 1 格拿」。這個設計配了幾個很實際的最佳化：

- 取值不建 dict：`FrameLocalsMapping` 直接按索引讀 CPython 的 fast locals，省掉建立 `f_locals` 字典的開銷。
- Fail fast：每個節點記著自己失敗過幾次（`fail_count`），同一層的子節點按失敗次數排序，最常失敗的先檢查，失敗得越早浪費越少。
- Dict 帶版本號：CPython 的 dict 有 watcher，版本沒變就整棵子樹跳過不查。

## 結語

Guard 的數量不是看輸入有幾個，是看翻譯時押了多少注：每 bake 一個常數、每特化一個 shape、每走死一條分支，就多一條前提要驗。「圖有多特化，Guard 就有多少條」是條守恆律，想少驗票就得少押注。

一個 code object 可以掛多張圖，驗票從最近用過的開始，全部失敗才重編，超過 `recompile_limit` 就放棄。而為了讓每次呼叫的驗票夠快，整棵樹是 C++ 寫的、按失敗頻率排序的、能整段跳過的。

到目前為止，被追蹤的程式都很乖：算，然後 return。但真實的 Python 會改東西：`self.counter += 1`、往 list 裡 `append`、寫全域變數。這些 side effect 不能進圖（圖是純函數式的），也不能丟掉（語意會錯）。明天講 Dynamo 的第三本帳：`SideEffects`，怎麼在翻譯期把每一筆修改記下來，等圖跑完，再用生成的 bytecode 把它們一筆一筆重播回真實世界。那我們明天見！

## 參考資料

- [torch/_dynamo/guards.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/guards.py)
- [torch/csrc/dynamo/guards.cpp（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/dynamo/guards.cpp)
- [torch/csrc/dynamo/extra_state.cpp（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/dynamo/extra_state.cpp)
- [torch._dynamo.config（recompile_limit）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/config.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.compile 疑難排解：Recompilation](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
