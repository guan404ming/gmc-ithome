# Day 9 | 新的 bytecode 誰來寫？TorchDynamo PyCodegen！

## 前言

Day 3 的時候我們說過，eval hook 最後會還給 CPython 一段改寫過的 bytecode。而之後的五天其實都在講「分析」這一側，依序是翻譯（Day 4）、包裝（Day 5）、記押注（Day 6）、記修改（Day 7）、收圖（Day 8）。今天就換到「合成」這一側，看看那段新 bytecode 到底是怎麼一條一條被生出來的。

今天的主角是兩個檔案，分工非常乾淨。[`codegen.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/codegen.py) 裡的 `PyCodegen` 負責生指令。給它一個值，它就吐出「把這個值弄上 stack」的指令序列。[`bytecode_transformation.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/bytecode_transformation.py) 則負責組裝，把一張指令表變回一個 CPython 肯執行的 code object，offset、跳轉、stacksize、linetable 這些髒活全都收在裡面。簡單來說，就是一個管內容、一個管形式。

正文開始！

## 新 bytecode 的任務清單

原函式的 code object 會被換成一段等價的改寫，而它要做的事其實就只有六件。

1. `LOAD_GLOBAL __compiled_fn_1`（Day 8 已經塞進 globals）
2. 把圖的每個輸入按 Source 載上 stack
3. `CALL`，拿回輸出 tuple
4. 拆包，把每個輸出放回該在的位置
5. replay side effect（Day 7 的帳本）
6. `RETURN_VALUE`

注意這張清單，裡面沒有任何一項是「計算」。乘法、加法、`torch.sin` 全都已經搬進 `__compiled_fn_1`，留在 bytecode 層的只剩搬運。所以 PyCodegen 生出來的碼幾乎清一色是 `LOAD_*`、`STORE_*`、`CALL`、`BUILD_*` 這幾類指令，這也是它能做得這麼小的原因。

## PyCodegen 怎麼挑最短路徑

它的核心介面小到有點誇張，`PyCodegen` 物件本身是可呼叫的。`__call__` 收一個 `VariableTracker` 或一條 `Source`，往內部的指令表 append「執行完之後，這個值會出現在 stack 頂端」的指令。收圖時的所有生成，就是對著一串值逐個呼叫它。而它真正的本事在於挑最短路徑，捷徑按優先序排下來大概像下表。


| 情況        | 生成什麼                                                                               |
| --------- | ---------------------------------------------------------------------------------- |
| 值有 Source | `source.reconstruct()`：從原位置載，`LOAD_FAST x` 或 `LOAD_GLOBAL cfg` + `LOAD_ATTR scale` |
| 值是圖的輸出    | 從暫存的輸出 tuple 取：`LOAD_FAST graph_out_0` 加下標                                         |
| 純常數       | `LOAD_CONST`                                                                       |
| 翻譯期新生的容器  | 重建碼：`BUILD_LIST`、`BUILD_MAP`                                                       |


第一列是最關鍵的節省點。有 Source 的值本來就在 frame 裡拿得到，就不需要讓圖多輸出一份了。這也是 Source 鏈第三次出場了，Day 6 生 Guard、Day 8 命名輸入、今天生載入的 bytecode，這些的共同點就是是 Dynamo 對一個值記下的關鍵資訊不是「它是什麼」，而是「runtime 怎麼拿到它」。

順帶一提，`reconstruct` 這個名字在 Day 5 VariableTracker 的介面表就出現過，回答的是「改寫後的 bytecode 要怎麼把我重建出來」。有 Source 的值由 `source.reconstruct()` 發指令，從原位置載回來。沒有 Source 的值，也就是翻譯期新生的 list、dict、閉包，由 `VariableTracker` 自己的 `reconstruct()` 用 `BUILD_LIST`、`BUILD_MAP` 一磚一瓦蓋出來，Day 4 那個圖上根本沒有的 `parts` 要被 return 出去時，走的就是這條路。而連 `reconstruct` 都寫不出來的值（某些 C 擴充物件），就會變成一條 `Reconstruction failure` 的 graph break。

## 改寫前後對照著看

接下來我們就用 `TORCH_LOGS="bytecode"`，把 `f(x, n) -> x * n + 1` 在 GPU上（Python 3.12、PyTorch 2.8.0）改寫前後的 bytecode 都印出來對照看看。改寫前就是 Day 4 看過的那六條，改寫後的完整版長成下面這樣。

```
MODIFIED BYTECODE f
   0 RESUME                   0
   2 LOAD_GLOBAL              1 (NULL + __compiled_fn_1_cd21c25a_...)
  12 LOAD_GLOBAL              3 (NULL + __import_torch_dot__dynamo_dot_utils)
  22 LOAD_ATTR                4 (record_pregraph_bytecode_enter)
  42 COPY                     1
  44 STORE_FAST               3 (tmp_1)
  46 CALL                     0
  54 STORE_FAST               4 (tmp_2)
  56 LOAD_FAST                0 (x)
  58 LOAD_GLOBAL              3 (NULL + __import_torch_dot__dynamo_dot_utils)
  68 LOAD_ATTR                6 (record_pregraph_bytecode_exit)
  88 COPY                     1
  90 STORE_FAST               5 (tmp_3)
  92 LOAD_FAST                4 (tmp_2)
  94 CALL                     1
 102 POP_TOP
 104 CALL                     1
 112 STORE_FAST               2 (graph_out_0)
 114 LOAD_FAST                2 (graph_out_0)
 116 LOAD_CONST               2 (0)
 118 BINARY_SUBSCR
 122 DELETE_FAST              2 (graph_out_0)
 124 RETURN_VALUE
```

先把雜訊剝掉。`record_pregraph_bytecode_enter/exit` 那兩段是 PyTorch 2.8 給 profiler 標記「準備圖的輸入」這個區間用的包裝，跟語意無關。剝完之後剩下的主幹，其實就是上面那張任務清單的直譯。

```
LOAD_GLOBAL  __compiled_fn_1     <- 任務 1：載入編譯產物
LOAD_FAST    x                   <- 任務 2：擺輸入
CALL         1                   <- 任務 3：呼叫
STORE_FAST   graph_out_0         <- 任務 4：拆包
LOAD_FAST    graph_out_0
LOAD_CONST   0
BINARY_SUBSCR
DELETE_FAST  graph_out_0
RETURN_VALUE                     <- 任務 6
```

有幾處值得圈起來細讀。

- `n` **沒被傳給** `__compiled_fn_1`：它是 Python int，Day 5 已經被 bake 成常數了，圖的輸入只剩 `x`，所以擺輸入只需要一條 `LOAD_FAST x`，而這條指令就是 `LocalSource("x")` 的 `reconstruct()` 生出來的。
- **輸出永遠是 tuple**：就算只有一個回傳值，圖的輸出也是 `(add,)`（Day 8 看過），所以要 `BINARY_SUBSCR` 取下標 0。`graph_out_0` 是現配的暫存區域變數，用完立刻 `DELETE_FAST` 歸還引用，跟圖裡中間值用完就 `= None` 是同一個潔癖。
- profiler 那段順便示範了另外兩招。`tmp_2` 先收著 `enter` 的回傳、等輸入擺完再交給 `exit`，`COPY 1` 加 `STORE_FAST tmp_1` 則是把剛載上來的函式快取一份，而快取正好是下一節的主題。

這段「原碼進、新碼出」的改寫現場，動起來就是下面這張圖。

![PyCodegen 把原 bytecode 逐段改寫成呼叫編譯產物的新指令，最後由 transform_code_object 組裝出帶 offset 的 code object](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day09/codegen.gif)

*圖一：`f(x, n) -> x * n + 1` 的改寫現場。左邊是原 bytecode，中間的 PyCodegen 逐條發出新指令，載入 `__compiled_fn_1`、按 Source 擺輸入（`n` 被 bake、不用載）、一條 `CALL` 吃掉整段計算、拆輸出、return。最後 `transform_code_object` 組裝，offset 這時才被算出來。*

## 為什麼要生兩遍？

先想一個小問題。如果圖有兩個輸入 `cfg.a.x` 和 `cfg.a.y`，loading code 是不是就得把 `LOAD_FAST cfg`、`LOAD_ATTR a` 這條前綴走兩次？

答案是不用，因為這段碼其實生了兩遍（[`output_graph.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py) 的 `compile_subgraph`）。第一遍的產出直接丟掉，只留下 side effect。PyCodegen 有一個 `uses` 計數器，記下每個值、每條 Source 各被載了幾次。兩遍之間掃一輪，被用超過一次、又不是本來就一條指令就拿得到的（區域變數不算），就登記進 `tempvars`。第二遍才是真的出碼。登記過的值第一次被載出來時，多生一條 `COPY` 和 `STORE_FAST tmp_N` 存進暫存變數，之後每次要用都只是一條便宜的 `LOAD_FAST tmp_N`。原始碼的註解「This essentially implements CSE.」說得很直白，也就是編譯器教科書裡的 common subexpression elimination，在 bytecode 生成層又出現了一次。

另一個更便宜的快取是 `top_of_stack`。PyCodegen 會記著「上一次生成完，stack 頂端是誰」，下一個要的值剛好就是它的話，一條 `COPY 1` 複製了事，連 `LOAD_FAST` 都省下來了。

## Graph Break 時要重建什麼

再問一個問題。RETURN 的時候 stack 上不就剩一個回傳值嗎？那「重建 stack」到底是在重建什麼？

乖乖翻完的情況確實沒什麼好重建的。但 `compile_subgraph` 的另一個觸發點是 graph break。斷點可以落在任何一條指令前，那一刻符號 stack 上可能疊著好幾個算到一半的值，locals 裡還有一堆斷點之後要用的變數。CPython 從斷點接手的前提，是真實 frame 的狀態要跟 Dynamo 模擬到那一刻的狀態一模一樣，所以生成的 bytecode 得把這個狀態原樣排出來。符號 stack 上的每個值逐個丟給 PyCodegen（`restore_stack`），有 Source 的從原位置載、是圖輸出的從 `graph_out_0` 取、常數直接 `LOAD_CONST`。活著的區域變數再一人一條 `STORE_FAST` 放回去。明天就會看到，resume function 的參數表接的就是這裡排好的東西。

整段後綴的順序也是固定的。先把翻譯期新生、又逃出去的物件蓋出來存好，再排 stack，然後 replay side effect 帳本，最後 `STORE_FAST` 活變數、`DELETE_FAST graph_out_0`、交出控制權。原始碼裡其實另有一條快速通道，回傳值全是彼此不同的 Tensor、帳本乾淨等一串條件都滿足時，呼叫完直接 `UNPACK_SEQUENCE` 拆包，連 `graph_out_0` 都省。

## 底層工具箱：bytecode_transformation.py

生指令本身不難，真正難的是要組回一個 CPython 肯認帳的 code object，而難的部分全都收在這個檔案裡。關鍵零件有幾個，一個一個看。

**Instruction 是可變的指令物件**。標準庫 `dis` 吐出的指令是唯讀的視角，沒辦法拿來編輯。所以第一步 `cleaned_instructions()` 會先把 code object 轉成一串自家的 `Instruction` dataclass，欄位可以改、可以隨意插入刪除，順手把 `EXTENDED_ARG` 拆掉、把跳轉虛擬化。

**跳轉虛擬化**。bytecode 裡的跳轉原本寫的是數字位置，像「跳到第 84 個 byte」。這種寫法其實很脆，中間插入或刪除任何一條指令，後面所有指令都會位移，每個寫死的數字同時作廢。所以 `Instruction` 把跳轉目標存成指向另一個 Instruction 物件的參照，比較像書籤而不是頁碼。我們可以實際跑一段來看看。

```python
from torch._dynamo.bytecode_transformation import cleaned_instructions

def g(x):
    if x is None:
        return 1
    return 2

for i in cleaned_instructions(g.__code__):
    print(i.opname, i.arg, i.target)
```

```
LOAD_FAST          arg=0
LOAD_CONST         arg=None
IS_OP              arg=1
POP_JUMP_IF_TRUE   arg=1  -> target=RETURN_CONST@12
RETURN_CONST       arg=1
RETURN_CONST       arg=2
```

可以看到 `POP_JUMP_IF_TRUE` 的 `target` 直接指著那條 `RETURN_CONST` 物件。中途隨意增刪都不會斷鏈，最後組裝時才把參照換算回真正的 offset。

**EXTENDED_ARG 收斂**。一條指令的參數欄位只有一個 byte，裝得下 0 到 255。跳轉距離更大就要在前面墊一條 `EXTENDED_ARG`，不過麻煩的地方在於，墊了這條之後整段 bytecode 就變長了、所有 offset 都會位移，原本剛好 250 的距離可能就被推超過 255，換它也得墊，然後又位移一次。組裝的收尾（`clean_and_assemble_instructions`）裡，「反覆掃描直到收斂」真的就是一個不動點迴圈。

```python
dirty = True
while dirty:
    update_offsets(instructions)
    devirtualize_jumps(instructions)
    dirty = bool(fix_extended_args(instructions))
```

算 offset、把跳轉參照換算回數字、補 `EXTENDED_ARG`。只要補了任何一條，長度就變了，就回頭重算，直到一整輪都沒有新增為止。

**其他形式問題**。stacksize 要重算，CPython 建 frame 時是按 `co_stacksize` 配 value stack 的，算小了就是越界，所以 `stacksize_analysis` 會對新指令表做一次資料流分析取最深點。3.11+ 的 exception table 要重建，linetable 也要重建（traceback 才指得回你的原始碼）。還有跨版本差異，3.11 的 `CALL` 前要 `PUSH_NULL`、每一版 `LOAD_GLOBAL` 的旗標都在變，這些全被藏在 `create_call_function` 這類 helper 後面，PyCodegen 只管叫它們，不用管版本。

**總出口 `transform_code_object`**。它把「改寫一個 code object」做成一個模板。

```python
def transform_code_object(code, transformations, safe=False):
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    instructions = cleaned_instructions(code, safe)
    propagate_line_nums(instructions)
    transformations(instructions, code_options)
    return clean_and_assemble_instructions(instructions, keys, code_options)[1]
```

簡單來說就是拆開、交給你改、組回去。中間的 `transformations` 是個 callback。Dynamo 的主改寫（[`convert_frame.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/convert_frame.py) 裡的 `transform()`，跑完 `InstructionTranslator`、把 OutputGraph 收好的指令整段換上去）就是餵給它的一個 callback，而明天 resume function 的生成餵的則是另一個。Day 3 的 eval hook 還給 CPython 的，就是這個函式的回傳值。

生成完、組裝前其實還有一輪 [`bytecode_analysis`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/bytecode_analysis.py)。`convert_frame.py` 的 `transform()` 收尾就這麼一行。

```python
instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))
```

liveness 分析找出沒人讀的 `STORE_FAST` 直接拔掉，跳到下一條指令的跳轉也一併拔掉。它掃的是浪費，不是錯誤。這個分工的划算之處在於，生成路徑有很多條（return、break、side effect、resume 的各種組合），廢碼集中交給一個清潔工，每條路徑就都能無腦生。

## 結語

PyCodegen 就是一位只寫搬運碼的代筆人，按 Source 和 `reconstruct` 挑最短路徑，還會生兩遍、照帳把重複載入折進暫存變數。bytecode_transformation 則收走全部的形式問題，最後由 `transform_code_object` 吐出一個合法的新 code object，交還給 Day 3 的 eval hook。

到這裡，「乖乖能翻完」的路線就全部打通了，攔截、翻譯、包裝、記前提、記修改、收圖、寫碼。不過 Day 4 就說過，翻譯隨時可能舉手放棄。明天就來把 Graph Break 的全套機制攤開，斷點前後兩段怎麼接、resume function 怎麼用今天這套工具生出來、以及 `fullgraph=True` 和 `explain()` 怎麼幫你抓 break。那我們明天見！

## 參考資料

- [torch/_dynamo/codegen.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/codegen.py)
- [torch/_dynamo/bytecode_transformation.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/bytecode_transformation.py)
- [torch/_dynamo/bytecode_analysis.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/bytecode_analysis.py)
- [torch/_dynamo/output_graph.py：compile_subgraph（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/output_graph.py)
- [torch/_dynamo/convert_frame.py：transform（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/convert_frame.py)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)

