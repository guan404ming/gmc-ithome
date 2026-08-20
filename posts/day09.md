# Day 9 | PyCodegen：新 bytecode 是怎麼生出來的

## 前言

Day 3 說 eval hook 還給 CPython 一段改寫過的 bytecode，之後五天全在講分析：翻譯、包裝、記前提、記修改、收圖。今天講合成：那段新 bytecode 到底怎麼一條一條生出來。兩個檔案：[`codegen.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/codegen.py) 負責生指令，[`bytecode_transformation.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/bytecode_transformation.py) 負責把指令表組裝回合法的 code object。

正文開始！

## 新 bytecode 的任務清單

原函式的 code object 會被換成一段等價改寫，要做的事就六件：

1. `LOAD_GLOBAL __compiled_fn_1`（昨天已經塞進 globals）
2. 把圖的每個輸入按 Source 載上 stack
3. `CALL`，拿回輸出 tuple
4. 拆包，把每個輸出放回該在的位置
5. 重播 side effect（Day 7 的帳本）
6. `RETURN_VALUE`

CPython 拿到這段照跑，完全不知道自己在執行一個編譯器的輸出。

## PyCodegen：會抄捷徑的碼生成器

核心介面很小：給 PyCodegen 一個 `VariableTracker`，它生出「把這個值弄上 stack」的最短指令序列。捷徑按優先序排：

| 情況 | 生成什麼 |
|---|---|
| 值有 Source | `source.reconstruct()`：從原位置載，`LOAD_FAST x` 或 `LOAD_GLOBAL cfg` + `LOAD_ATTR scale` |
| 值是圖的輸出 | 從暫存的輸出 tuple 取：`LOAD_FAST graph_out_0` 加下標 |
| 純常數 | `LOAD_CONST` |
| 翻譯期新生的容器 | 重建碼：`BUILD_LIST`、`BUILD_MAP` |

第一列是關鍵的省：有 Source 的值本來就在 frame 裡拿得到，何必讓圖多輸出一份。Source 鏈到這裡第三次出場：Day 6 用它生 Guard、Day 8 用它命名輸入、今天用它生載入碼。同一條鏈，三種輸出，背後的共同點是 Dynamo 對一個值記下的最關鍵資訊不是「它是什麼」，而是「runtime 怎麼拿到它」。

對照讀一次，`f(x, n) -> x * n + 1` 在 L40S 上的改寫結果（節錄，`record_pregraph_bytecode` 這類 profiler 標記已省略）：

```
ORIGINAL BYTECODE f          MODIFIED BYTECODE f
  LOAD_FAST    x               LOAD_GLOBAL  __compiled_fn_1
  LOAD_FAST    n               LOAD_FAST    x
  BINARY_OP    *               CALL         1
  LOAD_CONST   1               STORE_FAST   graph_out_0
  BINARY_OP    +               LOAD_FAST    graph_out_0
  RETURN_VALUE                 LOAD_CONST   0
                               BINARY_SUBSCR
                               DELETE_FAST  graph_out_0
                               RETURN_VALUE
```

`n` 沒被傳給 `__compiled_fn_1`：它被 bake 成常數，圖的輸入只剩 `x`。輸出 tuple 用 `[0]` 取回唯一的回傳值。如果有 side effect，重播碼會插在 `RETURN_VALUE` 之前，Day 7 已經看過那一段。

![跳轉目標：數字 offset vs Instruction 參照](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day09/jump_virtualization.png)

*圖一：跳轉虛擬化。左邊是頁碼式的數字 offset，插入一條指令全部作廢；右邊是書籤式的參照，`target` 直接指著另一個 Instruction 物件，最後組裝時才換算回 offset。*

## 底層工具箱：bytecode_transformation.py

生指令容易，組回合法的 code object 難，難的部分全在這個檔案。

**跳轉虛擬化**。bytecode 裡的跳轉原本寫的是數字位置：「跳到第 84 個 byte」。這種寫法很脆：中間插入或刪除任何一條指令，後面所有指令都會位移，每個寫死的數字同時作廢。所以 `Instruction` 把跳轉目標存成指向另一個 Instruction 物件的參照，像書籤而不是頁碼。可以實際看到：

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

`POP_JUMP_IF_TRUE` 的 `target` 直接指著那條 `RETURN_CONST` 物件。中途隨意增刪都不會斷鏈，最後組裝時才把參照換算回真正的 offset。

**EXTENDED_ARG 收斂**。一條指令的參數欄位只有一個 byte，裝得下 0 到 255。跳轉距離更大就要在前面墊一條 `EXTENDED_ARG`，麻煩在墊了這條整段 bytecode 變長、所有 offset 位移，原本剛好 250 的距離可能被推超過 255，換它也得墊，然後又位移一次。只能反覆掃描直到一整輪都沒有新增為止。

其他還有 stacksize 重算、3.11+ 的 exception table、linetable 重建（traceback 才指得回你的原始碼），以及跨版本差異：3.11 的 `CALL` 前要 `PUSH_NULL`、每一版 `LOAD_GLOBAL` 的旗標都在變，全藏在 `create_call_function` 這類 helper 後面。

總出口是 `transform_code_object`：吃原 code object 和新指令表，吐出合法的新 code object。Day 3 的 eval hook 還給 CPython 的就是這個東西。

## 收尾清掃

生成完還有一輪 `bytecode_analysis`：liveness 分析找出沒人讀的 `STORE_FAST` 直接拔（`remove_dead_code`）、跳到下一條指令的跳轉拔掉（`remove_pointless_jumps`）。注意它不是檢查器，它掃的是浪費，不是錯誤。這個分工便宜在複雜度只付一次：生成路徑有很多條，要是每一條都得自己小心不生廢碼，小心的成本會乘上路徑數；集中給一個清潔工，所有路徑都能無腦生。

## 結語

到這裡，「乖乖能翻完」的路線全通了：攔截、翻譯、包裝、記前提、記修改、收圖、寫碼。但 Day 4 就說過，翻譯隨時可能舉手放棄。明天把 Graph Break 的全套機制攤開：斷點前後兩段怎麼接、resume function 怎麼用今天這套工具生出來、以及 `fullgraph=True` 和 `explain()` 怎麼幫你抓 break。那我們明天見！

## 參考資料

- [torch/_dynamo/codegen.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/codegen.py)
- [torch/_dynamo/bytecode_transformation.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/bytecode_transformation.py)
- [torch/_dynamo/bytecode_analysis.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/bytecode_analysis.py)
