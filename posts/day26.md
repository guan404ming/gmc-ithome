# Day 26 | 怎麼會越跑越慢？Recompilation 爆炸的診斷與修法

## 前言

昨天把執行期的 launch overhead 收完帳，結尾留了一句「shape 要穩定」。其實不只 CUDA Graph 怕變動，整個 torch.compile 都建立在「這張圖的假設還成立」上面。假設一塌，Day 6 的 Guard 驗票失敗，Dynamo 就再編一張圖。偶爾一次是機制正常運作，每次呼叫都來一次就是事故，編譯換來的加速全數被編譯本身吃掉。更麻煩的是撞到上限之後整個函式會默默退回 eager，速度掉回原點，還一句錯誤訊息都沒有。今天就來處理這個 Part 4 最常見的事故，重編怎麼發生、怎麼用 log 找到兇手、怎麼對症下藥。

正文開始！

## 重編不是 bug，是機制在補洞

先把後半段機制補完。Day 6 講過 Dynamo 在 trace 時把所有假設寫成 Guard，附在編好的成品上掛回 code object。當時只說驗票失敗就重編，但掛在 code object 上的其實不是一張圖，而是一串 cache entry，每個 entry 是一組 Guard 加一份改寫過的 bytecode。呼叫進來時沿著這串 entry 一個一個驗，誰的 Guard 全過就直接執行誰，全都不過才輪到重編，新的 entry 再插進這串清單。所以重編的準確描述是，現有的每一張圖都接不住這組輸入，只好再編一張專屬的，代價就是 Day 24 量過的一次完整編譯，秒級起跳。負責印出驗票失敗現場的程式在 [`torch/_dynamo/guards.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/guards.py)。

值得先記下的是，這串清單不是免費的。每次呼叫都要沿著 entry 逐一驗票，驗到第一個全過的為止，圖越養越多，命中前要白驗的 Guard 也越多。所以重編爆炸的帳有兩筆，明著的是一次次編譯費，暗著的是就算命中，呼叫的固定開銷也被拉長，Day 25 才剛把執行期 overhead 壓下去，這裡一不小心又全還回來。

這個機制還有一根內建的保險絲。同一個 code object 的 entry 數量有上限，`recompile_limit` 預設 8，撞到之後 Dynamo 直接放棄這個函式，從此走 eager，處理這件事的位置在 [`torch/_dynamo/convert_frame.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/convert_frame.py)。設計邏輯很務實，都失敗八次了，多半是有什麼東西每次都在變，再編下去只是把時間丟進水裡。

## 一個常數引爆的連鎖反應

實驗照舊在本機 CPU 上跑（torch 2.8.0），完整程式在 `code/day26/`。第一個實驗品是訓練迴圈裡很容易長出來的寫法，函式讀了一個全域的 step 計數器。

```python
step = 0

def poly(x):
    return ((x * step).sin() + x.cos() * 0.5).relu().sum()
```

step 每呼叫加一，開 `TORCH_LOGS="recompiles"` 連跑十次，log 每次重編都會把死因印出來（節錄）。

```
Recompiling function poly in /Users/wchiu/Documents/GitHub/gmc-ithome/code/day26/recompile.py:9
    triggered by the following guard failure(s):
    - 0/1: G['step'] == 1
    - 0/0: G['step'] == 0
```

Dynamo 把從 global 讀進來的 Python 數值當常數直接烙進圖裡，Guard 記下它必須等於 0。step 變成 1，驗票失敗，再編一張烙著 1 的圖，然後是 2、是 3，每一張圖都只服務一個值。時間帳單長這樣。

```
call 0 (step=0):   4.514 s
call 1 (step=1):   0.831 s
call 2 (step=2):   0.809 s
...
call 7 (step=7):   0.838 s
call 8 (step=8):   0.004 s
call 9 (step=9):   0.000 s
first 8 calls total: 10.19 s
```

前八次呼叫總共付了 10.19 秒，養出八張圖，每張只被用過一次就再也派不上用場。這還是一個一行的玩具函式，換成真實模型，每次重編付的就是 Day 24 量過的那種冷編譯價錢，同樣的病灶乘上去有多痛可想而知。然後 call 8 突然只剩 4 ms，不是修好了，是保險絲斷了。

```
torch._dynamo hit config.recompile_limit (8)
   function: 'poly' (/Users/wchiu/Documents/GitHub/gmc-ithome/code/day26/recompile.py:9)
   last reason: 0/7: G['step'] == 7
```

之後再拿 step=999 呼叫，只花 0.17 ms 而且答案正確，因為根本沒有編譯這回事了，整個函式退回 eager，加速歸零。這一切只出現在 log 裡，程式本身安靜得像沒事發生，這正是重編爆炸最陰險的地方。

## 爆炸源圖鑑

同一份程式再做幾組對照，把常見的爆炸源分個類。第一組把同樣的常數改成引數傳入，十個不同的值只重編一次，log 只有一行 `s == 0.0` 的失敗。這是 Day 11 的 automatic dynamic 在接，第一次賭輸之後這個純量就升級成符號，一張圖吃下所有值。但它只照顧 frame 的輸入，從 global 撿來的值不在名單上，這就是剛剛炸掉的原因。

第二組是 shape。batch size 從 8 跳到 96 六種值，同樣只重編一次，automatic dynamic 的老本行。但它救不了的變動也有兩種。rank 一變就是一張新圖，`tensor 'x' rank mismatch. expected 1, actual 2`，符號能代替某一維的長度，代替不了維度的數量，四種 rank 就是四張圖。另一種是 1 混進 batch 裡，明明已經是 dynamic 的圖還是又編了一張，log 把原因講得很白（節錄）。

```
- 0/1: 2 <= x.size()[0]  # return (x @ x.T).relu().sum()  # code/day26/recompile.py:22 in net (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.mark_unbacked(tensor, dim))
```

0 和 1 永遠被特化，這是 Day 11 就照過面的規則，符號的下界是 2，最後一個 batch 落單剩 1 筆這種事，每次都會多養一張圖。

第三組是全域開關。`no_grad` 裡外交替呼叫同一個函式，log 印出 `GLOBAL_STATE changed: grad_mode`。這其實合理，有沒有 grad 生出來的圖本來就該不一樣。這種重編是有界的，兩張圖各守一邊，之後怎麼交替都有現貨，屬於可以共存的體質，不算爆炸。dtype 和 device 換來換去也是同一類，Day 6 說過 TENSOR_MATCH 把它們整組押住，混用 float32 和 float64 就是兩張圖，只要種類有限就有限。

第四組是函式身分。把函式當引數傳給編譯過的函式，每次傳一個內容不同的 lambda，每一個都觸發像 `___check_obj_id(fn.__code__, 4310573648)` 這樣的失敗。Guard 驗的是 code object 的身分，迴圈裡重建同一行 lambda 不會炸，因為 code object 是同一個，但每次傳字面上不同的函式、或動態重定義閉包，就是一函式一張圖。

把整場事故走一遍。

![cache entry 一格一格被塞滿，計數逼近 recompile_limit，爆表後改道 eager，修好後全部命中同一格](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day26/recompile.gif)

*圖一：重編爆炸的全景。每次呼叫先沿著 code object 上的 cache entry 驗票，step 變了誰都接不住，只好再編一張圖塞進新的一格，計數條一路逼近 recompile_limit 的 8。爆表之後 Dynamo 拉下閘門，之後的呼叫全部改道 eager，編譯的加速歸零。下半場把常數搬進 tensor，同一串呼叫全部命中同一格，櫃子從此只需要一格。*

## 讀 log 抓兇手

診斷手法其實剛剛已經全程示範了，就是開著 `TORCH_LOGS="recompiles"` 重現一次問題。這個開關平常是關的，開著也只在重編時印字，正常命中一個字都不吐，所以就算掛在正式環境也不吵。拿到 log 之後，整理成三個讀法。第一，編號 `0/2` 前面是第幾個 frame，後面是第幾個 cache entry，一次重編會把每個舊 entry 的死因整串列出來，所以失敗清單一次比一次長，本身就是爆炸的心電圖。第二，失敗理由就是 Day 6 的 Guard 語言，`G['step'] == 1` 是被烙死的 Python 常數，`size mismatch` 和 `rank mismatch` 是 shape，`GLOBAL_STATE` 是 grad mode 這類全域狀態，`___check_obj_id` 是物件身分，看到理由基本上就看到了兇手。第三，`hit config.recompile_limit` 這行只在撞牆那一刻印一次，長跑的服務很容易錯過，懷疑退回 eager 的時候，拿小腳本重現比在正式環境裡守株待兔省事得多。

## 對症下藥

兇手分幾類，處方也分幾類，對照組全部實跑過。會變的 Python 常數，把它搬進 tensor。同一個函式改成吃 `torch.tensor(float(i))`，十個值一張圖，第一次 0.817 秒之後每次都在 0.1 ms 以下。數值進了 tensor 就從「圖的一部分」變成「資料」，Day 6 說過 TENSOR_MATCH 不驗數值。當引數傳純量有 automatic dynamic 接著，也只多付一次重編，最怕的就是讓函式從 global 或 config 物件裡自己撿。

shape 會變的輸入，事先用 `mark_dynamic` 講明白。對照組跑了六種 batch，從頭到尾一張圖，連 automatic dynamic 那次賭輸都省了。嫌一個一個標麻煩，`dynamic=True` 直接宣布所有維度可變，一樣一張圖收工，代價是少了按具體 shape 特化的機會，生成的 kernel 可能慢一些。rank 變動則沒有符號可救，只能在進函式前把輸入整理成固定的維度數。

函式身分那一類，處方是讓身分穩定下來。要傳進編譯區的函式定義成模組層級的具名函式，不要在呼叫現場動態生一個，身分穩定了，code object 的 Guard 自然一直過。至於 grad mode 這種有界的重編，不用修，兩張圖共存本來就是 cache 的正常用法，值得動手的是每呼叫必炸的那種。保險絲本身也可以調，`torch._dynamo.config.recompile_limit` 改大能多撐幾張圖，但那只是延後爆炸，病因不除，多大的櫃子都塞得滿。

## 結語

把今天的事故報告歸檔。重編的本質是 Guard 驗票失敗後往 code object 的 cache entry 清單再塞一張圖，一張 0.8 秒，八張就是十秒，撞上 `recompile_limit` 後整個函式退回 eager 而且不吭聲。爆炸源就那幾類，被烙進圖裡的 Python 常數、超出 automatic dynamic 射程的 rank 與 0/1、全域狀態、函式身分。診斷開 `TORCH_LOGS="recompiles"` 讀失敗的 Guard，處方是常數進 tensor、shape 標 dynamic、輸入整理成固定 rank。

今天靠一個 log 開關就破了案，但 torch.compile 的疑難雜症不只重編這一種，圖被切碎、結果不對、速度沒變快，各有各的查法。明天就來把除錯工具箱整個攤開，TORCH_LOGS 全家桶、tlparse、minifier，一次看看官方配了哪些傢伙。那我們明天見！

## 參考資料

- [torch/_dynamo/guards.py：guard 失敗與 recompiles log（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/guards.py)
- [torch/_dynamo/convert_frame.py：recompile_limit 的處理（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/convert_frame.py)
- [torch/_dynamo/config.py：recompile_limit（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/config.py)
- [PyTorch Docs: torch.compile Troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
