# Day 25 | 把一整串 launch 錄成一次 replay：CUDA Graph 與 reduce-overhead

## 前言

昨天把快取講完，編譯的帳算是結清了，同一張圖第二次遇到直接從磁碟撿現成的，編譯成本只付一次，Part 3 也就走完了。從今天起進入 Part 4，視角從「怎麼把 kernel 編快」換成「這套東西在真實場景怎麼用、會出什麼事」。第一站先處理執行期的另一種 overhead。kernel 編好了、也快取好了，每一步還是得由 CPU 把它們一顆一顆丟上 GPU，這個發射動作本身有固定成本，模型裡小 kernel 一多，GPU 反而大部分時間在等 CPU。今天的主角 CUDA Graph 就是來收這筆帳的，torch.compile 把它包裝成 mode="reduce-overhead"。

正文開始！

## launch overhead 是什麼

GPU 不會自己動起來。每一顆 kernel 都要 CPU 透過 CUDA API 發射，填參數、排上 stream、通知 driver，這一串動作在 CPU 端要花幾個 us，而且跟 kernel 本身多大無關，是每次發射固定要付的門票。kernel 夠大時這筆錢無所謂，一顆大 matmul 跑幾百 us，前面幾 us 的發射早被蓋過去。麻煩的是另一種體質的模型，一長串小 kernel，每顆只跑幾 us，發射的時間跟執行的時間同一個量級甚至更長。這時 GPU 軌道上的景象會變成 kernel 之間全是空隙，算完一顆就停下來等 CPU 發下一顆，硬體大半時間在發呆。

平常感覺不到這件事，是因為 CUDA 的執行本來就是非同步的，CPU 把 kernel 丟進 stream 的佇列就繼續往前跑，只要 kernel 夠肥，CPU 發射的速度永遠追得上 GPU 消化的速度，佇列裡隨時有存貨。小 kernel 把這個緩衝打破了，GPU 幾 us 就清空一顆，CPU 這邊每顆卻要走完一整套 Python 呼叫、dispatcher、driver 的流程才發得出下一顆，存貨見底，換 GPU 排隊等 CPU。

值得強調的是這筆 overhead 不歸編譯管。Day 20 的 fusion 能把 pointwise 融進別人的 loop，但 matmul 這種 extern kernel 融不動，32 層的 MLP 編完還是幾十顆 kernel，每一步照樣幾十次 launch。編譯把每顆 kernel 變快了，發射它們的次數並沒有變。

## 先量出這筆帳有多大

實驗跑在 L40S 上（torch 2.8.0），模型故意挑 launch-bound 的體質，32 層 Linear(256, 256) 接 ReLU，batch 只有 8，每顆 kernel 都小得可憐。eager、預設 compile、reduce-overhead 三種各 bench 一輪，計時用 CUDA event 包住一百次呼叫取平均，再重複十輪取統計，完整程式在 `code/day25/`。

```
eager            1.124 ms (+/- 0.008)
compile default  0.960 ms (+/- 0.019)
reduce-overhead  0.119 ms (+/- 0.003)
default vs eager          1.17x
reduce-overhead vs eager  9.45x
```

預設模式只快了 1.17 倍，原因就是上面說的，matmul 是 extern kernel，fusion 拿它沒辦法，一步還是六十幾次 launch。換成 reduce-overhead，同一個模型直接 9.45 倍，而它生成的 kernel 跟預設模式是同一批，差距全部來自發射方式。再把 batch 放大到 8192 當對照組。

```
eager            2.228 ms (+/- 0.002)
compile default  1.585 ms (+/- 0.003)
reduce-overhead  1.567 ms (+/- 0.004)
reduce-overhead vs default 1.01x
```

kernel 一大，執行時間把 overhead 整個淹掉，reduce-overhead 跟預設模式只差 1.01 倍。同一招在兩種 batch 下賺頭差了快十倍，這是今天最重要的一組對照。

## 錄影機模型，capture 與 replay

reduce-overhead 底下是 CUDA 的原生功能 CUDA Graph，心智模型就是錄影。先正常發射一遍，把這段期間所有 kernel launch 的參數、順序、依賴全部側錄下來存成一張 graph，之後要跑同樣的序列，CPU 只發一次 replay，driver 把整張圖原樣重播，幾十次發射收成一次。PyTorch 把它包在 `torch.cuda.CUDAGraph`，手動玩一次長這樣。開錄之前要先在 side stream 上把模型跑過幾輪 warmup，讓 lazy 初始化、記憶體配置這些只發生一次的動作先發生完，免得被錄進圖裡，然後才進 `torch.cuda.graph` 的 context 開錄。

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y = model(static_x)

static_x.copy_(new_x)
g.replay()
```

replay 之後拿 `static_y` 跟 eager 對答案，順便量 32 層共 64 次發射跟一次 replay 的差距。

```
replay result matches eager: True
eager 64 launches  1.116 ms
graph replay       0.113 ms
replay vs eager    9.90x
```

答案一致，速度跟 reduce-overhead 量到的幾乎一樣，證明那 9.45 倍確實就是這台錄影機的功勞。代價也寫在這段程式裡了。錄下來的是「對哪個位址發射哪顆 kernel」，所以位址全是死的，輸入必須固定住，新資料要先 `copy_` 進 `static_x` 才能 replay，graph 內的中間 buffer 也全部來自一個專屬的 memory pool。capture 期間更不能有 CPU 參與的動作，`.item()` 這種要同步回 CPU 的操作會讓 capture 直接失敗，因為 replay 重播的只有 GPU 端的指令，CPU 端的邏輯不會跟著重來。

![CPU 軌道每次發射都付一段橘色 overhead，錄成一張 graph 之後一鍵重播](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day25/cudagraph.gif)

*圖一：launch overhead 的收帳過程。eager 下 CPU 每發射一顆 kernel 都付一段橘色成本，GPU 軌道上小 kernel 之間全是 idle。CUDAGraph 把整串發射側錄成一張圖，之後 CPU 只按一次 replay，kernel 緊緊相連，單步時間從 1.124 ms 掉到 0.119 ms。*

## reduce-overhead 背後的 cudagraph trees

把 mode="reduce-overhead" 翻開，這個模式其實只做一件事，把 Inductor 的 cudagraphs 開關打開，讓 Inductor 在編好的 wrapper 外面套上這台錄影機。真正的實作在 [`torch/_inductor/cudagraph_trees.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cudagraph_trees.py)，名字裡的 trees 是重點。手動的 CUDAGraph 一次只顧一張圖，但真實模型會被 graph break 切成好幾段，forward 和 backward 又是分開的兩張，如果每張各開一個 memory pool，中間結果得在 pool 之間複製，記憶體也是各算各的。cudagraph trees 讓所有 graph 共用同一個 pool，並把「先跑 A 再跑 B」和「先跑 A 再跑 B'」這些不同的執行路徑錄成一棵樹，走到哪條路徑就重播哪條，記憶體照樹上的最大路徑算而不是全部加總。

這個檔案裡還住著一位管家。每個函式先用真發射跑過 warmup，位址和記憶體都穩定了才開錄，之後進入 replay 狀態。它同時追蹤輸出的存活，graph B 開錄前會確認不會踩到 graph A 還活著的輸出，真的踩到就換個位置重錄一份。重錄也有保險絲，同一個函式重錄超過上限（預設 128 次）就放棄錄影，退回逐顆發射。這也解釋了為什麼 reduce-overhead 的前幾次呼叫特別慢，那不只是編譯，還有錄影機在架設。原始碼註解裡有個細節，warmup 這幾輪本身就跑在 graph 的 memory pool 裡，這樣錄影時不必為了留住輸入再多拷一份記憶體，warmup 結束、正式錄完之後，接下來的每次呼叫才真正進入一次 replay 的世界。

## 這筆交易的代價

位址押死這件事，Inductor 替你處理了大半，輸入自動複製進 static buffer，參數本來就長住在固定位址。但有一條逃不掉，輸出活在 graph 的 memory pool 裡，下一次 replay 會原地覆寫上一次的結果。拿 log 驗證，抓著上一輪的輸出不放再呼叫一次，兩個 tensor 的位址一模一樣。

```
hold output alive, call again: ptr 22480592830464 -> 22480592830464, same: True
```

想留住結果就得自己 `clone`，這是 reduce-overhead 最容易踩到的一顆雷。第二類代價是有些函式根本上不了車，例如會 mutate 輸入的圖，replay 每次都會原樣重做那個 mutation，語意對不上，Inductor 會直接跳過並在 perf_hints 裡留話。

```
[__perf_hints] skipping cudagraphs due to mutated inputs (1 instances). Found from :
   File "/root/cudagraph.py", line 90, in mutate
    x.add_(1)
```

同一個檢查也擋 CPU tensor 混在圖裡的情況。第三類是 dynamic shape，錄影錄的是固定位址加固定大小，Day 11 那種一張圖吃所有 batch size 的彈性到這裡失效，每個新 shape 都要重錄一張 graph，shape 太多時記憶體和重錄時間一起爆炸，重錄太多張還會被警告。最後是記憶體本身，pool 會把 workspace 一直留著換速度，這在 `torch.compile` 的文件裡寫得很直白，overhead 的減少是拿記憶體換的。順帶一提，除錯體驗也會變差，replay 中的 kernel 不會經過 Python，print 插不進去，出錯的堆疊也不會指向你的程式碼，開發階段先用預設模式把模型跑對，再換 reduce-overhead 收 overhead，是比較省事的順序。

## 什麼場景賺，什麼場景不賺

把今天的數字收成一條判斷準則。賺的場景是 overhead-bound，kernel 小、串很長、batch 小。典型例子是小 batch 推論和 LLM 的 decode 迴圈，每吐一個 token 都要把整個網路從頭走一遍，一次卻只算一個 token 的量，正好是幾百顆小 kernel 排成長串的形狀，而且 shape 固定、輸入位址可控，錄影的限制一條都不礙事，這也是 vLLM 這類推論引擎都內建 CUDA Graph 的原因。不賺的場景是 compute-bound，大 batch 訓練裡 kernel 動輒幾百 us，1.01 倍的差距換不回多吃的記憶體和押死位址的不自由。判斷方法也簡單，profiler 裡 GPU 時間軸的空隙就是可以收的帳，空隙不多就別開。開了之後記得看一眼 `TORCH_LOGS=perf_hints`，如果印出 skipping cudagraphs，那你付了限制卻沒拿到加速。

## 結語

執行期的這筆帳今天收完了。launch overhead 是 CPU 每次發射 kernel 的固定成本，編譯管不到它，CUDA Graph 用 capture 和 replay 把幾十次發射收成一次，reduce-overhead 靠 cudagraph trees 把錄影、共用 memory pool、跨 graph break 這些髒活自動化，代價是位址押死、輸出會被覆寫、shape 要穩定、記憶體多吃一些。batch 8 的 9.45 倍和 batch 8192 的 1.01 倍放在一起，就是這個模式的完整說明書，收的是 overhead 的帳，模型本身算得越重，這筆帳就越不值得收。

不過剛剛有一句話埋了雷，「shape 要穩定」。不只 CUDA Graph 怕 shape 變來變去，整個 torch.compile 都怕，Day 6 的 Guard 每失敗一次就重編一次，錄影機還得跟著重錄。明天就來看 recompilation 爆炸這個 Part 4 最常見的事故，怎麼發生、怎麼診斷、怎麼修。那我們明天見！

## 參考資料

- [torch/_inductor/cudagraph_trees.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cudagraph_trees.py)
- [torch/_inductor/cudagraph_utils.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cudagraph_utils.py)
- [torch/_inductor/__init__.py：list_mode_options（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/__init__.py)
- [Accelerating PyTorch with CUDA Graphs（PyTorch Blog）](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [torch.cuda.CUDAGraph（PyTorch Docs）](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)
