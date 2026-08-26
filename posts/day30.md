# Day 30 | 總結：什麼時候快、為什麼慢、怎麼修

## 前言

30 天前，這個系列從一個很樸素的問題出發，一行 `torch.compile` 背後到底發生了什麼。系列開場就說它常被當成黑魔法，加上去就變快，卻很少人說得清快在哪裡、為什麼有時候沒效。今天是最後一天，不打算再打開任何新的原始碼，而是把前面二十九天拆下來的零件收回同一張桌上，拼成一張帶得走的地圖。這張地圖其實只需要回答三個問題，什麼時候快、為什麼慢、壞了怎麼修。正文開始！

## 先把 30 天收成一張地圖

一行 `torch.compile` 落下之後，你的程式會依序經過四站。這是系列開頭畫過的骨架，只是現在每一站的行李都比當時重得多。

第一站 Dynamo 負責把圖從 Python 手裡拿出來。Day 3 看到它靠 CPython 的 frame evaluation hook 在 bytecode 執行前攔截，Day 4 和 Day 5 看到它逐條模擬指令、用替身物件追蹤每一個值，Day 6 的 Guard 記下這張圖成立的前提，Day 7 到 Day 9 則把被改動的狀態記帳、把收好的圖存進中央倉庫、最後改寫出一份新的 bytecode。碰到吃不下的東西就 graph break，Day 10 看到一次 break 會把函式切成兩張圖中間夾一段 eager，Day 11 讓 shape 可以是符號，一張圖吃下所有 batch size。

第二站 AOTAutograd 把 forward 圖展開成能訓練的樣子。Day 12 用一次沙盤推演把 forward 和 backward 一起 trace 成 joint graph，Day 13 的 functionalization 讓 in-place 和 view 從圖裡消失，Day 14 把大 op 拆解成基本運算，Day 15 用 min-cut 決定 backward 該存誰、該重算誰。而這一切都踩在 Day 16 的 FakeTensor 上，只推形狀、不算數值。

第三站 Inductor 把圖變成真的程式碼。Day 17 和 Day 18 把 ATen 圖 lower 成 loop-level IR，Day 19 和 Day 20 由 scheduler 依讀寫關係決定誰跟誰融合，Day 21 和 Day 22 分別生出 GPU 的 Triton 和 CPU 的 C++，Day 23 用 autotune 拿碼表實測挑出最快的寫法，Day 24 把成品鎖進快取，讓編譯費只付一次。

最後一段是執行期與實戰。Day 25 的 CUDA Graph 把整串 kernel launch 錄成一次 replay，Day 26 診斷 recompilation 爆炸，Day 27 整理除錯工具箱，Day 28 自己動手寫了一個 backend，Day 29 則用 torch.export 和 AOTInductor 讓模型不帶 Python 也能跑。

值得再提一次的是，這四站不是黑箱接黑箱。每一站的中間產物都印得出來，Dynamo 吐的 FX Graph、AOTAutograd 展開後的 ATen 圖、Inductor 生出的 kernel，系列裡每一篇的實驗都是靠這些產物對照原始碼講的。地圖上任何一站出了狀況，你都有辦法停在那一站，把它手上的東西攤開來看。

![一行 torch.compile 沿著四站走過，每站亮起累積的關鍵字，最後全管線點亮又收回那一行](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day30/recap.gif)

*圖一：30 天的路線圖。一行 torch.compile 落下，鏡頭沿管線走過四站，Dynamo 亮起 eval hook 與 Guard，AOTAutograd 亮起 joint graph 與 min-cut，Inductor 亮起 fusion 與 Triton，執行期亮起 cache 與 CUDA Graph，最後整條管線一起點亮，收回最初的那一行。*

## 什麼時候快

系列裡實測過的加速來源有三種，它們共同的特徵是各自都挑體質。

第一種是 fusion 省下記憶體流量。開場那個 sin 乘 cos 加 tanh 的函式，eager 下五個 kernel 要搬大約 768 MB，融成一顆 Triton kernel 之後只剩 128 MB，量到的加速是 6.12 倍，幾乎就是流量的比值。Scheduler 那一站進一步看到 Inductor 給融合打分的單位，根本就是省下的 byte 數。吃這招的體質是 pointwise 密集、被記憶體頻寬綁住的運算，matmul 這種 extern kernel 就融不太動。

第二種是特化與 autotune 用資訊換速度。Guard 讓 Dynamo 敢把 Python 常數和 shape 直接烤進圖裡，省掉所有執行期的動態判斷。autotune 則拿真實 shape 上機實測，實驗裡方正的 2048 矩陣本來就是 cuBLAS 的主場，開了 max-autotune 也只是打平，換成 16x4096 的瘦長條，Triton 模板就贏出 1.48 倍。吃這招的體質是 shape 固定、又長得剛好偏離現成函式庫特調範圍的工作負載。

第三種是 reduce-overhead 收 launch 的帳。CUDA Graph 那組 32 層小 Linear 在 batch 8 時快了 9.45 倍，batch 放大到 8192 就只剩 1.01 倍。吃這招的體質是 kernel 小、串很長的 overhead-bound 模型，典型就是小 batch 推論和 LLM 的 decode 迴圈。

同一行 `torch.compile`，三種收益對應三種完全不同的體質。這也是為什麼「加了會快多少」這題永遠只能實測。

## 為什麼慢

慢的原因，系列裡也量過三種。

第一種是 graph break 把圖切碎。一個 print 就讓函式裂成兩張圖夾一段 eager，而藏在被 inline 函式深處的 break 還會往上傳染，一個工具函式裡的 print 換來四次編譯。圖一碎，跨不過斷點的 fusion 機會就全沒了，執行期進出 Dynamo 的次數也跟著變多。

第二種是 recompile。Guard 每失敗一次就重編一次，batch size 一變就會觸發。automatic dynamic 會在第二次把 shape 升級成符號，但如果程式裡藏著針對 shape 的 if，符號會被默默押死，每個新 shape 都重編一輪，撞上預設 8 次的上限之後整個 frame 退回 eager，全程沒有錯誤訊息，只有越跑越慢。

第三種是編譯本身的成本。開場實驗的第一次呼叫花了 1592.9 ms，是之後每次呼叫的八千倍。快取那組實驗量得更細，一個三行小函式冷編譯要 3.75 秒，靠磁碟快取的下一個 process 降到 0.79 秒，但 shape、config、torch 版本任何一項變動都會讓快取的鑰匙對不上，整段重付。

這三種慢有一個共同點，它們都不是 bug，而是設計裡明碼標價的成本。break 是「保住 Eager 語意」的代價，recompile 是「敢做特化」的代價，冷編譯是「執行期什麼都不用判斷」的代價。問題從來不是成本存不存在，而是你付出去的有沒有賺回來。一個每小時重啟一次、每次都冷編譯幾分鐘的服務，和一個編一次跑一個月的訓練 job，對同一筆帳的感受會完全不同。

## 怎麼修：三步排查

把系列裡散落的工具，收成一個固定的排查順序。

第一步數 break。`torch._dynamo.explain` 列出每張圖、每個 break 的位置和原因，`fullgraph=True` 把 break 升級成錯誤，`TORCH_LOGS="graph_breaks"` 適合掛在長跑的 job 上，這是數 graph break 的工具箱。記得兇手常常不在報告的第一行，而在被 inline 的函式深處。

第二步數 recompile。`TORCH_LOGS="recompiles"` 會印出是哪一條 Guard 沒過，shape 引起的可以用 `mark_dynamic` 提前宣告，順便把「符號被默默押死」變成大聲的失敗，這是應付 dynamic shape 的做法。啟動變慢則照查快取的順序，先看 counters 是 hit 還是 miss，再開 log 對鑰匙，是 shape 變了、config 變了，還是有人升了版本。

第三步看產物。結果不對，就用 backend 參數把 pipeline 切段，eager、aot_eager、inductor 一段一段驗，錯在哪一段就往那一層鑽。速度不如預期，就開 `TORCH_LOGS` 把三段中間產物印出來，看圖有沒有被切碎、誰跟誰融了、生出來的 kernel 是不是你想的那顆，perf_hints 還會直接告訴你 CUDA Graph 有沒有真的上車。

## 一行背後的 30 天

如果要把整個系列壓縮成一個觀點，筆者會選這個，編譯器就是一連串「先付出、後回收」的賭注。Dynamo 付出 trace 的時間，賭 Guard 之後每次都會通過。特化把常數烤死，賭它不會變。fusion 和 autotune 燒掉編譯時間，賭執行的次數夠多。CUDA Graph 押死位址，賭 shape 穩定。快，就是賭贏了。慢，就是賭輸了。修，就是弄清楚它替你下了什麼注，然後幫它把注下對。這個視角建立之後，`torch.compile` 的每個行為都不再神秘，它只是一個把賠率誠實寫在 log 裡的賭徒。

系列到這裡就完賽了。第一天說過，這些問題其實都有很明確的答案，只是需要耐心打開那個超大的黑盒子。三十天下來，希望這個盒子在你眼裡已經變得透明。整個系列的文章、實驗程式、log 和動畫都整理在 [gmc-ithome](https://github.com/guan404ming/gmc-ithome)，每一篇引用的數字都能在對應的 code 目錄裡重跑出來。

如果想繼續往下走，給三個方向。第一個是官方的 Dynamo deep dive 和 troubleshooting 文件，它們是把這個系列對回官方語彙最快的路，很多這裡用比喻講的東西，那邊有正式的名字。第二個是 ASPLOS 的 PyTorch 2 論文，開場就說它是整套設計的正式論述，值得完整讀一遍，這時候的你讀起來應該已經完全不吃力。最後是原始碼本身，挑一個系列講過的檔案，開著 log 拿自己的模型對照著讀，看到不懂的分支就寫個小實驗把它踩出來，那是理解得最深的一條路，也是這三十天每一篇的寫法。

最後，謝謝一路讀到這裡的你，也謝謝「源來適你」的小夥伴們陪著一起走完。第一次參賽就選了一個這麼硬的題目，能寫完靠的是每天知道有人在等更新。願你下次再看到那一行 `torch.compile` 的時候，看到的不是黑魔法，而是一條你親手走過的 pipeline。

## 參考資料

- Ansel et al., [*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024
- [torch.compiler 概觀](https://pytorch.org/docs/stable/torch.compiler.html)
- [Dynamo Deep Dive](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.compile 除錯與疑難排解](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- [系列 repo：gmc-ithome](https://github.com/guan404ming/gmc-ithome)
