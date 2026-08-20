# 一行 torch.compile 背後發生了什麼？30 天深度拆解 PyTorch 編譯器

## Part 0：全景

- [Day 1 | torch.compile 是怎麼長出來的？](https://ithelp.ithome.com.tw/articles/10403466)
- [Day 2 | torch.compile 之後，你的 Python 去哪了？](https://ithelp.ithome.com.tw/articles/10403473)

## Part 1：TorchDynamo

- Day 3 | Dynamo 憑什麼攔得住你的 Python？
- Day 4 | 沒有真值，Dynamo 怎麼把 bytecode 走完？
- Day 5 | Dynamo 眼中的每一個 Python 值：VariableTracker 與 Source
- Day 6 | Guard：這張圖什麼時候還能用？
- Day 7 | self.counter += 1 怎麼過純函數的圖？SideEffects
- Day 8 | OutputGraph：散落的產出怎麼收成一張圖
- Day 9 | PyCodegen：新 bytecode 是怎麼生出來的
- Day 10 | Graph Break：斷在哪裡，怎麼接回來
- Day 11 | Symbolic Shapes：讓一張圖吃下所有 batch size

## Part 2：AOTAutograd

- Day 12 | AOTAutograd：為什麼 backward 也要先編好
- Day 13 | Functionalization：把 in-place 變不見
- Day 14 | Decomposition：兩千個 op 拆成幾百個
- Day 15 | Joint Graph 與 Partitioner：backward 該存什麼、該重算什麼
- Day 16 | Min-cut Recomputation：重算還是存下來
- Day 17 | FakeTensor 與 Meta Device：不算數值也能 Trace

## Part 3：TorchInductor

- Day 18 | Inductor 總覽：從 FX Graph 到 Kernel 的路
- Day 19 | Lowering 與 Loop-level IR：圖怎麼變成迴圈
- Day 20 | Scheduler：誰跟誰可以融合
- Day 21 | Fusion 的邊界：垂直、水平與 Reduction
- Day 22 | Triton Codegen：讀懂 Inductor 生出來的 GPU Kernel
- Day 23 | C++ Codegen：CPU 後端與 OpenMP
- Day 24 | Autotune 與 max-autotune：讓機器自己挑 Kernel
- Day 25 | 快取：編譯結果存在哪裡，什麼時候失效

## Part 4：整合與實戰

- Day 26 | CUDA Graph 與 reduce-overhead：壓掉 Kernel 啟動開銷
- Day 27 | Recompilation 爆炸：怎麼發生，怎麼診斷，怎麼修
- Day 28 | 除錯工具箱：TORCH_LOGS、explain、depyf 與 Minifier
- Day 29 | 自己寫一個 Backend：從 FX Graph 接手
- Day 30 | 總結：什麼時候快、為什麼慢、怎麼修
