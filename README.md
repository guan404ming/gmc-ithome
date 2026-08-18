# 一行 torch.compile 背後發生了什麼？30 天深度拆解 PyTorch 編譯器

2026 iThome 鐵人賽系列文章的原稿、程式碼與圖檔。系列頁：https://ithelp.ithome.com.tw/users/20183617/ironman/9241

## 30 天目錄

### Part 0：全景

- [Day 1 | torch.compile 是怎麼長出來的？](https://ithelp.ithome.com.tw/articles/10403466)
- [Day 2 | torch.compile 之後，你的 Python 去哪了？](https://ithelp.ithome.com.tw/articles/10403473)

### Part 1：TorchDynamo

- Day 3 | Dynamo 憑什麼攔得住你的 Python？
- Day 4 | 沒有真值，Dynamo 怎麼把 bytecode 走完？
- Day 5 | Dynamo 眼中的每一個 Python 值：VariableTracker 與 Source
- Day 6 | Guard：這張圖什麼時候還能用？
- Day 7 | SideEffects：會改東西的 Python，怎麼過純函數的圖
- Day 8 | OutputGraph：散落的產出怎麼收成一張 FX Graph
- Day 9 | PyCodegen：Dynamo 怎麼把新 Bytecode 寫回 CPython
- Day 10 | Graph Break 全機制：斷在哪裡，怎麼接回來
- Day 11 | Symbolic Shapes：讓一張圖吃下所有 batch size

### Part 2：AOTAutograd

- Day 12 | AOTAutograd 總覽：為什麼 backpropagation 也要 Ahead-of-Time
- Day 13 | Functionalization：In-place 與 View 怎麼被改寫成純函數
- Day 14 | Decomposition 與 PrimTorch：兩千個 Operator 拆成幾百個
- Day 15 | Joint Graph 與 Partitioner：forward 與 backpropagation 的切分
- Day 16 | Min-cut Recomputation：重算還是存下來
- Day 17 | FakeTensor 與 Meta Device：不算數值也能 Trace

### Part 3：TorchInductor

- Day 18 | Inductor 總覽：從 FX Graph 到 Kernel 的路
- Day 19 | Lowering 與 Loop-level IR：圖怎麼變成迴圈
- Day 20 | Scheduler：誰跟誰可以融合
- Day 21 | Fusion 的邊界：垂直、水平與 Reduction
- Day 22 | Triton Codegen：讀懂 Inductor 生出來的 GPU Kernel
- Day 23 | C++ Codegen：CPU 後端與 OpenMP
- Day 24 | Autotune 與 max-autotune：讓機器自己挑 Kernel
- Day 25 | 快取：編譯結果存在哪裡，什麼時候失效

### Part 4：整合與實戰

- Day 26 | CUDA Graph 與 reduce-overhead：壓掉 Kernel 啟動開銷
- Day 27 | Recompilation 爆炸：怎麼發生，怎麼診斷，怎麼修
- Day 28 | 除錯工具箱：TORCH_LOGS、explain、depyf 與 Minifier
- Day 29 | 自己寫一個 Backend：從 FX Graph 接手
- Day 30 | 總結：什麼時候快、為什麼慢、怎麼修


## 目錄結構

```
posts/        每天的文章原稿（Markdown），README.md 是 30 天的標題規劃
code/         文章裡跑的實驗程式，依天數分資料夾，附輸出 log
assets/       文章用到的圖與動畫，依天數分資料夾
animations/   產生動畫的 manim 場景檔與字型
```

## 重現實驗

程式都在 [Modal](https://modal.com/) 上跑（GPU: L40S，PyTorch 2.8.0 + CUDA 12.8）：

```bash
cd code/day02
modal run bench.py
```

## 重新輸出動畫

```bash
uv venv .venv && uv pip install --python .venv/bin/python manim
.venv/bin/manim -qh animations/day02_pipeline.py Pipeline
ffmpeg -i media/videos/day02_pipeline/1080p60/Pipeline.mp4 -vf "fps=30,scale=1280:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff" pal.png
ffmpeg -i media/videos/day02_pipeline/1080p60/Pipeline.mp4 -i pal.png -lavfi "fps=30,scale=1280:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle" assets/day02/pipeline.gif
```
