# 一行 torch.compile 背後發生了什麼？30 天深度拆解 PyTorch 編譯器

2026 iThome 鐵人賽系列文章的原稿、程式碼與圖檔。系列頁：https://ithelp.ithome.com.tw/users/20183617/ironman/9241

## 30 天目錄

### Part 0：全景

- [Day 1 | torch.compile 是怎麼長出來的？](https://ithelp.ithome.com.tw/articles/10403466)
- [Day 2 | torch.compile 之後，你的 Python 去哪了？](https://ithelp.ithome.com.tw/articles/10403473)

### Part 1：TorchDynamo

- [Day 3 | TorchDynamo？它憑什麼攔得住你的 Python？](https://ithelp.ithome.com.tw/articles/10403588)
- [Day 4 | 走進 TorchDynamo 的心臟：InstructionTranslator](https://ithelp.ithome.com.tw/articles/10403739)
- [Day 5 | TorchDynamo 裡有替身使者？ VariableTracker 登場！](https://ithelp.ithome.com.tw/articles/10403971)
- [Day 6 | TorchDynamo 的高速驗票員：Guard](https://ithelp.ithome.com.tw/articles/10404358)
- Day 7 | 改掉的值去哪了？TorchDynamo 的隨行記帳員 SideEffects
- Day 8 | TorchDynamo 的中央倉庫：OutputGraph
- Day 9 | 新的 bytecode 誰來寫？TorchDynamo PyCodegen！
- Day 10 | TorchDynamo 的破「圖」重接，Graph Break and Resume！
- Day 11 | TorchDynamo 的伸縮量尺 Symbolic Shapes

### Part 2：AOTAutograd

- Day 12 | torch.compile 的沙盤推演師：AOTAutograd
- Day 13 | 讓 in-place 消失的潔癖書記官 Functionalization
- Day 14 | AOTAutograd 的樂高拆解師：Decomposition
- Day 15 | backward 該存誰、該重算誰？min-cut 分家公證人
- Day 16 | 沒有數值要怎麼 trace？編譯管線的空殼替身 FakeTensor

### Part 3：TorchInductor

- Day 17 | torch.compile 的程式碼鑄造廠：TorchInductor
- Day 18 | Loop-level IR 不是一棵樹，是一條函式
- Day 19 | 誰跟誰可以同桌？TorchInductor 的宴席總管 Scheduler
- Day 20 | 磁鐵與牆：Inductor Fusion 的邊界
- Day 21 | 一人一塊磚？讀懂 Inductor 生出來的 Triton Kernel
- Day 22 | 同一條迴圈，C++ 後端有三段變速
- Day 23 | 讓碼表說話：Autotune 與 max-autotune
- Day 24 | 編譯費能只付一次嗎？torch.compile 的置物櫃與鑰匙

### Part 4：整合與實戰

- Day 25 | 把一整串 launch 錄成一次 replay：CUDA Graph 與 reduce-overhead
- Day 26 | Recompilation 爆炸：怎麼發生，怎麼診斷，怎麼修
- Day 27 | 除錯工具箱：TORCH_LOGS、explain、depyf 與 Minifier
- Day 28 | 自己寫一個 Backend：從 FX Graph 接手
- Day 29 | torch.export 與 AOTInductor：不帶 Python 也能跑
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
