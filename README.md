# 一行 torch.compile 背後發生了什麼？30 天深度拆解 PyTorch 編譯器

2026 iThome 鐵人賽系列文章的原稿、程式碼與圖檔。

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
