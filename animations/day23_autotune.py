import re
from pathlib import Path

import manimpango
from manim import *

FONT_DIR = Path(__file__).parent / "fonts"
for f in FONT_DIR.glob("*.ttf"):
    manimpango.register_font(str(f))

BG = "#161719"
CARD = "#23272e"
CARD_DIM = "#1b1e23"
EDGE = "#3a3f47"
TXT = "#e8e6e3"
MUTED = "#8b8f96"
DIM = "#5a5e66"
ACCENT = "#e8622a"
config.background_color = BG
MONO = "Menlo"
SANS = "TASA Orbiter"
CJK = "PingFang TC"


def T(txt, font_size, **kw):
    return Text(txt, font_size=font_size * 4, **kw).scale(0.25)


def pill(name, zh):
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    runs = re.findall(r"[一-鿿，、。]+|[^一-鿿，、。 ]+", zh)
    zs = [T(r, font=CJK if re.search(r"[一-鿿]", r) else MONO, font_size=17, color=BG) for r in runs]
    zt = VGroup(*zs).arrange(RIGHT, buff=0.1)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def card(name, sub, w=3.3):
    nm = T(name, font=MONO, font_size=16, color=TXT)
    sb = T(sub, font=MONO, font_size=16, color=MUTED)
    inner = VGroup(nm, sb).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
    box = RoundedRectangle(corner_radius=0.12, width=w, height=0.82, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    inner.move_to(box.get_left() + RIGHT * 0.28, aligned_edge=LEFT)
    return VGroup(box, inner)


ROWS_Y = [1.75, 0.85, -0.05, -0.95, -1.85]
CARD_X = -6.35
BAR_X0 = -2.75
BAR_SCALE = 47


class Autotune(Scene):
    def construct(self):
        title = T("y = x @ w   (2048, 2048, 2048) fp16", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("NODE", "一顆 mm", "lowering 走到 aten.mm，tuned_mm 不急著生 kernel，先開一張候選名單")
        node = card("aten.mm", "2048 x 2048 x 2048", w=3.6).move_to([0, 0.4, 0])
        self.play(FadeIn(node, shift=UP * 0.15), run_time=0.6)
        self.wait(3)

        switch("CANDIDATES", "分身", "extern 的 cuBLAS，加上一排不同 BLOCK 配置的 Triton template，每個都算得出正確答案")
        cands = [
            card("mm", "extern · cuBLAS"),
            card("triton_mm_16", "128x128x32 · w4"),
            card("triton_mm_17", "128x128x64 · w4"),
            card("triton_mm_9", "64x128x32 · w4"),
            card("triton_mm_7", "64x64x64 · w8"),
        ]
        for c, y in zip(cands, ROWS_Y):
            c.move_to([CARD_X, y, 0], aligned_edge=LEFT)
        self.play(ReplacementTransform(node, cands[0]), LaggedStart(*[FadeIn(c, shift=LEFT * 0.3) for c in cands[1:]], lag_ratio=0.12), run_time=1.2)
        start = DashedLine([BAR_X0, ROWS_Y[0] + 0.55, 0], [BAR_X0, ROWS_Y[-1] - 0.55, 0], stroke_color=DIM, stroke_width=2, dash_length=0.12)
        self.play(Create(start), run_time=0.5)
        self.wait(3.2)

        switch("BENCH", "同場計時", "AlgorithmSelectorCache 拿真的 tensor 把每個候選各跑一遍，碼表一支一支按下去")
        times = [0.0848, 0.0870, 0.0932, 0.1004, 0.1239]
        bars, labels = [], []
        for c, y, t in zip(cands, ROWS_Y, times):
            bar = Rectangle(width=0.01, height=0.34, stroke_width=0, fill_color=MUTED, fill_opacity=0.85).move_to([BAR_X0, y, 0], aligned_edge=LEFT)
            lab = T(f"{t:.4f} ms", font=MONO, font_size=16, color=TXT)
            self.add(bar)
            self.play(bar.animate.stretch_to_fit_width(t * BAR_SCALE).move_to([BAR_X0, y, 0], aligned_edge=LEFT), run_time=0.4 + t * 4, rate_func=linear)
            lab.next_to(bar, RIGHT, buff=0.2)
            self.play(FadeIn(lab), run_time=0.2)
            bars.append(bar)
            labels.append(lab)
        oom = T("3 choices out of resource · ignored", font=MONO, font_size=16, color=DIM).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(oom, shift=UP * 0.08), run_time=0.4)
        self.wait(3)

        switch("PICK", "挑最快", "min(timings) 定案，這一題 cuBLAS 贏了，最快的 Triton 候選只差 3%")
        self.play(*[c[1].animate.set_opacity(0.35) for c in cands[1:]], *[b.animate.set_opacity(0.3) for b in bars[1:]], *[l.animate.set_opacity(0.35) for l in labels[1:]], *[c[0].animate.set_fill(CARD_DIM) for c in cands[1:]], FadeOut(oom), run_time=0.7)
        self.play(cands[0][0].animate.set_stroke(ACCENT, 2.4), bars[0].animate.set_fill(ACCENT, 1), labels[0].animate.set_color(ACCENT), run_time=0.5)
        self.play(Flash(cands[0], color=ACCENT, line_length=0.16, flash_radius=1.1), run_time=0.5)
        self.wait(3.2)

        switch("COST", "代價", "為了這個答案，20 個候選精編 2.8 秒、實測 0.7 秒，編譯時間從 1.0 秒漲到 3.8 秒")
        cost = T("compile  1.0 s -> 3.8 s", font=MONO, font_size=18, color=MUTED).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT)
        cost[8:].set_color(ACCENT)
        self.play(FadeIn(cost, shift=UP * 0.1), run_time=0.5)
        self.wait(3.2)

        switch("CACHE", "蓋章", "冠軍寫進 autotune 快取，同一組 shape 下次不用再比，直接翻答案")
        stxt = T("CACHED", font=SANS, font_size=20, weight=BOLD, color=ACCENT)
        sbox = RoundedRectangle(corner_radius=0.1, width=stxt.width + 0.4, height=0.52, stroke_color=ACCENT, stroke_width=2.4, fill_opacity=0)
        stamp = VGroup(sbox, stxt.move_to(sbox)).rotate(-8 * DEGREES).move_to(cands[0]).shift(RIGHT * 0.75 + UP * 0.06)
        self.play(FadeIn(stamp, scale=1.6), run_time=0.5)
        self.play(Flash(stamp, color=ACCENT, line_length=0.12, flash_radius=0.9), run_time=0.4)
        self.wait(5.5)
