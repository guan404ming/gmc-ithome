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
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    zt = T(zh, font=zh_font, font_size=17, color=BG)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def chip(name):
    t = T(name, font=MONO, font_size=16, color=TXT)
    bg = RoundedRectangle(corner_radius=0.12, width=t.width + 0.4, height=0.46, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD_DIM, fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def cells_row(n, w, y):
    cs = VGroup(*[RoundedRectangle(corner_radius=0.05, width=w, height=w, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD_DIM, fill_opacity=1) for _ in range(n)])
    cs.arrange(RIGHT, buff=0.05).move_to([0, y, 0])
    return cs


ROW_Y = 0.35
N = 32


class Cpp(Scene):
    def construct(self):
        title = T("f(x, y) -> relu(x + y) * 2", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("IR", "一條迴圈", "lowering 完的 loop-level IR 只說每一輪做什麼，誰來跑、跑多寬，輪到 codegen 決定")
        ir_lines = VGroup(
            T("for i in range(N):", font=MONO, font_size=17, color=TXT),
            T("  tmp = load(x, i) + load(y, i)", font=MONO, font_size=17, color=TXT),
            T("  store(out, i, relu(tmp) * 2)", font=MONO, font_size=17, color=TXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        ir_lines[1].shift(RIGHT * 0.35)
        ir_lines[2].shift(RIGHT * 0.35)
        ir_box = RoundedRectangle(corner_radius=0.16, width=ir_lines.width + 0.8, height=ir_lines.height + 0.6, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
        ir = VGroup(ir_box, ir_lines.move_to(ir_box)).move_to([0, 0.9, 0])
        self.play(FadeIn(ir, shift=UP * 0.15), run_time=0.7)
        self.wait(3)

        switch("SPLIT", "雙後端分流", "同一條 IR 走到 codegen 才分家，GPU 生 Triton kernel，CPU 生 C++ 迴圈")
        self.play(ir.animate.scale(0.85).move_to([0, 1.7, 0]), run_time=0.6)
        gchip = chip("triton · GPU").move_to([-3.4, 0.15, 0])
        cchip = chip("cpp · CPU").move_to([3.4, 0.15, 0])
        ga = Line(ir.get_bottom() + LEFT * 0.4, gchip.get_top() + UP * 0.08, stroke_color=DIM, stroke_width=2.4)
        ca = Line(ir.get_bottom() + RIGHT * 0.4, cchip.get_top() + UP * 0.08, stroke_color=DIM, stroke_width=2.4)
        tiles = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.42, height=0.42, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD_DIM, fill_opacity=1) for _ in range(18)])
        tiles.arrange_in_grid(rows=3, cols=6, buff=0.09).next_to(gchip, DOWN, buff=0.4)
        mini = cells_row(8, 0.34, 0).next_to(cchip, DOWN, buff=0.4)
        self.play(Create(ga), Create(ca), FadeIn(gchip), FadeIn(cchip), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(t, scale=0.7) for t in tiles], lag_ratio=0.02), LaggedStart(*[FadeIn(c, scale=0.7) for c in mini], lag_ratio=0.05), run_time=0.9)
        self.wait(3)

        switch("GPU", "千軍萬馬", "Triton 假設 thread 要多少有多少，每個 program 抓一塊 tile，一次全部點亮")
        self.play(*[t.animate.set_fill(ACCENT, 0.9) for t in tiles], run_time=0.5)
        self.play(Flash(tiles, color=ACCENT, line_length=0.16, flash_radius=1.7), run_time=0.5)
        self.wait(3)

        switch("SCALAR", "第一段變速", "CPU 這條先是純量迴圈，一步一格、一次算一個 float，32 格要走 32 步")
        cells = cells_row(N, 0.26, ROW_Y)
        self.play(FadeOut(ir), FadeOut(ga), FadeOut(ca), FadeOut(gchip), FadeOut(tiles), FadeOut(cchip), ReplacementTransform(mini, cells), run_time=0.9)
        slabel = T("steps", font=MONO, font_size=16, color=MUTED)
        sval = T("0", font=MONO, font_size=22, color=TXT)
        steps = VGroup(slabel, sval).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(steps, shift=UP * 0.1), run_time=0.3)
        cursor = SurroundingRectangle(cells[0], color=ACCENT, stroke_width=2.4, buff=0.045)
        self.play(Create(cursor), run_time=0.3)
        for i in range(6):
            nv = T(str(i + 1), font=MONO, font_size=22, color=TXT).move_to(sval, aligned_edge=LEFT)
            self.play(cursor.animate.move_to(cells[i]), cells[i].animate.set_fill(ACCENT, 0.9), Transform(sval, nv), run_time=0.28)
        nv = T("32", font=MONO, font_size=22, color=TXT).move_to(sval, aligned_edge=LEFT)
        self.play(cursor.animate.move_to(cells[N - 1]), LaggedStart(*[c.animate.set_fill(ACCENT, 0.9) for c in cells[6:]], lag_ratio=0.04), Transform(sval, nv), run_time=1.6)
        tag_s = chip("scalar · 32 steps").move_to([-3.5, ROW_Y - 1.15, 0])
        self.play(FadeIn(tag_s, shift=UP * 0.1), FadeOut(cursor), run_time=0.4)
        self.wait(3)

        switch("SIMD", "第二段變速", "at::vec::Vectorized 把 4 格併成一步，NEON 一包 4 個 float，32 格剩 8 步")
        nv = T("0", font=MONO, font_size=22, color=TXT).move_to(sval, aligned_edge=LEFT)
        self.play(*[c.animate.set_fill(CARD_DIM, 1) for c in cells], Transform(sval, nv), run_time=0.5)
        vcursor = SurroundingRectangle(VGroup(*cells[0:4]), color=ACCENT, stroke_width=2.4, buff=0.05)
        self.play(Create(vcursor), run_time=0.3)
        for g in range(8):
            nv = T(str(g + 1), font=MONO, font_size=22, color=TXT).move_to(sval, aligned_edge=LEFT)
            grp = cells[g * 4 : g * 4 + 4]
            self.play(vcursor.animate.move_to(VGroup(*grp)), *[c.animate.set_fill(ACCENT, 0.9) for c in grp], Transform(sval, nv), run_time=0.3)
        tag_v = chip("vectorized · 8 steps").move_to([0, ROW_Y - 1.15, 0])
        self.play(FadeIn(tag_v, shift=UP * 0.1), FadeOut(vcursor), run_time=0.4)
        self.wait(3)

        switch("OMP", "第三段變速", "#pragma omp parallel 把迴圈切給 8 個 worker，每人一段、段內照樣 SIMD")
        nv = T("0", font=MONO, font_size=22, color=TXT).move_to(sval, aligned_edge=LEFT)
        self.play(*[c.animate.set_fill(CARD_DIM, 1) for c in cells], Transform(sval, nv), run_time=0.5)
        workers = VGroup(*[T(f"t{i}", font=MONO, font_size=16, color=MUTED).next_to(VGroup(*cells[i * 4 : i * 4 + 4]), UP, buff=0.22) for i in range(8)])
        divs = VGroup(*[DashedLine(cells[i * 4].get_corner(UL) + LEFT * 0.045 + UP * 0.14, cells[i * 4].get_corner(DL) + LEFT * 0.045 + DOWN * 0.14, stroke_color=DIM, stroke_width=1.8, dash_length=0.07) for i in range(1, 8)])
        self.play(LaggedStart(*[FadeIn(w, shift=UP * 0.08) for w in workers], lag_ratio=0.06), Create(divs), run_time=0.8)
        nv = T("1", font=MONO, font_size=22, color=ACCENT).move_to(sval, aligned_edge=LEFT)
        self.play(*[c.animate.set_fill(ACCENT, 0.9) for c in cells], *[w.animate.set_color(ACCENT) for w in workers], Transform(sval, nv), run_time=0.6)
        self.play(Flash(sval, color=ACCENT, line_length=0.12, flash_radius=0.45), run_time=0.4)
        tag_o = chip("omp x8 · 1 step").move_to([3.5, ROW_Y - 1.15, 0])
        self.play(FadeIn(tag_o, shift=UP * 0.1), run_time=0.4)
        self.wait(3)

        switch("GATE", "換檔門檻", "thread 不是免費的，每條分不到 min_chunk_size 4096 個元素就不開這一檔")
        gate1 = T("numel / threads < 4096  ->  single thread", font=MONO, font_size=17, color=MUTED)
        gate2 = T("n=16384: single    n=32768: omp parallel", font=MONO, font_size=17, color=TXT)
        gates = VGroup(gate1, gate2).arrange(DOWN, buff=0.24).next_to(workers, UP, buff=0.5)
        self.play(FadeIn(gates, shift=UP * 0.1), run_time=0.5)
        self.play(gate2[22:33].animate.set_color(ACCENT), run_time=0.4)
        self.wait(3)

        switch("RESULT", "三段變速", "同一條 IR，純量 32 步、SIMD 8 步、OpenMP 一步，換不換檔由 shape 在編譯期決定")
        self.play(FadeOut(gates), run_time=0.3)
        for tag in (tag_s, tag_v, tag_o):
            self.play(tag[0].animate.set_stroke(ACCENT, 2), run_time=0.45)
        self.wait(5.5)
