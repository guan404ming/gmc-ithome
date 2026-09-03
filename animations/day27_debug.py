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


def chip(label, color=TXT, fill=CARD_DIM):
    t = T(label, font=MONO, font_size=16, color=color)
    bg = RoundedRectangle(corner_radius=0.12, width=t.width + 0.36, height=0.44, stroke_color=EDGE, stroke_width=1.2, fill_color=fill, fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def station(name):
    t = T(name, font=MONO, font_size=16, color=MUTED)
    box = RoundedRectangle(corner_radius=0.14, width=t.width + 0.5, height=0.66, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    return VGroup(box, t.move_to(box))


def card(l1, l2, w=6.3):
    a = T(l1, font=MONO, font_size=16, color=TXT)
    b = T(l2, font=MONO, font_size=16, color=MUTED)
    body = VGroup(a, b).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    r = RoundedRectangle(corner_radius=0.12, width=w, height=body.height + 0.36, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD_DIM, fill_opacity=1)
    body.move_to(r).align_to(r.get_left() + RIGHT * 0.22, LEFT)
    return VGroup(r, body)


SX = -4.0
YS = [0.5, -0.55, -1.6, -2.65]
CX = 3.2


class Debug(Scene):
    def construct(self):
        title = T("torch.compile(f)  ·  still slow", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        names = ["TORCH_LOGS=graph_breaks", "TORCH_LOGS=recompiles", "TORCH_LOGS=output_code", "depyf.prepare_debug"]
        stations = VGroup(*[station(n).move_to([SX, y, 0]) for n, y in zip(names, YS)])
        flow = VGroup(*[Line(stations[i][0].get_bottom(), stations[i + 1][0].get_top(), stroke_color=DIM, stroke_width=2.4) for i in range(3)])
        code_lines = ["y = torch.sin(x) + 1", "if y.sum() > 0:", "    y = y * 2", "return torch.relu(y) * n"]
        code = VGroup(*[T(l, font=MONO, font_size=16, color=TXT) for l in code_lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        codebox = RoundedRectangle(corner_radius=0.12, width=4.6, height=code.height + 0.44, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
        codebox.move_to([CX, 0, 0]).align_to([0, title.get_center()[1] - 0.52, 0], UP)
        code.move_to(codebox).align_to(codebox.get_left() + RIGHT * 0.28, LEFT)
        code[2].shift(RIGHT * 0.55)
        dot = Dot(radius=0.09, color=ACCENT).move_to(stations[0][0].get_left() + LEFT * 0.35 + UP * 1.1)

        switch("SYMPTOM", "掛號", "編好了卻沒變快，症狀掛號進診間，左手病歷右手決策表，一站一站檢查")
        self.play(FadeIn(codebox), FadeIn(code), run_time=0.5)
        self.play(FadeIn(stations, shift=UP * 0.15), FadeIn(flow), run_time=0.7)
        self.play(FadeIn(dot, scale=0.6), run_time=0.4)
        self.wait(3.5)

        cards = []

        def visit(i, l1, l2):
            self.play(dot.animate.move_to(stations[i][0].get_left() + LEFT * 0.35), run_time=0.5)
            self.play(stations[i][0].animate.set_stroke(ACCENT, 2.2), stations[i][1].animate.set_color(TXT), run_time=0.4)
            c = card(l1, l2).move_to([CX, YS[i], 0])
            ln = Line(stations[i][0].get_right(), c[0].get_left(), stroke_color=DIM, stroke_width=2)
            self.play(Create(ln), run_time=0.3)
            self.play(FadeIn(c, shift=RIGHT * 0.2), run_time=0.4)
            cards.append(VGroup(c, ln))
            return c

        switch("GRAPH BREAKS", "先數 break", "graph_breaks 指出第 21 行的資料相依分支，一個 break，圖被切成兩張")
        visit(0, "Data-dependent branching @ f:21", "Graph Count: 2  Break Count: 1")
        self.wait(2.5)

        switch("RECOMPILES", "再數 recompile", "recompiles 點名倒下的 guard，32 變 48，連 resume function 也重編一次")
        visit(1, "tensor 'x' size mismatch: 32 -> 48", "f + torch_dynamo_resume_in_f_at_21")
        self.wait(2.5)

        switch("OUTPUT CODE", "驗產物", "output_code 交出兩顆分家的 kernel，break 的痕跡一路留到產物層")
        visit(2, "cpp_fused_add_gt_sin_sum_0", "cpp_fused_mul_relu_0")
        self.wait(2.5)

        switch("DEPYF", "X 光", "depyf 把改寫後的 bytecode 攤成 Python，if 留在原地，兩條路各接一個 resume")
        visit(3, "if graph_out_0[0]:", "-> __resume_at_88 / __resume_at_98")
        self.wait(3.0)

        switch("DIAGNOSIS", "鎖定病灶", "四站證據都指向同一行，病灶就是那個資料相依的 if")
        self.play(dot.animate.move_to(stations[3][0].get_left() + LEFT * 0.35 + DOWN * 0.8).set_opacity(0), run_time=0.4)
        self.play(LaggedStart(*[c[1].animate(rate_func=there_and_back).set_stroke(ACCENT, 3.2) for c in cards], lag_ratio=0.2), run_time=1.4)
        hl = SurroundingRectangle(code[1], color=ACCENT, buff=0.08, corner_radius=0.06)
        self.play(Create(hl), code[1].animate.set_color(ACCENT), run_time=0.6)
        self.play(Flash(hl, color=ACCENT, line_length=0.14, flash_radius=1.6), run_time=0.5)
        self.wait(2.5)

        switch("RULE", "排查順序", "先數 break，再數 recompile，最後才懷疑 kernel，出事就照這個順序按")
        self.play(FadeOut(stations), FadeOut(flow), *[FadeOut(c) for c in cards], run_time=0.6)
        o1 = chip("1 · count breaks")
        o2 = chip("2 · count recompiles")
        o3 = chip("3 · blame the kernel")
        order = VGroup(o1, T("->", font=MONO, font_size=18, color=DIM), o2, T("->", font=MONO, font_size=18, color=DIM), o3).arrange(RIGHT, buff=0.35).move_to([-0.4, -0.6, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.15) for m in order], lag_ratio=0.15), run_time=1.0)
        self.play(Indicate(o1, scale_factor=1.08, color=ACCENT), run_time=0.5)
        self.wait(5.5)
