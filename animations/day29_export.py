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


def mixed(s, size, color=TXT, font=MONO):
    t2f = {ch: CJK for ch in s if "一" <= ch <= "鿿"}
    return T(s, font=font, font_size=size, color=color, t2f=t2f)


def chip(s, edge=EDGE, color=TXT, fill=CARD_DIM, size=16):
    t = mixed(s, size, color=color)
    r = RoundedRectangle(corner_radius=0.24, width=t.width + 0.55, height=0.58, stroke_color=edge, stroke_width=1.5, fill_color=fill, fill_opacity=1)
    return VGroup(r, t.move_to(r))


def card(tag, lines, w=6.6, tag_color=ACCENT, size=16):
    tg = mixed(tag, 16, color=tag_color)
    body = VGroup(*[mixed(s, size) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    inner = VGroup(tg, body).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
    box = RoundedRectangle(corner_radius=0.16, width=w, height=inner.height + 0.6, stroke_color=EDGE, stroke_width=1.6, fill_color=CARD, fill_opacity=1)
    inner.move_to(box).align_to(box.get_left() + RIGHT * 0.4, LEFT)
    return VGroup(box, inner)


def arr(a, b, color=ACCENT):
    return Arrow(a, b, buff=0.12, color=color, stroke_width=2.5, tip_length=0.16)


class Export(Scene):
    def construct(self):
        title = T("Tiny(x) = relu(fc(x)) + 1", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = mixed(caption, 19).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("MODEL", "同一個模型", "同一個模型站在岔路口：往上是 JIT 常駐，往下是 AOT 出貨")
        model = chip("model", edge=ACCENT).move_to([-5.6, 0.2, 0])
        up_lab = mixed("torch.compile · JIT", 16, color=MUTED).move_to([-5.3, 1.9, 0])
        dn_lab = mixed("export + AOTInductor · AOT", 16, color=MUTED).move_to([-4.4, -1.5, 0])
        a_up = arr(model.get_top(), [-5.6, 1.55, 0], color=DIM)
        a_dn = arr(model.get_bottom(), [-5.6, -1.15, 0], color=DIM)
        self.play(FadeIn(model, scale=0.9), run_time=0.4)
        self.play(GrowArrow(a_up), GrowArrow(a_dn), FadeIn(up_lab), FadeIn(dn_lab), run_time=0.5)
        self.wait(3.5)

        switch("JIT", "邊跑邊編", "上軌活在 Python runtime 裡：eval hook 常駐，每次呼叫都經過 guard 驗票")
        panel = RoundedRectangle(corner_radius=0.16, width=10.3, height=2.15, stroke_color=EDGE, stroke_width=1.6, fill_color=CARD, fill_opacity=1).move_to([1.55, 1.8, 0])
        plab = mixed("PYTHON RUNTIME", 16, color=MUTED).move_to(panel.get_corner(UL) + RIGHT * 0.3 + DOWN * 0.22, aligned_edge=UL)
        stages = VGroup(chip("eval hook"), chip("Dynamo"), chip("Inductor"), chip("kernel", edge=ACCENT)).arrange(RIGHT, buff=0.85)
        stages.move_to(panel).shift(DOWN * 0.02)
        sarrs = VGroup(*[arr(stages[i].get_right(), stages[i + 1].get_left(), color=DIM) for i in range(3)])
        gnote = mixed("guard · 每次呼叫都驗，換 shape 就重編", 16, color=MUTED).move_to(panel).shift(DOWN * 0.72)
        self.play(FadeOut(up_lab), FadeIn(panel), FadeIn(plab), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in stages], lag_ratio=0.2), *[GrowArrow(a) for a in sarrs], run_time=0.8)
        dot = Dot(radius=0.07, color=ACCENT).move_to(stages[0].get_left() + LEFT * 0.3)
        self.play(dot.animate.move_to(stages[3].get_right() + RIGHT * 0.3), run_time=0.9, rate_func=linear)
        self.play(FadeIn(gnote, shift=UP * 0.08), FadeOut(dot), run_time=0.4)
        self.wait(2.5)

        switch("EXPORT", "一次收成", "下軌先把全圖收進 ExportedProgram：不准 graph break，動態維度先用 Dim 講好")
        gate1 = chip("torch.export", edge=ACCENT, fill=CARD_DIM).move_to([-3.5, -1.7, 0])
        ep = card("ExportedProgram", [
            "linear -> relu -> add · ATen 全圖",
            "signature · PARAMETER + USER_INPUT",
            "range · s77 = VR[1, 1024]",
        ]).move_to([2.3, -1.7, 0])
        a1 = arr([-5.6, -1.7, 0], gate1.get_left())
        a2 = arr(gate1.get_right(), ep.get_left())
        self.play(FadeOut(dn_lab), FadeOut(a_dn), FadeOut(a_up), model.animate.move_to([-6.1, -1.7, 0]), run_time=0.5)
        self.play(GrowArrow(a1), FadeIn(gate1), run_time=0.4)
        self.play(GrowArrow(a2), FadeIn(ep, shift=RIGHT * 0.15), run_time=0.6)
        self.wait(3)

        switch("FORGE", "鑄成一顆 .pt2", "AOTInductor 接手：連 wrapper 都翻成 C++，kernel、權重一起封進 .pt2")
        self.play(FadeOut(model), FadeOut(gate1), FadeOut(a1), FadeOut(a2), ep.animate.move_to([-3.6, -1.7, 0]), run_time=0.5)
        gate2 = chip("AOTInductor", edge=ACCENT).move_to([0.85, -1.7, 0])
        pt2_lines = VGroup(
            mixed("wrapper.so · C++ wrapper", 16),
            mixed("kernel.cpp · 運算核心", 16),
            mixed("權重 · 全部內含", 16),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        pt2_tag = mixed("tiny.pt2", 17, color=ACCENT)
        pt2_inner = VGroup(pt2_tag, pt2_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        pt2_box = RoundedRectangle(corner_radius=0.16, width=4.1, height=pt2_inner.height + 0.55, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD_DIM, fill_opacity=1)
        pt2_inner.move_to(pt2_box).align_to(pt2_box.get_left() + RIGHT * 0.35, LEFT)
        pt2 = VGroup(pt2_box, pt2_inner).move_to([4.55, -1.7, 0])
        a3 = arr(ep.get_right(), gate2.get_left())
        a4 = arr(gate2.get_right(), pt2.get_left())
        self.play(GrowArrow(a3), FadeIn(gate2), run_time=0.4)
        self.play(GrowArrow(a4), FadeIn(pt2, scale=0.92), Flash(gate2.get_center(), color=ACCENT, line_length=0.16, flash_radius=0.7), run_time=0.8)
        self.wait(3)

        switch("SHIP", "Python 退場", "出貨時整個 Python 圖層淡出：現場只剩 .pt2，載入即滿速，沒有 trace 也沒有 guard")
        pygroup = VGroup(panel, plab, stages, sarrs, gnote)
        self.play(pygroup.animate.set_opacity(0.12), FadeOut(ep), FadeOut(gate2), FadeOut(a3), FadeOut(a4), run_time=0.7)
        rail = Line([-5.9, -0.15, 0], [6.3, -0.15, 0], stroke_color=EDGE, stroke_width=1.6)
        rlab = mixed("C++ RUNTIME", 16, color=MUTED).next_to(rail, UP, buff=0.14).align_to(rail, LEFT)
        self.play(FadeOut(pygroup), Create(rail), FadeIn(rlab), run_time=0.6)
        self.play(pt2.animate.move_to([0, 1.15, 0]), run_time=0.8)
        xin = chip("x · batch=512", color=MUTED).move_to([-4.9, 0.5, 0])
        yout = chip("out · (512, 8)", color=MUTED).move_to([4.9, 0.5, 0]).set_opacity(0)
        self.play(FadeIn(xin, shift=RIGHT * 0.2), run_time=0.4)
        self.play(xin.animate.move_to(pt2.get_left() + LEFT * 0.2).set_opacity(0), run_time=0.6)
        self.play(yout.animate.set_opacity(1), Indicate(pt2_box, scale_factor=1.03, color=ACCENT), run_time=0.6)
        proof = mixed("dynamo frames traced = 0", 17, color=ACCENT).move_to([0, -0.85, 0])
        self.play(FadeIn(proof, shift=UP * 0.1), run_time=0.4)
        self.wait(3)

        switch("RULE", "同一條產線，兩個出口", "Dynamo 抓圖、AOTAutograd 攤平、Inductor 生碼：JIT 交還 Python，AOT 出貨 .pt2")
        self.play(FadeOut(rail), FadeOut(rlab), FadeOut(xin), FadeOut(yout), FadeOut(proof), FadeOut(pt2), run_time=0.5)
        line = VGroup(chip("Dynamo"), chip("AOTAutograd"), chip("Inductor")).arrange(RIGHT, buff=0.95).move_to([-1.7, 0.2, 0])
        larrs = VGroup(*[arr(line[i].get_right(), line[i + 1].get_left(), color=DIM) for i in range(2)])
        out_jit = chip("Python wrapper · JIT", color=MUTED).move_to([4.6, 1.15, 0])
        out_aot = chip(".pt2 · AOT", edge=ACCENT).move_to([4.6, -0.75, 0])
        b1 = arr(line[2].get_right(), out_jit.get_left(), color=DIM)
        b2 = arr(line[2].get_right(), out_aot.get_left())
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.15) for m in (line[0], larrs[0], line[1], larrs[1], line[2])], lag_ratio=0.15), run_time=1.0)
        self.play(GrowArrow(b1), FadeIn(out_jit), GrowArrow(b2), FadeIn(out_aot), run_time=0.7)
        self.play(Indicate(out_aot, scale_factor=1.06, color=ACCENT), run_time=0.5)
        self.wait(5.5)
