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
EDGE_DIM = "#262a30"
TXT = "#e8e6e3"
MUTED = "#8b8f96"
ACCENT = "#e8622a"
ACTIVE_FILL = "#2b2622"
config.background_color = BG
MONO = "Menlo"
SANS = "TASA Orbiter"
CJK = "PingFang TC"


def T(txt, font_size, **kw):
    return Text(txt, font_size=font_size * 4, **kw).scale(0.25)


def label(s, size=15, color=MUTED):
    return T(s, font=MONO, font_size=size, color=color)


def panel(w, h, fill=CARD, edge=EDGE):
    return RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=edge, stroke_width=1.5, fill_color=fill, fill_opacity=1)


def titled(w, h, name, sub, fill=CARD):
    r = panel(w, h, fill)
    t = T(name, font=SANS, font_size=22, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=15, color=MUTED)
    hdr = VGroup(t, s).arrange(RIGHT, buff=0.25, aligned_edge=DOWN).next_to(r.get_corner(UL), DR, buff=0.25).align_to(r.get_corner(UL) + RIGHT * 0.25, LEFT)
    return VGroup(r, hdr)


def code_lines(lines, size=14, color=TXT):
    rows = VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
    return rows


def pill(name, zh):
    t = VGroup(Dot(radius=0.06, color=ACCENT), T(name, font=SANS, font_size=18, weight=BOLD, color=BG), T("·", font=MONO, font_size=16, color="#666"), T(zh, font=CJK, font_size=16, color=BG)).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.08, color=color, stroke_width=w, max_tip_length_to_length_ratio=0.18, max_stroke_width_to_length_ratio=8)


class EvalFrame(Scene):
    def construct(self):
        title = T("torch.compile(f)(x)", font=MONO, font_size=26, color=TXT).to_corner(UL, buff=0.5)
        title[0:13].set_color(ACCENT)
        self.play(FadeIn(title), run_time=0.4)

        cur_pill = None
        cur_cap = None

        def switch(name, zh, caption):
            nonlocal cur_pill, cur_cap
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=20, color=TXT).to_edge(DOWN, buff=0.55)
            if cur_pill is not None:
                self.play(FadeOut(cur_pill), FadeOut(cur_cap), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur_pill, cur_cap = p, c

        bc = ["LOAD_DEREF   torch", "LOAD_ATTR    sin", "LOAD_FAST    x", "CALL         1", "LOAD_CONST   1", "BINARY_OP    +", "RETURN_VALUE"]
        frame = titled(3.9, 3.2, "FRAME", "f 的 frame")
        frame.move_to(LEFT * 4.9 + DOWN * 0.15)
        rows = code_lines(bc).next_to(frame[1], DOWN, buff=0.3).align_to(frame[1], LEFT)
        frame_lbl = label("CODE OBJECT  ·  bytecode").next_to(frame, UP, buff=0.25).align_to(frame, LEFT)

        sw = titled(2.6, 1.15, "eval_frame", "函式指標")
        sw.move_to(LEFT * 0.9 + DOWN * 0.15)
        sw_lbl = label("INTERPRETER STATE").next_to(sw, UP, buff=0.25).align_to(sw, LEFT)

        default = titled(5.2, 1.35, "_PyEval_EvalFrameDefault", "預設")
        default.move_to(RIGHT * 3.6 + UP * 1.5)
        default_body = code_lines(["for op in bytecode: run(op)"], size=13, color=MUTED).next_to(default[1], DOWN, buff=0.22).align_to(default[1], LEFT)
        default_lbl = label("PATH A  ·  CPython 照舊").next_to(default, UP, buff=0.25).align_to(default, LEFT)

        dyn = titled(5.2, 3.35, "Dynamo eval", "接手")
        dyn.move_to(RIGHT * 3.6 + DOWN * 1.3)
        dyn_lbl = label("PATH B  ·  torch.compile 換上的").next_to(dyn, UP, buff=0.25).align_to(dyn, LEFT)

        self.play(FadeIn(frame_lbl), FadeIn(frame), FadeIn(rows), run_time=0.5)
        self.play(FadeIn(sw_lbl), FadeIn(sw), run_time=0.4)
        a1 = arrow(frame[0].get_right(), sw[0].get_left())
        self.play(GrowArrow(a1), run_time=0.3)

        # Path A
        switch("CPYTHON", "預設路徑", "沒有 torch.compile 時：frame 交給 _PyEval_EvalFrameDefault，一條一條 bytecode 執行")
        self.play(FadeIn(default_lbl), FadeIn(default), FadeIn(default_body), run_time=0.4)
        a2 = arrow(sw[0].get_right(), default[0].get_left())
        self.play(GrowArrow(a2), run_time=0.3)
        for r in rows:
            self.play(r.animate.set_color(ACCENT), run_time=0.12)
            self.play(r.animate.set_color(TXT), run_time=0.12)
        self.wait(0.8)

        # Path B
        switch("DYNAMO", "接手", "torch.compile 把 eval_frame 指標換掉：同一個 frame 改送進 Dynamo")
        self.play(FadeIn(dyn_lbl), FadeIn(dyn), sw[0].animate.set_stroke(ACCENT, width=2.5).set_fill(ACTIVE_FILL), a2.animate.set_color(EDGE_DIM), run_time=0.4)
        a3 = arrow(sw[0].get_right(), dyn[0].get_left(), color=ACCENT, w=2.5)
        self.play(GrowArrow(a3), run_time=0.3)

        # inside dynamo: three mini cards
        def mini(name, lines, w=1.5):
            body = code_lines(lines, size=11)
            r = panel(max(w, body.width + 0.35), body.height + 0.55, fill=CARD_DIM, edge=EDGE)
            hd = T(name, font=MONO, font_size=11, color=MUTED)
            g = VGroup(r, VGroup(hd, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT))
            return g

        m1 = mini("FX GRAPH", ["sin = torch.sin(x)", "add = sin + 1", "return (add,)"], w=2.3)
        m2 = mini("GUARDS", ["x: f32[8] cuda", "torch.sin is same obj", "grad_mode == False"], w=2.3)
        m3 = mini("MODIFIED BYTECODE", ["LOAD_GLOBAL __compiled_fn_1   LOAD_FAST x", "CALL 1   RETURN_VALUE"], w=4.75)
        top = VGroup(m1, m2).arrange(RIGHT, buff=0.15, aligned_edge=UP)
        top.next_to(dyn[1], DOWN, buff=0.22).align_to(dyn[1], LEFT)
        m3.next_to(top, DOWN, buff=0.15).align_to(top, LEFT)
        minis = VGroup(m1, m2, m3)
        caps = ["符號執行 bytecode，記成 FX Graph", "同時記下前提：Guards", "改寫 bytecode：整段運算換成呼叫編好的函式"]
        for m, c in zip(minis, caps):
            switch("DYNAMO", "接手", c)
            self.play(FadeIn(m, shift=UP * 0.1), run_time=0.4)
            self.wait(0.9)

        # back to CPython
        switch("CPYTHON", "執行改寫後的 bytecode", "改寫後的 code object 交回 CPython 執行，之後 Guard 通過就直接重用")
        a4 = arrow(dyn[0].get_left() + DOWN * 1.3, frame[0].get_right() + DOWN * 1.2, color=ACCENT, w=2.5)
        self.play(GrowArrow(a4), run_time=0.5)
        new_rows = code_lines(["LOAD_GLOBAL  __compiled_fn_1", "LOAD_FAST    x", "CALL         1", "RETURN_VALUE"], color=ACCENT).move_to(rows, aligned_edge=UL)
        self.play(FadeOut(rows), FadeIn(new_rows), run_time=0.5)
        self.wait(3)
