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


def panel(w, h, fill=CARD, edge=EDGE, r=0.12, sw=1.5):
    return RoundedRectangle(corner_radius=r, width=w, height=h, stroke_color=edge, stroke_width=sw, fill_color=fill, fill_opacity=1)


def header(name, sub):
    t = T(name, font=SANS, font_size=20, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=14, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub, edge=EDGE, sw=1.5, fill=CARD):
    r = panel(w, h, edge=edge, sw=sw, fill=fill)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
    return VGroup(r, hdr)


def rows(lines, size=12, color=TXT, buff=0.1):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def mini(title, lines, w, active=False):
    body = rows(lines, size=11, buff=0.08)
    r = panel(w, body.height + 0.62, fill=ACTIVE_FILL if active else CARD_DIM, edge=ACCENT if active else EDGE, r=0.08, sw=2 if active else 1.5)
    hd = T(title, font=MONO, font_size=11, color=ACCENT if active else MUTED)
    inner = VGroup(hd, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, inner)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


class OG(Scene):
    def construct(self):
        title = T("OutputGraph", font=MONO, font_size=26, color=TXT).to_corner(UL, buff=0.5)
        title[0:6].set_color(ACCENT)
        self.add(title)
        self.add(label("一個 frame 的一次編譯只有一個，所有產出都寫在它上面", size=14).next_to(title, DOWN, buff=0.15).align_to(title, LEFT))

        W = 3.4
        # left: producers
        lx = -4.9
        p1 = mini("InstructionTranslator", ["每翻一條指令 ->"], W)
        p2 = mini("VariableBuilder", ["每包一個外來值 ->"], W)
        p3 = mini("mutation handlers", ["每記一筆修改 ->"], W)
        prods = VGroup(p1, p2, p3).arrange(DOWN, buff=0.55).move_to([lx, -0.35, 0])
        self.add(label("PRODUCERS  ·  逐條累積").next_to(prods, UP, buff=0.25).align_to(prods, LEFT))
        self.add(prods)

        # center: OutputGraph warehouse
        cw, ch = 4.5, 4.9
        og = titled(cw, ch, "OutputGraph", "倉庫", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([0.1, -0.35, 0])
        m1 = mini("graph  (SubgraphTracer)", ["fx.Graph: matmul, add, relu"], cw - 0.5, active=False)
        m2 = mini("graphargs", ["L['x'], L['y'], L['bias']", "用到才登記，unused 不進來"], cw - 0.5)
        m3 = mini("guards", ["TENSOR_MATCH x3 ..."], cw - 0.5)
        m4 = mini("side_effects", ["(Day 7 的帳本)"], cw - 0.5)
        inner = VGroup(m1, m2, m3, m4).arrange(DOWN, buff=0.14).next_to(og[1], DOWN, buff=0.25).align_to(og[1], LEFT)
        self.add(label("STATE  ·  正在長大").next_to(og, UP, buff=0.25).align_to(og, LEFT))
        self.add(og, inner)

        for p, m in [(p1, m1), (p2, m2), (p3, m4)]:
            self.add(arrow(p[0].get_right(), [og[0].get_left()[0], p[0].get_right()[1], 0], color=MUTED))

        # right: compile_subgraph outputs
        rx = 4.95
        rW = 3.6
        q1 = mini("call_user_compiler", ["GraphModule -> backend", "拿回 compiled_fn"], rW, active=True)
        q2 = mini("install_global", ["globals['__compiled_fn_1']"], rW, active=True)
        q3 = mini("guards -> CheckFunctionManager", ["編成 C++ GuardManager 樹"], rW)
        outs = VGroup(q1, q2, q3).arrange(DOWN, buff=0.45).move_to([rx, -0.35, 0])
        self.add(label("COMPILE_SUBGRAPH  ·  收圖時").next_to(outs, UP, buff=0.25).align_to(outs, LEFT))
        self.add(outs)
        for q in (q1, q2, q3):
            self.add(arrow([og[0].get_right()[0], q[0].get_left()[1], 0], q[0].get_left(), color=ACCENT))

        self.add(T("翻譯期一路累積，RETURN 或 Graph Break 的瞬間 compile_subgraph 收攏一切：算活性、結帳、清輸入、交後端、塞 globals。", font=CJK, font_size=14, color=MUTED).to_edge(DOWN, buff=0.25))
