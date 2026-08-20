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


def pill(name, zh):
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    body = T(f"{name}  ·  {zh}", font_size=17, font=SANS, color=BG, t2f={name: SANS, "·": MONO, zh: zh_font}, t2w={name: BOLD}, t2c={"·": "#666"})
    t = VGroup(Dot(radius=0.06, color=ACCENT), body).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def rows(lines, size=12, color=TXT, buff=0.12):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


class Break(Scene):
    def construct(self):
        title = T("def f(x):  x = x * 2;  print('mid');  return x + 1", font=MONO, font_size=20, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        # left: source with three lines
        W = 4.07
        TOP, BOT = 2.45, -2.85
        H = TOP - BOT
        src_card = titled(W, H, "f", "翻譯進行中").move_to([-4.575, (TOP + BOT) / 2, 0])
        lbl0 = label("TRACE  ·  bytecode").next_to(src_card, UP, buff=0.22).align_to(src_card, LEFT)
        src = rows(["x = x * 2", "print('mid')", "return x + 1"], size=14, buff=0.55).next_to(src_card[1], DOWN, buff=0.5).align_to(src_card[1], LEFT)
        switch("TRACE", "翻譯到 print", "InstructionTranslator 逐條翻譯，走到 print：builtin、有 side effect，符號值走不下去")
        self.play(FadeIn(lbl0), FadeIn(src_card), FadeIn(src), run_time=0.5)
        self.play(src[0].animate.set_color(MUTED), run_time=0.3)
        self.play(src[1].animate.set_color(ACCENT), run_time=0.3)
        gb = T("Unsupported!", font=MONO, font_size=13, color=ACCENT, weight=BOLD).next_to(src[1], DOWN, buff=0.18).align_to(src[1], LEFT)
        self.play(FadeIn(gb, shift=UP * 0.1), run_time=0.3)
        self.wait(1.5)

        # right: three segments
        RX, RW = 2.3, 8.6
        seg_h = 1.55
        ys = [TOP - seg_h / 2, TOP - seg_h * 1.5 - 0.25, TOP - seg_h * 2.5 - 0.5]
        lbl1 = label("RESULT  ·  兩張圖夾一段 eager").move_to([RX - RW / 2, TOP + 0.32, 0], aligned_edge=LEFT)

        s1 = titled(RW, seg_h, "GRAPH 1", "__compiled_fn_2", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([RX, ys[0], 0])
        s1b = rows(["x = L_x_ * 2   ->  Inductor kernel"], size=12).next_to(s1[1], DOWN, buff=0.18).align_to(s1[1], LEFT)
        switch("STEP 1", "收前半段", "斷點之前的節點照常收圖、結帳、編譯成 __compiled_fn_2")
        self.play(FadeIn(lbl1), FadeIn(s1), FadeIn(s1b), src[0].animate.set_color(ACCENT), run_time=0.5)
        a1 = arrow(src_card[0].get_right() + UP * (ys[0] - (TOP + BOT) / 2) * 0 + UP * 1.6, s1[0].get_left(), color=ACCENT)
        self.play(GrowArrow(a1), run_time=0.3)
        self.wait(1.5)
        self.play(src[0].animate.set_color(MUTED), run_time=0.2)

        s2 = titled(RW, seg_h, "EAGER", "斷點指令原樣保留", edge=EDGE, sw=1.5).move_to([RX, ys[1], 0])
        s2b = rows(["LOAD print ('mid') / CALL   ->  CPython 自己跑"], size=12).next_to(s2[1], DOWN, buff=0.18).align_to(s2[1], LEFT)
        switch("STEP 2", "斷點回 eager", "print 那條指令留在改寫後的 bytecode 裡，CPython 照常執行")
        a2 = arrow(src_card[0].get_right() + UP * 0.0, s2[0].get_left(), color=MUTED)
        self.play(FadeIn(s2), FadeIn(s2b), GrowArrow(a2), run_time=0.5)
        self.wait(1.5)

        s3 = titled(RW, seg_h, "RESUME", "__resume_at_32_3", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([RX, ys[2], 0])
        s3b = rows(["JUMP 進函式中間, return x + 1  ->  被 eval hook 再攔截", "編成 GRAPH 2:  add = L_x_ + 1"], size=12, buff=0.1).next_to(s3[1], DOWN, buff=0.18).align_to(s3[1], LEFT)
        switch("STEP 3", "剩下包成 resume fn", "斷點之後的 bytecode 包成新函式，一被呼叫又被攔截，編成第二張圖")
        self.play(src[2].animate.set_color(ACCENT), run_time=0.25)
        a3 = arrow(src_card[0].get_right() + DOWN * 1.6, s3[0].get_left(), color=ACCENT)
        self.play(FadeIn(s3), FadeIn(s3b), GrowArrow(a3), run_time=0.5)
        self.wait(1.5)
        self.play(src[2].animate.set_color(MUTED), run_time=0.2)

        switch("COST", "break 為什麼貴", "圖被切小 fusion 沒了、eager 段本身慢、帳本提前結算、resume fn 多編一次")
        for s in (s1, s2, s3):
            self.play(s[0].animate.set_stroke(ACCENT, width=2.5), run_time=0.15)
            self.play(s[0].animate.set_stroke(EDGE if s is s2 else ACCENT, width=1.5 if s is s2 else 2), run_time=0.15)
        self.wait(3.5)
