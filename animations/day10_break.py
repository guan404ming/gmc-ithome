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


def rows(lines, size=12, color=TXT, buff=0.1):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


def card(w, h, name, sub, lines, edge=EDGE, sw=1.5, fill=CARD, size=12):
    t = titled(w, h, name, sub, edge=edge, sw=sw, fill=fill)
    body = rows(lines, size=size, buff=0.08).next_to(t[1], DOWN, buff=0.18).align_to(t[1], LEFT)
    return VGroup(t, body)


class Break(Scene):
    def construct(self):
        title = T("def f(x): x = x * 2; print('mid'); return x + 1", font=MONO, font_size=18, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        Y_SEG, Y_BOT = 1.24, -1.96
        lane_top = titled(12.9, 2.3, "COMPILED", "圖的世界，Inductor 接手", fill=CARD_DIM).move_to([0, 1.45, 0])
        lane_bot = titled(12.9, 2.3, "EAGER", "CPython 逐條執行", fill=CARD_DIM).move_to([0, -1.75, 0])

        codes = ["x = x * 2", "print('mid')", "return x + 1"]
        segs = []
        for code, x in zip(codes, [-4.1, 0.0, 4.1]):
            p = panel(3.9, 1.0, r=0.08)
            t = T(code, font=MONO, font_size=14, color=TXT).move_to(p)
            segs.append(VGroup(p, t).move_to([x, Y_SEG, 0]))
        j1 = Line([-2.15, Y_SEG, 0], [-1.95, Y_SEG, 0], color=EDGE, stroke_width=4)
        j2 = Line([1.95, Y_SEG, 0], [2.15, Y_SEG, 0], color=EDGE, stroke_width=4)

        switch("TRACE", "逐條往右翻", "InstructionTranslator 沿著 bytecode 時間軸往右走，Tensor 運算一路收進同一張圖")
        self.play(FadeIn(lane_top), FadeIn(lane_bot), *[FadeIn(s) for s in segs], FadeIn(j1), FadeIn(j2), run_time=0.5)
        dot = Dot(radius=0.07, color=ACCENT).move_to([-5.9, 0.52, 0])
        self.play(FadeIn(dot), run_time=0.2)
        self.play(dot.animate.move_to([-2.3, 0.52, 0]), segs[0][0].animate.set_fill(ACTIVE_FILL), run_time=1.2)
        self.wait(2.5)

        switch("BREAK", "handler 舉手", "走到 print：builtin、有 side effect，符號值走不下去，unimplemented_v2 丟出 Unsupported")
        self.play(dot.animate.move_to([-1.7, 0.52, 0]), run_time=0.4)
        self.play(segs[1][0].animate.set_stroke(ACCENT, width=2), j1.animate.set_color(ACCENT), j2.animate.set_color(ACCENT), run_time=0.3)
        gb = T("raise Unsupported", font=MONO, font_size=13, color=ACCENT, weight=BOLD).move_to([0, -0.15, 0])
        self.play(FadeIn(gb, shift=UP * 0.1), run_time=0.3)
        self.wait(3.5)

        g1 = card(3.9, 1.4, "GRAPH 1", "__compiled_fn_2", ["x = l_x_ * 2"], edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([-4.1, Y_SEG, 0])
        switch("GRAPH 1", "收前半段", "斷點之前的節點照常收圖、結帳，編成 __compiled_fn_2 交給 Inductor")
        self.play(FadeOut(j1), ReplacementTransform(segs[0], g1), run_time=0.5)
        self.wait(3.5)

        e = card(3.6, 1.4, "EAGER", "斷點指令原樣保留", ["LOAD print · CALL 1"]).move_to([0, Y_BOT, 0])
        switch("EAGER", "print 回 eager", "斷點那條指令在改寫後的 bytecode 裡原樣保留，掉回 eager 讓 CPython 自己跑")
        self.play(FadeOut(gb), FadeOut(dot), FadeOut(j2), ReplacementTransform(segs[1], e), run_time=0.6)
        self.wait(2.5)

        r = card(3.6, 1.4, "RESUME", "__resume_at_32_3", ["___stack0 · JUMP_FORWARD", "return x + 1"], edge=ACCENT, sw=2).move_to([4.1, Y_BOT, 0])
        switch("RESUME", "從中間接手", "斷點之後的 bytecode 包成 __resume_at_32_3，開場一條 JUMP_FORWARD 跳進函式中間")
        self.play(ReplacementTransform(segs[2], r), run_time=0.6)
        self.wait(3.5)

        g2 = card(3.9, 1.4, "GRAPH 2", "__compiled_fn_5", ["add = l_x_ + 1"], edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([4.1, Y_SEG, 0])
        switch("HOOK", "又被攔截", "resume fn 也是函式：eval hook 照樣攔下它，把斷點之後編成第二張圖")
        a3 = arrow(r[0][0].get_top(), g2[0][0].get_bottom(), color=ACCENT)
        hook = VGroup(panel(2.0, 0.5, fill=CARD, edge=ACCENT, r=0.25), T("eval hook", font=MONO, font_size=12, color=ACCENT))
        hook[1].move_to(hook[0])
        hook.move_to([4.1, -0.15, 0])
        self.play(GrowArrow(a3), run_time=0.3)
        self.play(FadeIn(hook), run_time=0.3)
        self.play(FadeIn(g2, shift=UP * 0.1), run_time=0.4)
        self.wait(3.5)

        switch("RUN", "縫回一條路", "執行順序：圖一、掉到 eager 跑 print、呼叫 resume fn、接上被攔截編好的圖二")
        a1 = arrow(g1[0][0].get_bottom(), e[0][0].get_top(), color=MUTED)
        a2 = arrow(e[0][0].get_right(), r[0][0].get_left(), color=MUTED)
        self.play(GrowArrow(a1), run_time=0.3)
        self.play(GrowArrow(a2), run_time=0.3)
        rd = Dot(radius=0.09, color=ACCENT).move_to(g1[0][0].get_center())
        self.play(FadeIn(rd), run_time=0.2)
        self.play(rd.animate.move_to(e[0][0].get_center()), run_time=0.5)
        self.play(rd.animate.move_to(r[0][0].get_center()), run_time=0.5)
        self.play(rd.animate.move_to(g2[0][0].get_center()), run_time=0.5)
        self.play(Flash(rd, color=ACCENT, line_length=0.15), run_time=0.4)
        self.wait(3.5)
        self.play(FadeOut(rd), run_time=0.2)

        switch("RULE", "兩張圖夾一段 eager", "一次 break：fusion 消失、中段 eager、resume fn 多編一次，效能調校第一課是先數 break")
        self.wait(5.5)
