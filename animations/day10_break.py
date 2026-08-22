from pathlib import Path

import manimpango
from manim import *

FONT_DIR = Path(__file__).parent / "fonts"
for f in FONT_DIR.glob("*.ttf"):
    manimpango.register_font(str(f))

BG = "#161719"
CARD = "#23272e"
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
    body = T(f"{name}  ·  {zh}", font_size=17, font=SANS, color=BG, t2f={name: SANS, "·": MONO, zh: zh_font}, t2w={name: BOLD}, t2c={"·": "#666"})
    t = VGroup(Dot(radius=0.06, color=ACCENT), body).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def line(s, color=TXT, size=14):
    return T(s, font=MONO, font_size=size, color=color)


def capsule(name, fn, codes):
    hdr = VGroup(T(name, font=SANS, font_size=15, weight=BOLD, color=TXT), T(fn, font=MONO, font_size=13, color=ACCENT)).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
    body = VGroup(hdr, *[T(c, font=MONO, font_size=13, color=TXT) for c in codes]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    box = RoundedRectangle(corner_radius=0.14, width=body.width + 0.6, height=body.height + 0.5, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD, fill_opacity=1)
    return VGroup(box, body.move_to(box))


ORIG = [
    (" 2", "LOAD_FAST     x"),
    (" 4", "LOAD_CONST    2"),
    (" 6", "BINARY_OP     *"),
    ("10", "STORE_FAST    x"),
    ("12", "LOAD_GLOBAL   print"),
    ("22", "LOAD_CONST    'mid'"),
    ("24", "CALL          1"),
    ("32", "POP_TOP"),
    ("34", "LOAD_FAST     x"),
    ("36", "LOAD_CONST    1"),
    ("38", "BINARY_OP     +"),
    ("42", "RETURN_VALUE"),
]

LX = -4.5
Y0, DY = 2.55, 0.46


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

        offs = VGroup(*[T(o, font=MONO, font_size=12, color=MUTED).move_to([LX - 0.3, Y0 - i * DY, 0], aligned_edge=RIGHT) for i, (o, _) in enumerate(ORIG)])
        ins = VGroup(*[line(s).move_to([LX, Y0 - i * DY, 0], aligned_edge=LEFT) for i, (_, s) in enumerate(ORIG)])

        switch("SETUP", "原始 bytecode", "f 的 bytecode 一字排開：前四條算 x * 2，中間三條呼叫 print，offset 32 之後算 x + 1")
        self.play(LaggedStart(*[AnimationGroup(FadeIn(o, shift=RIGHT * 0.2), FadeIn(l, shift=RIGHT * 0.2)) for o, l in zip(offs, ins)], lag_ratio=0.08), run_time=1.3)
        self.wait(2.5)

        switch("TRACE", "逐條往下翻", "InstructionTranslator 帶著游標逐條往下翻，Tensor 運算沿路收成圖的節點")
        cursor = Triangle(fill_color=ACCENT, fill_opacity=1, stroke_width=0).scale(0.09).rotate(-90 * DEGREES).move_to([LX - 1.05, Y0, 0])
        self.play(ins.animate.set_color(DIM), offs.animate.set_color(DIM), FadeIn(cursor), run_time=0.4)
        for i in range(6):
            anims = [cursor.animate.match_y(offs[i]), ins[i].animate.set_color(ACCENT), offs[i].animate.set_color(MUTED)]
            if i:
                anims.append(ins[i - 1].animate.set_color(TXT))
            self.play(*anims, run_time=0.3)
        self.wait(2.5)

        switch("BREAK", "handler 舉手", "游標走到 CALL：print 是有 side effect 的 builtin，符號值走不下去，unimplemented_v2 丟出 Unsupported")
        self.play(cursor.animate.match_y(offs[6]), ins[5].animate.set_color(TXT), ins[6].animate.set_color(ACCENT), offs[6].animate.set_color(ACCENT), run_time=0.4)
        raise_t = T("raise Unsupported", font=MONO, font_size=14, color=ACCENT, weight=BOLD).next_to(ins[6], RIGHT, buff=0.6)
        self.play(FadeIn(raise_t, shift=LEFT * 0.15), Flash(cursor, color=ACCENT, line_length=0.12, flash_radius=0.25), run_time=0.5)
        self.wait(3.5)

        switch("GRAPH 1", "前半收成圖", "斷點之前收到的節點照常結帳收圖，編成 __compiled_fn_2 交給 Inductor")
        cap1 = capsule("GRAPH 1", "__compiled_fn_2", ["x = l_x_ * 2"]).move_to([LX, Y0 - 1.5 * DY, 0], aligned_edge=LEFT)
        self.play(FadeOut(cursor), *[m.animate.set_color(ACCENT) for m in ins[0:4]], run_time=0.3)
        g1 = VGroup(*offs[0:4], *ins[0:4])
        self.play(ReplacementTransform(g1, cap1), run_time=0.9)
        self.remove(*offs[0:4], *ins[0:4])
        self.add(cap1)
        self.wait(3.5)

        switch("EAGER", "print 掉回 eager", "斷點那三條指令在改寫後的 bytecode 原樣保留，掉出圖的世界讓 CPython 自己跑")
        self.play(ins[6].animate.set_color(TXT), offs[6].animate.set_color(MUTED), FadeOut(raise_t), run_time=0.3)
        self.play(VGroup(*offs[4:7], *ins[4:7]).animate.shift(LEFT * 0.45), run_time=0.5)
        tag = T("-> eager", font=MONO, font_size=12, color=ACCENT).next_to(ins[6], RIGHT, buff=0.5)
        self.play(FadeIn(tag, shift=LEFT * 0.1), run_time=0.3)
        self.wait(2.5)

        switch("RESUME", "剩下包成續集", "斷點之後的指令從 offset 32 起包成 __resume_at_32_3，開場一條 JUMP_FORWARD 跳進函式中間")
        rest = VGroup(*offs[7:], *ins[7:])
        frame = RoundedRectangle(corner_radius=0.14, width=rest.width + 0.5, height=rest.height + 0.4, stroke_color=ACCENT, stroke_width=1.5, fill_opacity=0).move_to(rest)
        self.play(Create(frame), *[m.animate.set_color(TXT) for m in ins[7:]], run_time=0.7)
        capr = capsule("RESUME", "__resume_at_32_3", ["___stack0 -> JUMP_FORWARD", "return x + 1"]).move_to([LX, Y0 - 9 * DY, 0], aligned_edge=LEFT)
        wrap = VGroup(frame, *offs[7:], *ins[7:])
        self.play(ReplacementTransform(wrap, capr), run_time=0.9)
        self.remove(frame, *offs[7:], *ins[7:])
        self.add(capr)
        self.wait(3.5)

        switch("HOOK", "續集又被攔下", "resume fn 也是函式：eval hook 照樣攔截它，斷點之後編成第二張圖 __compiled_fn_5")
        cap2 = capsule("GRAPH 2", "__compiled_fn_5", ["add = l_x_ + 1"]).move_to([capr.get_right()[0] + 1.9, Y0 - 9 * DY, 0], aligned_edge=LEFT)
        a3 = Arrow(capr.get_right(), cap2.get_left(), buff=0.12, color=ACCENT, stroke_width=2, tip_length=0.16)
        hook = T("eval hook", font=MONO, font_size=12, color=ACCENT).next_to(a3, UP, buff=0.15)
        self.play(GrowArrow(a3), FadeIn(hook, shift=UP * 0.08), run_time=0.5)
        self.play(TransformFromCopy(capr, cap2), run_time=0.9)
        self.wait(3.5)

        switch("RUN", "縫回一條路", "執行順序把三段縫回一條路：圖一、掉到 eager 跑 print、呼叫 resume、接上圖二")
        emid = VGroup(*offs[4:7], *ins[4:7])
        a1 = Arrow(cap1.get_bottom(), emid.get_top(), buff=0.14, color=MUTED, stroke_width=2, tip_length=0.14)
        a2 = Arrow(emid.get_bottom(), capr.get_top(), buff=0.14, color=MUTED, stroke_width=2, tip_length=0.14)
        self.play(GrowArrow(a1), run_time=0.3)
        self.play(GrowArrow(a2), run_time=0.3)
        ball = Dot(radius=0.09, color=ACCENT).move_to(cap1.get_center())
        self.play(FadeIn(ball), run_time=0.2)
        self.play(ball.animate.move_to(emid.get_center()), run_time=0.55)
        self.play(ball.animate.move_to(capr.get_center()), run_time=0.55)
        self.play(ball.animate.move_to(cap2.get_center()), run_time=0.55)
        self.play(Flash(ball, color=ACCENT, line_length=0.15), FadeOut(ball), run_time=0.4)
        self.wait(3.5)

        switch("RULE", "先數 break", "一次 break 兩張圖夾一段 eager：fusion 消失、resume fn 多編一次，效能調校第一課是先數 break")
        self.wait(5.5)
