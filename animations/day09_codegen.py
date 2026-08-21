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
    t = T(name, font=SANS, font_size=21, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=14, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub, fill=CARD, edge=EDGE, sw=1.5):
    r = panel(w, h, fill=fill, edge=edge, sw=sw)
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


ORIG = [
    "LOAD_FAST     x",
    "LOAD_FAST     n",
    "BINARY_OP     *",
    "LOAD_CONST    1",
    "BINARY_OP     +",
    "RETURN_VALUE",
]

NEW = [
    ("0", "LOAD_GLOBAL   __compiled_fn_1"),
    ("10", "LOAD_FAST     x"),
    ("12", "CALL          1"),
    ("20", "STORE_FAST    graph_out_0"),
    ("22", "LOAD_FAST     graph_out_0"),
    ("24", "LOAD_CONST    0"),
    ("26", "BINARY_SUBSCR"),
    ("30", "DELETE_FAST   graph_out_0"),
    ("32", "RETURN_VALUE"),
]


class Codegen(Scene):
    def construct(self):
        title = T("f(x, n)  ->  x * n + 1", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        TOP, BOT = 2.45, -2.9
        H = TOP - BOT
        MIDY = (TOP + BOT) / 2
        left = titled(3.9, H, "ORIGINAL", "原 bytecode").move_to([-4.76, MIDY, 0])
        right = titled(5.5, H, "MODIFIED", "新 bytecode").move_to([3.94, MIDY, 0])
        emitter = titled(3.0, 2.2, "PYCODEGEN", "codegen.py").move_to([-0.81, MIDY, 0])
        lbls = [label("SOURCE  ·  f.__code__").next_to(left, UP, buff=0.22).align_to(left, LEFT),
                label("OUTPUT  ·  to CPython").next_to(right, UP, buff=0.22).align_to(right, LEFT)]

        RX, GX = 2.06, 1.86
        Y0, DY = 1.35, 0.42
        LX = left[0].get_left()[0] + 0.25

        switch("SETUP", "要生一段等價改寫", "運算已經收進 __compiled_fn_1，PyCodegen 要生的只剩搬運：載入、擺輸入、呼叫、拆輸出、return")
        self.play(FadeIn(left), FadeIn(right), FadeIn(emitter), *[FadeIn(l) for l in lbls], run_time=0.5)
        olines = VGroup(*[T(l, font=MONO, font_size=13, color=TXT).move_to([LX, Y0 - i * DY, 0], aligned_edge=LEFT) for i, l in enumerate(ORIG)])
        self.play(FadeIn(olines, shift=RIGHT * 0.1), run_time=0.4)
        self.wait(2.5)

        drv = [None]

        def driver(lines_):
            g = rows(lines_, size=11, buff=0.2).next_to(emitter[1], DOWN, buff=0.3).align_to(emitter[1], LEFT)
            g[0].set_color(ACCENT)
            for r in g[1:]:
                r.set_color(MUTED)
            anims = [FadeIn(g, shift=UP * 0.05)]
            if drv[0] is not None:
                anims.append(FadeOut(drv[0]))
            self.play(*anims, run_time=0.25)
            drv[0] = g

        nrows = []

        def emit(i, rt=0.3):
            m = T(NEW[i][1], font=MONO, font_size=12, color=ACCENT).move_to([RX, Y0 - i * DY, 0], aligned_edge=LEFT)
            self.play(FadeIn(m, shift=RIGHT * 0.25), run_time=rt)
            nrows.append(m)
            return m

        def settle(ms, extra=()):
            self.play(*[m.animate.set_color(TXT) for m in ms], *extra, run_time=0.15)

        switch("STEP 1", "載入編譯產物", "__compiled_fn_1 是 Day 8 塞進 globals 的名字，一條 LOAD_GLOBAL 就載得到")
        driver(["install_global()", "__compiled_fn_1"])
        m0 = emit(0)
        self.wait(3.5)
        settle([m0])

        switch("STEP 2", "按 Source 擺輸入", "x 有 Source：L['x'].reconstruct() 生一條 LOAD_FAST；n 被 bake 成常數，不用載")
        driver(["source.reconstruct()", "L['x'] -> LOAD_FAST x"])
        self.play(olines[0].animate.set_color(ACCENT), run_time=0.2)
        m1 = emit(1)
        bake = T("-> bake", font=MONO, font_size=10, color=ACCENT).next_to(olines[1], RIGHT, buff=0.2)
        self.play(olines[1].animate.set_color(DIM), FadeIn(bake), run_time=0.3)
        self.wait(3.5)
        settle([m1], extra=[olines[0].animate.set_color(MUTED)])

        switch("STEP 3", "一條 CALL 吃掉計算", "乘與加全部發生在編譯產物裡，三條運算指令在新 bytecode 沒有對應物")
        driver(["create_call_function(1)"])
        m2 = emit(2)
        gtag = T("-> graph", font=MONO, font_size=10, color=ACCENT).next_to(olines[3], RIGHT, buff=0.2)
        self.play(*[olines[k].animate.set_color(DIM) for k in (2, 3, 4)], FadeIn(gtag), run_time=0.35)
        self.wait(3.5)
        settle([m2])

        switch("STEP 4", "拆輸出", "圖的輸出永遠是 tuple：暫存進 graph_out_0、取下標 0、用完 DELETE_FAST 歸還引用")
        driver(["graph_out_0 = new_var()", "tuple -> [0]"])
        ms = [emit(i, rt=0.22) for i in range(3, 8)]
        self.wait(3.5)
        settle(ms)

        switch("STEP 5", "RETURN", "RETURN_VALUE 收尾；如果有 side effect，Day 7 的重播碼會插在它前面")
        driver(["RETURN_VALUE"])
        m8 = emit(8)
        self.play(olines[5].animate.set_color(MUTED), run_time=0.2)
        self.wait(2.5)
        settle([m8])

        switch("ASSEMBLE", "組回 code object", "組裝時才算 offset：跳轉參照換算回數字、EXTENDED_ARG 掃描到收斂、stacksize 重算")
        new_hdr = header("ASSEMBLE", "組裝").move_to(emitter[0].get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
        self.play(FadeOut(emitter[1]), FadeIn(new_hdr), run_time=0.3)
        driver(["transform_code_object", "fix_extended_args()"])
        offs = VGroup(*[T(NEW[i][0], font=MONO, font_size=11, color=MUTED) for i in range(9)])
        for i, o in enumerate(offs):
            o.move_to([GX, Y0 - i * DY, 0], aligned_edge=RIGHT).match_y(nrows[i])
        self.play(LaggedStart(*[FadeIn(o, shift=RIGHT * 0.1) for o in offs], lag_ratio=0.12), run_time=1.2)
        self.play(right[0].animate.set_stroke(color=ACCENT, width=2).set_fill(ACTIVE_FILL), run_time=0.4)
        self.wait(3.5)

        switch("RULE", "只生搬運碼", "值有 Source 就從原位置載、是圖輸出就從 graph_out_0 取、常數直接 LOAD_CONST，能省的絕不多生")
        self.wait(5.5)
