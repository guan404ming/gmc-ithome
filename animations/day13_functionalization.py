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
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    runs = re.findall(r"[一-鿿，、。]+|[^一-鿿，、。 ]+", zh)
    zs = [T(r, font=CJK if re.search(r"[一-鿿]", r) else MONO, font_size=17, color=BG) for r in runs]
    zt = VGroup(*zs).arrange(RIGHT, buff=0.1)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def chip(main, sub, w, main_color=TXT, edge=EDGE, fill=CARD_DIM):
    m = T(main, font=MONO, font_size=12, color=main_color)
    s = T(sub, font=MONO, font_size=10, color=MUTED)
    body = VGroup(m, s).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    r = panel(w, body.height + 0.3, fill=fill, edge=edge, r=0.08)
    body.move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, body)


def trunk_link(points, color):
    v = VMobject(stroke_color=color, stroke_width=2)
    v.set_points_as_corners(points)
    return v


class Functionalization(Scene):
    def construct(self):
        title = T("functionalize(f)", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
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
        CY = (TOP + BOT) / 2
        WL, WM, WR = 4.0, 3.3, 5.3
        GAP = (13.22 - WL - WM - WR) / 2
        XL = -6.61 + WL / 2
        XM = -6.61 + WL + GAP + WM / 2
        XR = 6.61 - WR / 2
        src_card = titled(WL, H, "USER CODE", "graph_code").move_to([XL, CY, 0])
        ledger_card = titled(WM, H, "LEDGER", "FunctionalTensor").move_to([XM, CY, 0])
        aot_card = titled(WR, H, "AOT GRAPH", "functionalized").move_to([XR, CY, 0])
        lbls = [label("BEFORE  ·  in-place 還在").next_to(src_card, UP, buff=0.22).align_to(src_card, LEFT),
                label("REWRITE  ·  指標記帳").next_to(ledger_card, UP, buff=0.22).align_to(ledger_card, LEFT),
                label("AFTER  ·  ATen 層，純函數").next_to(aot_card, UP, buff=0.22).align_to(aot_card, LEFT)]

        src_lines = ["y = x.view(2, 8)", "y.add_(1)", "y.relu_()", "return x * 2"]
        src = VGroup(*[T(l, font=MONO, font_size=13, color=TXT) for l in src_lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        src.next_to(src_card[1], DOWN, buff=0.45).align_to(src_card[1], LEFT)

        card_l = ledger_card[0].get_left()[0]
        cor_x = card_l + 0.4
        chip_x0 = card_l + 0.65
        cw = WM - 0.9
        chip_x = chip("x  ·  arg0_1", "f32[4,4]  呼叫者持有", cw)
        chip_y = chip("y  ·  x.view(2, 8)", "alias  ·  同一塊 storage", cw)
        chip_v0 = chip("S0", "storage 目前的值", cw)
        chip_x.next_to(ledger_card[1], DOWN, buff=0.45)
        chip_x.shift(RIGHT * (chip_x0 - chip_x[0].get_left()[0]))
        chip_y.next_to(chip_x, DOWN, buff=0.4).align_to(chip_x, LEFT)
        chip_v0.next_to(chip_y, DOWN, buff=0.6).align_to(chip_x, LEFT)

        def link(color=MUTED):
            xm = chip_x[0].get_left() + LEFT * 0.0
            ym = chip_y[0].get_left()
            vm = chip_v0[0].get_left()
            pts = [xm, [cor_x, xm[1], 0], [cor_x, vm[1], 0], vm]
            l1 = trunk_link(pts, color)
            l2 = trunk_link([ym, [cor_x, ym[1], 0]], color)
            d = Dot([cor_x, ym[1], 0], radius=0.04, color=color)
            return VGroup(l1, l2, d)

        switch("SETUP", "先掛帳本", "每顆 Tensor 都被包進 FunctionalTensor：記著自己指向哪個最新值")
        self.play(FadeIn(src_card), FadeIn(ledger_card), FadeIn(aot_card), *[FadeIn(l) for l in lbls], run_time=0.5)
        self.play(FadeIn(src), FadeIn(chip_x), FadeIn(chip_v0), run_time=0.4)
        lk = trunk_link([chip_x[0].get_left(), [cor_x, chip_x[0].get_left()[1], 0], [cor_x, chip_v0[0].get_left()[1], 0], chip_v0[0].get_left()], MUTED)
        self.play(Create(lk), run_time=0.4)
        self.wait(2.5)

        gx = aot_card[1].get_corner(DL) + DOWN * 0.45
        glines = []

        def emit(txt, color=TXT, note=None, rt=0.35):
            t = T(txt, font=MONO, font_size=11, color=color)
            if glines:
                t.next_to(glines[-1], DOWN, buff=0.17).align_to(glines[0], LEFT)
            else:
                t.move_to(gx, aligned_edge=UL)
            glines.append(t)
            anims = [FadeIn(t, shift=RIGHT * 0.1)]
            if note:
                n = T(note, font=CJK, font_size=10, color=MUTED).next_to(t, RIGHT, buff=0.25)
                anims.append(FadeIn(n))
            self.play(*anims, run_time=rt)
            return t

        def strike(i, tag):
            line = Line(src[i].get_left() + LEFT * 0.05, src[i].get_right() + RIGHT * 0.05, color=ACCENT, stroke_width=2)
            tg = T(tag, font=MONO, font_size=10, color=ACCENT).next_to(src[i], RIGHT, buff=0.25)
            self.play(Create(line), FadeIn(tg), run_time=0.3)
            self.play(src[i].animate.set_color(DIM), line.animate.set_color(DIM), run_time=0.2)

        switch("STEP 1", "y = x.view(2, 8)", "view 不搬資料：帳本記下 y 是 x 的 alias，兩個名字接上同一條線")
        self.play(src[0].animate.set_color(ACCENT), run_time=0.25)
        self.play(FadeIn(chip_y, shift=DOWN * 0.1), run_time=0.35)
        lk2 = link(MUTED)
        self.play(FadeOut(lk), run_time=0.15)
        self.play(Create(lk2), run_time=0.4)
        emit("view = view(arg0_1, [2,8])")
        self.play(src[0].animate.set_color(TXT), run_time=0.2)
        self.wait(2.5)

        switch("STEP 2", "y.add_(1)", "就地改被攔下：換成 out-of-place 的 add，view 重放把新值搬回 base")
        self.play(src[1].animate.set_color(ACCENT), run_time=0.25)
        emit("add = add(view, 1)", color=ACCENT)
        emit("view_1 = view(add, [4,4])", color=MUTED, note="寫回 base")
        emit("view_2 = view(view_1, [2,8])", color=MUTED, note="重新長出 view")
        chip_v1 = chip("S1 = add(S0, 1)", "x -> view_1   y -> view_2", cw, main_color=ACCENT, edge=ACCENT, fill=ACTIVE_FILL).move_to(chip_v0).align_to(chip_v0, LEFT)
        self.play(lk2.animate.set_color(ACCENT), run_time=0.2)
        self.play(FadeOut(lk2, scale=0.95), run_time=0.25)
        self.play(Transform(chip_v0, chip_v1), run_time=0.4)
        lk3 = link(ACCENT)
        self.play(Create(lk3), run_time=0.4)
        strike(1, "-> add")
        self.wait(3.5)

        switch("STEP 3", "y.relu_()", "同一套規則再來一次：relu_ 變 relu，重放後 base 的最新值是 view_3")
        self.play(src[2].animate.set_color(ACCENT), run_time=0.25)
        emit("relu = relu(view_2)", color=ACCENT)
        emit("view_3 = view(relu, [4,4])", color=MUTED, note="寫回 base")
        chip_v2 = chip("S2 = relu(S1)", "base 最新值 = view_3", cw, main_color=ACCENT, edge=ACCENT, fill=ACTIVE_FILL).move_to(chip_v0).align_to(chip_v0, LEFT)
        self.play(Indicate(lk3, color=ACCENT, scale_factor=1.02), run_time=0.4)
        self.play(Transform(chip_v0, chip_v2), run_time=0.4)
        strike(2, "-> relu")
        self.wait(2.5)

        switch("STEP 4", "return x * 2", "讀 x 拿到的是帳本上的最新值：mul 吃 view_3，不是原始的 arg0_1")
        self.play(src[3].animate.set_color(ACCENT), run_time=0.25)
        emit("mul = mul(view_3, 2)")
        self.play(src[3].animate.set_color(TXT), run_time=0.2)
        self.wait(3.5)

        switch("EPILOGUE", "邊界結清", "呼叫者手上的 x 也要看到修改：圖尾端補一條 copy_，一次寫回輸入")
        cp = emit("copy_(arg0_1, view_3)", color=ACCENT)
        emit("return (mul,)")
        chip_x2 = chip("x  ·  arg0_1", "copy_ 寫回，呼叫者看得到", cw, main_color=TXT, edge=ACCENT, fill=ACTIVE_FILL).move_to(chip_x).align_to(chip_x, LEFT)
        self.play(Transform(chip_x, chip_x2), Indicate(cp, color=ACCENT, scale_factor=1.05), run_time=0.6)
        self.wait(3.5)

        switch("RULE", "圖內純函數，邊界一條 copy_", "add_ 變 add、view 用重放維持一致；後端從此可以自由重排融合")
        self.wait(5.5)
