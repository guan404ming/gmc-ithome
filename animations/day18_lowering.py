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


def label(s, size=16, color=MUTED):
    return T(s, font=MONO, font_size=size, color=color)


def panel(w, h, fill=CARD, edge=EDGE, r=0.12, sw=1.5):
    return RoundedRectangle(corner_radius=r, width=w, height=h, stroke_color=edge, stroke_width=sw, fill_color=fill, fill_opacity=1)


def header(name, sub):
    t = T(name, font=SANS, font_size=20, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=16, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub, fill=CARD, edge=EDGE, sw=1.5):
    r = panel(w, h, fill=fill, edge=edge, sw=sw)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.24, aligned_edge=UL)
    return VGroup(r, hdr)


def pill(name, zh):
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    zt = T(zh, font=zh_font, font_size=17, color=BG)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def rows(lines, size=16, color=TXT, buff=0.16):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.08, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


def node(txt, w, accent=False):
    r = panel(w, 0.56, fill=ACTIVE_FILL if accent else CARD_DIM, edge=ACCENT if accent else EDGE, r=0.08, sw=2 if accent else 1.5)
    t = T(txt, font=MONO, font_size=16, color=TXT).move_to(r)
    return VGroup(r, t)


def fn_card(name, sub, body_str, w=5.9):
    c = titled(w, 1.55, name, sub)
    b = T(body_str, font=MONO, font_size=17, color=TXT).move_to(c[0].get_bottom() + UP * 0.42)
    return VGroup(c, b)


NODE_Y = 2.0


class Lowering(Scene):
    def construct(self):
        title = T("f(x) = relu(x + 1).sum(dim=1)", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("SETUP", "ATen 圖", "拆解完的圖只剩 ATen node，GraphLowering 順著拓撲序一個一個 node 走訪")
        n_add = node("aten.add", 1.9).move_to([-3.4, NODE_Y, 0])
        n_relu = node("aten.relu", 2.0).move_to([-0.6, NODE_Y, 0])
        n_sum = node("aten.sum(dim=1)", 2.9).move_to([2.6, NODE_Y, 0])
        a1 = arrow(n_add[0].get_right(), n_relu[0].get_left())
        a2 = arrow(n_relu[0].get_right(), n_sum[0].get_left())
        glbl = label("FX Graph  ·  after decomposition").next_to(n_add, UP, buff=0.3).align_to(n_add, LEFT)
        self.play(FadeIn(glbl), FadeIn(n_add), GrowArrow(a1), FadeIn(n_relu), GrowArrow(a2), FadeIn(n_sum), run_time=0.9)
        self.wait(3.5)

        switch("LOOKUP", "查 lowering table", "lowering table 是一個 dict，每個 ATen op 對到一條 Python 函式，碰到 node 就查表")
        table = titled(6.4, 2.5, "lowerings", "dict · 1713 entries").move_to([-3.2, -1.75, 0])
        trows = rows(["aten.add   ->  make_pointwise", "aten.relu  ->  make_pointwise", "aten.sum   ->  make_reduction"], size=16).move_to(table[0].get_center() + DOWN * 0.28)
        self.play(FadeIn(table), FadeIn(trows, shift=UP * 0.1), run_time=0.6)
        look = arrow(n_add[0].get_bottom(), trows[0].get_top() + RIGHT * 0.4, color=ACCENT, w=2.5)
        self.play(GrowArrow(look), trows[0].animate.set_color(ACCENT), run_time=0.5)
        self.wait(3.5)

        switch("MORPH", "node 變成函式", "add 不再是節點，它變成一句「第 i 格怎麼算」，值還沒被算，buffer 也還不存在")
        pw_add = fn_card("Pointwise", "ranges=[4, 8]", "inner_fn = λ i: load(x, i) + 1").move_to([-3.5, 0.75, 0])
        self.play(FadeOut(look), trows[0].animate.set_color(DIM), run_time=0.3)
        self.play(ReplacementTransform(n_add, pw_add), FadeOut(a1), run_time=0.9)
        self.play(Flash(pw_add[1], color=ACCENT, line_length=0.15, flash_radius=0.5), run_time=0.5)
        self.wait(3.5)

        switch("INLINE", "疊進下游", "relu 的 inner_fn 直接呼叫上游的 inner_fn，兩條函式內聯成一條，中間值不落地")
        pw_relu = fn_card("Pointwise", "ranges=[4, 8]", "inner_fn = λ i: relu( ... )").move_to([3.1, 0.75, 0])
        self.play(trows[1].animate.set_color(ACCENT), run_time=0.3)
        self.play(ReplacementTransform(n_relu, pw_relu), FadeOut(a2), trows[1].animate.set_color(DIM), run_time=0.9)
        self.wait(1)
        fused = fn_card("Pointwise", "ranges=[4, 8]", "inner_fn = λ i: relu(load(x, i) + 1)", w=6.4).move_to([0.4, 0.75, 0])
        ghost = pw_add[1].copy().set_color(ACCENT)
        self.play(ghost.animate.move_to(pw_relu[1]).scale(0.85), run_time=0.8)
        self.play(ReplacementTransform(VGroup(pw_add, pw_relu, ghost), fused), run_time=0.9)
        self.play(Flash(fused[1], color=ACCENT, line_length=0.15, flash_radius=0.5), run_time=0.5)
        self.wait(3.5)

        switch("REDUCE", "Reduction 收編", "sum 走 make_reduction，多一組 reduction var r，pointwise 鏈整段被吸進它的 body")
        self.play(trows[2].animate.set_color(ACCENT), run_time=0.3)
        red = fn_card("Reduction", "size=[4] · reduction=[8] · sum", "inner_fn = λ i, r: relu(load(x, 8i+r) + 1)", w=7.2).move_to([0.4, -1.5, 0])
        self.play(FadeOut(table), FadeOut(trows), run_time=0.4)
        self.play(ReplacementTransform(n_sum, red), run_time=0.9)
        ghost2 = fused[1].copy().set_color(ACCENT)
        self.play(ghost2.animate.move_to(red[1]).scale(0.9), FadeOut(fused), run_time=0.8)
        self.remove(ghost2)
        self.play(Flash(red[1], color=ACCENT, line_length=0.15, flash_radius=0.5), run_time=0.5)
        self.wait(3.5)

        switch("REALIZE", "落地成 loop body", "被輸出用到才 realize 成 buffer，三個 node 收攏成一個 loop body，記憶體只寫一次")
        body = titled(6.6, 3.3, "op0_loop_body", "唯一的 kernel 雛形").move_to([0.4, 0.0, 0])
        blines = rows(["load = ops.load('x', 8*p0 + p1)", "add  = ops.add(load, 1.0)", "relu = ops.relu(add)", "red  = ops.reduction('sum', relu)", "ops.store_reduction('buf0', p0)"], size=16).move_to(body[0].get_center() + DOWN * 0.32)
        blines[4].set_color(ACCENT)
        self.play(FadeOut(glbl), ReplacementTransform(red, body), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.15) for l in blines], lag_ratio=0.15), run_time=1.2)
        self.wait(3.5)

        switch("RULE", "融合就是內聯", "define-by-run 的 IR 天生可組合：融合就是函式內聯。誰跟誰融，明天交給 Scheduler 決定")
        self.wait(5.5)
