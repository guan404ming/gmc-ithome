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


def shell(label, w=None):
    t = T(label, font=MONO, font_size=16, color=TXT)
    box = RoundedRectangle(corner_radius=0.22, width=w or t.width + 0.55, height=0.6, stroke_width=0, fill_color=CARD_DIM, fill_opacity=1)
    edge = DashedVMobject(RoundedRectangle(corner_radius=0.22, width=box.width, height=0.6), num_dashes=42).set_stroke(MUTED, 1.6)
    return VGroup(box, edge.move_to(box), t.move_to(box))


def gate(name, sub):
    n = T(name, font=MONO, font_size=18, color=ACCENT)
    s = T(sub, font=MONO, font_size=16, color=MUTED)
    inner = VGroup(n, s).arrange(DOWN, buff=0.16)
    box = RoundedRectangle(corner_radius=0.18, width=inner.width + 0.7, height=inner.height + 0.55, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD, fill_opacity=1)
    return VGroup(box, inner.move_to(box))


STAGE_Y = 0.35


class FakeT(Scene):
    def construct(self):
        title = T("f(x, w) -> tanh(x @ w).item()", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("REAL", "真值世界", "一顆真 Tensor：格子裡裝滿數值，但編譯器要的只有外圈的 metadata")
        vals = [
            " 0.42  -1.07   0.88   0.13",
            "-0.55   1.94  -0.23   0.71",
            " 0.09  -1.31   0.64  -0.86",
        ]
        grid = VGroup(*[T(s, font=MONO, font_size=16, color=TXT) for s in vals]).arrange(DOWN, buff=0.2)
        hdr = T("x · Tensor", font=MONO, font_size=17, color=MUTED)
        box = RoundedRectangle(corner_radius=0.16, width=grid.width + 0.9, height=grid.height + 1.4, stroke_color=EDGE, stroke_width=1.6, fill_color=CARD, fill_opacity=1).move_to([0, STAGE_Y, 0])
        hdr.next_to(box.get_top(), DOWN, buff=0.2)
        grid.move_to(box).shift(DOWN * 0.15)
        meta = T("shape (8, 16) · float32 · cpu", font=MONO, font_size=16, color=MUTED).next_to(box, DOWN, buff=0.3)
        self.play(FadeIn(box), FadeIn(hdr), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in grid], lag_ratio=0.2), FadeIn(meta), run_time=1.0)
        self.wait(3.5)

        switch("DRAIN", "抽乾數值", "from_tensor() 把數值整袋倒掉：storage 落在 meta device，一個 byte 都不留")
        tag = T("FakeTensorMode.from_tensor(x)", font=MONO, font_size=16, color=ACCENT).next_to(box, UP, buff=0.35)
        self.play(FadeIn(tag, shift=UP * 0.1), run_time=0.4)
        self.play(LaggedStart(*[r.animate.shift(DOWN * 1.3).set_opacity(0) for r in grid], lag_ratio=0.15), run_time=1.3)
        self.remove(grid)
        dashed = DashedVMobject(RoundedRectangle(corner_radius=0.16, width=box.width, height=box.height), num_dashes=64).set_stroke(ACCENT, 1.8).move_to(box)
        hdr2 = T("x · FakeTensor", font=MONO, font_size=17, color=ACCENT).move_to(hdr)
        self.play(box.animate.set_stroke(width=0).set_fill(CARD_DIM), Create(dashed), ReplacementTransform(hdr, hdr2), run_time=0.9)
        self.wait(2.5)

        switch("SHELL", "只剩 metadata", "殼上記著 shape、dtype、stride，device 欄位撒謊報 cpu，data 永遠是空的")
        fields = VGroup(
            T("shape   (8, 16)", font=MONO, font_size=16, color=TXT),
            T("dtype   float32", font=MONO, font_size=16, color=TXT),
            T("device  cpu  <- fake_device", font=MONO, font_size=16, color=TXT),
            T("data    (meta, 0 bytes)", font=MONO, font_size=16, color=DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        fields.next_to(hdr2, DOWN, buff=0.3).set_x(box.get_x())
        fields[2][6:9].set_color(ACCENT)
        self.play(ReplacementTransform(meta, fields), FadeOut(tag), run_time=0.8)
        self.play(Indicate(fields[2][6:9], scale_factor=1.15, color=ACCENT), run_time=0.6)
        self.wait(3)

        switch("OP", "空殼相撞", "op 進來：dispatch 攔下 matmul 丟給 meta kernel，只憑兩張殼就推出新 shape")
        sx = shell("x · (8, 16)").move_to([-4.6, STAGE_Y + 0.75, 0])
        self.play(ReplacementTransform(VGroup(box, dashed, hdr2, fields), sx), run_time=0.9)
        sw = shell("w · (16, 4)").move_to([-8, STAGE_Y - 0.75, 0])
        self.play(sw.animate.move_to([-4.6, STAGE_Y - 0.75, 0]), run_time=0.7)
        g1 = gate("aten.matmul", "FakeTensorMode -> meta kernel").move_to([-0.4, STAGE_Y, 0])
        self.play(FadeIn(g1, shift=RIGHT * 0.2), run_time=0.5)
        self.play(sx.animate.move_to(g1.get_center()).scale(0.3).set_opacity(0), sw.animate.move_to(g1.get_center()).scale(0.3).set_opacity(0), run_time=0.8)
        self.remove(sx, sw)
        self.play(Indicate(g1, scale_factor=1.05, color=ACCENT), run_time=0.5)
        so = shell("(8, 4)")
        so.move_to([g1.get_right()[0] + 0.5 + so.width / 2, STAGE_Y, 0])
        self.play(FadeIn(so, shift=RIGHT * 0.3), run_time=0.6)
        self.wait(3.5)

        switch("CHAIN", "一路推下去", "tanh 是 pointwise，shape 原樣通過：整個 forward 就這樣零成本乾跑完")
        g2 = gate("aten.tanh", "meta kernel").move_to([3.9, STAGE_Y, 0])
        self.play(VGroup(g1, so).animate.shift(LEFT * 2.2), FadeIn(g2, shift=RIGHT * 0.2), run_time=0.7)
        self.play(so.animate.move_to(g2.get_center()).scale(0.3).set_opacity(0), run_time=0.6)
        self.remove(so)
        s2 = shell("(8, 4)")
        s2.move_to([g2.get_right()[0] + 0.5 + s2.width / 2, STAGE_Y, 0])
        self.play(Indicate(g2, scale_factor=1.05, color=ACCENT), FadeIn(s2, shift=RIGHT * 0.3), run_time=0.7)
        self.wait(2.5)

        switch("ITEM", "殼答不上來", ".item() 要一個具體數值：殼裡什麼都沒有，fake 世界當場卡死")
        self.play(FadeOut(g1), FadeOut(g2), s2.animate.move_to([0, STAGE_Y, 0]).scale(1.25), run_time=0.8)
        tok = T(".item()", font=MONO, font_size=20, color=TXT).move_to([5.5, STAGE_Y, 0])
        self.play(FadeIn(tok, shift=LEFT * 0.3), run_time=0.4)
        stop = s2.get_right()[0] + tok.width / 2 + 0.35
        self.play(tok.animate.move_to([stop, STAGE_Y, 0]), run_time=0.7)
        self.play(tok.animate.set_color(ACCENT), Wiggle(s2, scale_value=1.08, rotation_angle=0.02 * TAU), run_time=0.7)
        self.play(s2[1].animate.set_stroke(ACCENT, 2.4), s2[2].animate.set_color(ACCENT), Flash(s2.get_center(), color=ACCENT, line_length=0.18, flash_radius=0.65), run_time=0.5)
        err = VGroup(
            T("DataDependentOutputException", font=MONO, font_size=18, color=ACCENT),
            T("aten._local_scalar_dense.default", font=MONO, font_size=16, color=MUTED),
        ).arrange(DOWN, buff=0.18).next_to(s2, DOWN, buff=0.5)
        self.play(FadeIn(err, shift=UP * 0.15), run_time=0.5)
        self.wait(3.5)

        switch("RULE", "空殼的邊界", "只推 metadata、不碰數值：真正需要數值的那一刻，就是空殼的極限")
        self.play(VGroup(s2, tok, err).animate.set_opacity(0.45), run_time=0.4)
        self.wait(5.5)
