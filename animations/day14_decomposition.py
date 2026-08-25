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
    inner = VGroup(m, s).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    r = panel(w, inner.height + 0.3, fill=fill, edge=edge, r=0.08)
    inner.move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, inner)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


LN_ROWS = [
    "var_mean = var_mean(x, [1])",
    "add    = add(var, 1e-05)",
    "rsqrt  = rsqrt(add)",
    "sub    = sub(x, mean)",
    "mul    = mul(sub, rsqrt)",
    "mul_1  = mul(mul, weight)",
    "add_1  = add(mul_1, bias)",
]
GELU_ROWS = [
    "mul_2 = mul(add_1, 0.5)",
    "mul_3 = mul(add_1, 0.7071)",
    "erf   = erf(mul_3)",
    "add_2 = add(erf, 1)",
    "mul_4 = mul(mul_2, add_2)",
]


class Decomposition(Scene):
    def construct(self):
        title = T("f(x) = gelu(ln(x))", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
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
        MID_Y = (TOP + BOT) / 2
        LW, MW, RW = 3.9, 4.9, 3.9
        LX, MX, RX = -4.7, 0.0, 4.7
        src = titled(LW, H, "SOURCE", "torch 層").move_to([LX, MID_Y, 0])
        dec = titled(MW, H, "DECOMPOSE", "aot_graphs", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([MX, MID_Y, 0])
        ind = titled(RW, H, "INDUCTOR", "融合收攏").move_to([RX, MID_Y, 0])
        lbls = [label("OPS  ·  2000+ 個").next_to(src, UP, buff=0.22).align_to(src, LEFT),
                label("TABLE  ·  1123 條規則").next_to(dec, UP, buff=0.22).align_to(dec, LEFT),
                label("KERNELS  ·  拼回去").next_to(ind, UP, buff=0.22).align_to(ind, LEFT)]
        switch("SETUP", "兩個高階 op", "torch 層有兩千多個 op，後端寫不完；decomposition 把它們拆成少數基本運算")
        self.play(FadeIn(src), FadeIn(dec), FadeIn(ind), *[FadeIn(l) for l in lbls], run_time=0.5)

        cw = LW - 0.5
        ln_chip = chip("ln = LayerNorm(8)", "高階 op", cw).next_to(src[1], DOWN, buff=0.35).align_to(src[1], LEFT)
        gelu_chip = chip("F.gelu(...)", "高階 op", cw).next_to(ln_chip, DOWN, buff=0.2).align_to(ln_chip, LEFT)
        self.play(FadeIn(ln_chip, shift=RIGHT * 0.1), FadeIn(gelu_chip, shift=RIGHT * 0.1), run_time=0.4)
        self.wait(2.5)

        ln_hdr = label("LayerNorm ->", size=11, color=ACCENT)
        gelu_hdr = label("GELU ->", size=11, color=ACCENT)
        rows = [ln_hdr] + [T(s, font=MONO, font_size=11, color=TXT) for s in LN_ROWS] + [gelu_hdr] + [T(s, font=MONO, font_size=11, color=TXT) for s in GELU_ROWS]
        col = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(dec[1], DOWN, buff=0.3).align_to(dec[1], LEFT)
        ln_part = VGroup(*rows[:8])
        gelu_part = VGroup(*rows[8:])

        switch("STEP 1", "LayerNorm 炸開", "查表：LayerNorm 拆成 var_mean、rsqrt、sub、mul、add，一個 op 變七行基本運算")
        self.play(ln_chip[0].animate.set_stroke(color=ACCENT, width=2), run_time=0.25)
        a1 = arrow(ln_chip[0].get_right(), [dec[0].get_left()[0], ln_chip.get_y(), 0], color=ACCENT)
        self.play(GrowArrow(a1), run_time=0.3)
        for r in ln_part:
            self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.12)
        self.wait(3.5)

        switch("STEP 2", "GELU 炸開", "GELU 拆成數學定義 0.5 * x * (1 + erf(x / sqrt(2)))，0.7071 就是 1 / sqrt(2)")
        self.play(ln_chip[0].animate.set_stroke(color=EDGE, width=1.5), gelu_chip[0].animate.set_stroke(color=ACCENT, width=2), run_time=0.25)
        a2 = arrow(gelu_chip[0].get_right(), [dec[0].get_left()[0], gelu_chip.get_y(), 0], color=ACCENT)
        self.play(GrowArrow(a2), run_time=0.3)
        for r in gelu_part:
            self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.12)
        self.wait(3.5)

        switch("STEP 3", "戰略 op 不拆", "matmul、conv 拆成迴圈就認不出來了，留著原樣直達後端，交給 matmul template")
        self.play(gelu_chip[0].animate.set_stroke(color=EDGE, width=1.5), run_time=0.2)
        mm_chip = chip("x @ w", "戰略 op，不在表裡", cw).next_to(gelu_chip, DOWN, buff=0.2).align_to(gelu_chip, LEFT)
        self.play(FadeIn(mm_chip, shift=RIGHT * 0.1), run_time=0.3)
        rw = RW - 0.5
        mm_out = chip("mm", "matmul template / cuBLAS", rw).move_to([RX, mm_chip.get_y(), 0]).align_to(ind[1], LEFT)
        ghost = mm_chip.copy()
        self.play(ghost.animate.move_to(mm_out).align_to(mm_out, LEFT), run_time=0.9)
        self.play(ReplacementTransform(ghost, mm_out), run_time=0.3)
        self.wait(3.5)

        switch("STEP 4", "Inductor 融合收攏", "拆出來的 pointwise 被融回一個 kernel：讀一次、算完、寫一次，拆了不會變慢")
        pw = VGroup(*rows[2:8], *rows[9:])
        box_pw = SurroundingRectangle(pw, color=ACCENT, buff=0.1, corner_radius=0.08, stroke_width=1.5)
        self.play(Create(box_pw), run_time=0.5)
        kchip = chip("triton_poi_fused_*", "10 個 pointwise -> 1 kernel", rw, main_color=ACCENT, edge=ACCENT, fill=ACTIVE_FILL).next_to(ind[1], DOWN, buff=0.35).align_to(ind[1], LEFT)
        pw_ghost = pw.copy()
        self.play(pw_ghost.animate.move_to(kchip), run_time=0.9)
        self.play(ReplacementTransform(pw_ghost, kchip), FadeOut(box_pw), run_time=0.35)
        self.wait(2.5)
        rchip = chip("triton_red_*", "var_mean 是 reduction", rw).next_to(kchip, DOWN, buff=0.2).align_to(kchip, LEFT)
        vm_ghost = rows[1].copy()
        self.play(vm_ghost.animate.move_to(rchip), run_time=0.7)
        self.play(ReplacementTransform(vm_ghost, rchip), run_time=0.3)
        note = VGroup(T("拆解 + 融合", font=CJK, font_size=13, color=MUTED), T("= 免手寫 fused kernel", font=CJK, font_size=13, color=MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(mm_out, DOWN, buff=0.4).align_to(mm_out, LEFT)
        self.play(FadeIn(note), run_time=0.3)
        self.wait(3.5)

        switch("RULE", "拆是為了拼回去", "規則是普通的 Python 函式、表是可以換的字典：一份基本 op 的 codegen 涵蓋所有組合")
        self.wait(5.5)
