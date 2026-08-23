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
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    zt = T(zh, font=zh_font, font_size=17, color=BG)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def cell(label, ghost=False):
    sq = Square(side_length=0.58, stroke_color=DIM if ghost else EDGE, stroke_width=1.5, fill_color=BG if ghost else CARD, fill_opacity=1)
    t = T(label, font=MONO, font_size=16, color=DIM if ghost else TXT)
    return VGroup(sq, t.move_to(sq))


def chip(name):
    t = T(name, font=MONO, font_size=16, color=MUTED)
    bg = RoundedRectangle(corner_radius=0.12, width=t.width + 0.34, height=0.42, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD_DIM, fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


ROW_Y = 1.95
PITCH = 0.66
CENTERS = [-3.2, 0.0, 3.2]


class Triton(Scene):
    def construct(self):
        title = T("relu(x + y) * 2", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("TENSOR", "攤平成一條", "Inductor 把輸出攤平成一條線性索引,這裡拿 xnumel = 10 當縮小版例子")
        cells = [cell(str(i)) for i in range(10)]
        for i, c in enumerate(cells):
            c.move_to([(i - 4.5) * PITCH, ROW_Y, 0])
        xn = T("xnumel = 10", font=MONO, font_size=17, color=MUTED).next_to(cells[0], LEFT, buff=0.5)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in cells], lag_ratio=0.06), run_time=1.0)
        self.play(FadeIn(xn, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(3.5)

        switch("SPLIT", "切成磚", "XBLOCK = 4 一磚四格,10 格切成 3 塊磚,每塊磚配一個 program instance")
        ghosts = [cell(str(10 + i), ghost=True) for i in range(2)]
        for i, g in enumerate(ghosts):
            g.move_to([(10 + i - 4.5) * PITCH, ROW_Y, 0])
        allc = cells + ghosts
        moves = []
        for gi in range(3):
            for j in range(4):
                moves.append(allc[gi * 4 + j].animate.move_to([CENTERS[gi] + (j - 1.5) * PITCH, ROW_Y, 0]))
        self.play(FadeIn(ghosts[0]), FadeIn(ghosts[1]), xn.animate.shift(LEFT * 1.1), run_time=0.4)
        self.play(*moves, run_time=0.9)
        chips = [chip(f"pid {gi}").move_to([CENTERS[gi], ROW_Y - 0.72, 0]) for gi in range(3)]
        self.play(LaggedStart(*[FadeIn(ch, shift=UP * 0.1) for ch in chips], lag_ratio=0.15), run_time=0.6)
        self.wait(3.5)

        switch("XINDEX", "一人一塊磚", "每個 instance 只憑 pid 就算出自己的 xindex，一次管 XBLOCK 條 lane")
        grp2 = VGroup(*allc[8:12])
        self.play(*[allc[i][0].animate.set_stroke(ACCENT, 2.2) for i in (8, 9)], chips[2][0].animate.set_stroke(ACCENT, 1.8), chips[2][1].animate.set_color(ACCENT), run_time=0.4)
        card = RoundedRectangle(corner_radius=0.16, width=10.4, height=2.9, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1).move_to([0, -1.62, 0])
        hdr = T("triton_poi_fused_add_mul_relu_0 · pid 2", font=MONO, font_size=16, color=MUTED).move_to(card.get_corner(UL) + [0.35, -0.35, 0], aligned_edge=UL)
        lanes = VGroup(*[Square(side_length=0.55, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1) for _ in range(4)]).arrange(RIGHT, buff=0.1).move_to([3.15, -1.75, 0])
        rlab = T("registers", font=MONO, font_size=16, color=MUTED).next_to(lanes, UP, buff=0.22)
        self.play(FadeIn(card), FadeIn(hdr), FadeIn(lanes), FadeIn(rlab), run_time=0.5)

        codeslot = [None]

        def show_lines(lines, keep=False):
            new = VGroup(*[T(s, font=MONO, font_size=17, color=TXT) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            new.move_to(card.get_corner(UL) + [0.35, -0.85, 0], aligned_edge=UL)
            anims = [FadeIn(new, shift=UP * 0.08)]
            if codeslot[0] is not None and not keep:
                anims.append(FadeOut(codeslot[0]))
            self.play(*anims, run_time=0.5)
            codeslot[0] = new
            return new

        l1 = show_lines(["xoffset = pid * XBLOCK -> 8", "xindex = xoffset + arange(0, 4) -> [8 9 10 11]"])
        echo = VGroup(allc[8][1].copy(), allc[9][1].copy(), allc[10][1].copy(), allc[11][1].copy())
        self.play(Transform(echo, l1[1][-9:-1].copy().set_color(ACCENT)), run_time=0.8)
        self.remove(echo)
        self.wait(3.5)

        switch("MASK", "遮住越界", "10 和 11 不存在,xmask 把最後兩條 lane 關掉,load 和 store 都帶著它")
        l2 = show_lines(["xmask = xindex < 10 -> [1 1 0 0]"])
        crosses = VGroup()
        for i in (10, 11):
            cr = VGroup(Line(UL * 0.16, DR * 0.16), Line(UR * 0.16, DL * 0.16)).set_stroke(ACCENT, 3.5).move_to(allc[i])
            crosses.add(cr)
        self.play(FadeIn(crosses, scale=1.3), *[allc[i][1].animate.set_color(DIM) for i in (10, 11)], *[lanes[k].animate.set_fill(BG).set_stroke(DIM, 1.5) for k in (2, 3)], run_time=0.6)
        self.wait(2.5)

        switch("LOAD", "把磚吸進來", "tl.load 一次把一塊磚吸進 register,被遮住的 lane 什麼都不讀")
        show_lines(["tmp0 = tl.load(in_ptr0 + x0, xmask)", "tmp1 = tl.load(in_ptr1 + x0, xmask)"])
        flying = VGroup(allc[8].copy(), allc[9].copy())
        vals = VGroup()
        for k in range(2):
            v = T(str(8 + k), font=MONO, font_size=16, color=TXT).move_to(lanes[k])
            vals.add(v)
        self.play(flying[0].animate.scale(0.9).move_to(lanes[0]), flying[1].animate.scale(0.9).move_to(lanes[1]), run_time=0.9)
        self.remove(flying)
        self.add(vals)
        self.play(*[lanes[k].animate.set_stroke(ACCENT, 1.8) for k in (0, 1)], run_time=0.3)
        self.wait(2.5)

        switch("COMPUTE", "在暫存器交棒", "add、relu、mul 三個 op 在 register 裡一路交棒,中間值從頭到尾不落地")
        show_lines(["tmp2 = tmp0 + tmp1", "tmp4 = maximum(0, tmp2)", "tmp6 = tmp4 * 2.0"])
        self.play(Indicate(VGroup(lanes[0], lanes[1], vals), scale_factor=1.1, color=ACCENT), run_time=0.7)
        self.play(vals[0].animate.set_color(ACCENT), vals[1].animate.set_color(ACCENT), run_time=0.3)
        self.wait(3)

        switch("STORE", "放回去", "tl.store 帶著同一個 xmask 把結果寫回 out_ptr0,一磚進一磚出")
        show_lines(["tl.store(out_ptr0 + x0, tmp6, xmask)"])
        back = VGroup(vals[0].copy(), vals[1].copy())
        self.play(back[0].animate.move_to(allc[8]), back[1].animate.move_to(allc[9]), run_time=0.9)
        self.remove(back)
        self.play(*[allc[i][0].animate.set_fill("#3a2a20") for i in (8, 9)], run_time=0.4)
        self.wait(2.5)

        switch("GRID", "一起開工", "wrapper 只傳 xnumel,heuristics 算出 grid,3 個 instance 其實同時在跑")
        gline = VGroup(
            T("triton_poi_fused_add_mul_relu_0.run(x, y, buf0, 10, stream=stream0)", font=MONO, font_size=16, color=TXT),
            T("grid = (ceil(xnumel / XBLOCK), 1, 1) -> (3, 1, 1)", font=MONO, font_size=16, color=ACCENT),
        ).arrange(DOWN, buff=0.18).move_to([0, 0.55, 0])
        self.play(card.animate.set_fill(opacity=0.55), VGroup(hdr, codeslot[0], lanes, rlab, vals).animate.set_opacity(0.45), FadeIn(gline, shift=UP * 0.1), run_time=0.6)
        self.play(*[c[0].animate.set_stroke(ACCENT, 2.2) for c in allc[:10]], *[ch[0].animate.set_stroke(ACCENT, 1.8) for ch in chips], *[ch[1].animate.set_color(ACCENT) for ch in chips], run_time=0.5)
        self.play(*[Flash(chips[g], color=ACCENT, line_length=0.12, flash_radius=0.5) for g in range(3)], run_time=0.5)
        self.wait(5.5)
