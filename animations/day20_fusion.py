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


def pill(name, zh):
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    runs = re.findall(r"[一-鿿，、。]+|[^一-鿿，、。 ]+", zh)
    zs = [T(r, font=CJK if re.search(r"[一-鿿]", r) else MONO, font_size=17, color=BG) for r in runs]
    zt = VGroup(*zs).arrange(RIGHT, buff=0.1)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def op_box(label, reduction=False):
    t = T(label, font=MONO, font_size=16, color=TXT)
    box = RoundedRectangle(corner_radius=0.12, width=1.0, height=0.62, stroke_color=ACCENT if reduction else EDGE, stroke_width=1.8 if reduction else 1.5, fill_color=CARD, fill_opacity=1)
    return VGroup(box, t.move_to(box))


def flow_arrow(a, b):
    return Line(a.get_right(), b.get_left(), stroke_color=DIM, stroke_width=2.2, buff=0.06).add_tip(tip_length=0.12, tip_width=0.12)


def hull(group, pad_w=0.3, pad_h=0.3):
    return RoundedRectangle(corner_radius=0.16, width=group.width + pad_w, height=group.height + pad_h, stroke_color=ACCENT, stroke_width=1.8, fill_color=ACTIVE_FILL, fill_opacity=1).move_to(group).set_z_index(-1)


def wall_piece(height=2.4):
    w = Rectangle(width=0.3, height=height, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1)
    lines = VGroup(*[Line(w.get_left() + UP * y, w.get_right() + UP * y, stroke_color=EDGE, stroke_width=1.2) for y in np.arange(-height / 2 + 0.3, height / 2, 0.3)])
    return VGroup(w, lines)


ROW_Y = 0.55


class Fusion(Scene):
    def construct(self):
        title = T("y = sin(relu(x+1)*2)  ->  relu(y - y.mean())", font=MONO, font_size=22, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("OPS", "七個 op", "lowering 之後的資料流：六個 pointwise 加一個 reduction，eager 的話就是七顆 kernel")
        names = ["add", "relu", "mul", "sin", "mean", "sub", "relu"]
        boxes = VGroup(*[op_box(n, reduction=(n == "mean")) for n in names]).arrange(RIGHT, buff=0.42).move_to([0.3, ROW_Y, 0])
        xin = T("x", font=MONO, font_size=18, color=MUTED).next_to(boxes[0], LEFT, buff=0.55)
        arrows = VGroup(flow_arrow(xin, boxes[0]), *[flow_arrow(boxes[i], boxes[i + 1]) for i in range(6)])
        rtag = T("reduction", font=MONO, font_size=16, color=ACCENT).next_to(boxes[4], UP, buff=0.22)
        self.play(FadeIn(xin), LaggedStart(*[FadeIn(m, shift=RIGHT * 0.15) for m in [*boxes, *arrows]], lag_ratio=0.06), run_time=1.4)
        self.play(FadeIn(rtag, shift=UP * 0.08), run_time=0.3)
        self.wait(3.5)

        switch("MAGNET", "pointwise 磁吸", "pointwise 之間沒有牆：中間值不落地，在暫存器裡直接交棒，四個 op 黏成一段")
        chain = VGroup(*boxes[:4])
        self.play(FadeOut(arrows[1]), FadeOut(arrows[2]), FadeOut(arrows[3]), chain.animate.arrange(RIGHT, buff=0.1).move_to([boxes[1].get_center()[0], ROW_Y, 0]), run_time=0.9)
        c_hull = hull(chain)
        self.play(FadeIn(c_hull), Flash(c_hull.get_center(), color=ACCENT, line_length=0.15, flash_radius=0.5), run_time=0.6)
        self.wait(2.5)

        switch("VERTICAL", "垂直融進 reduction", "producer 跟 consumer 融合：整條鏈被吸進 mean 的內圈，一邊掃 x 一邊累加")
        k1_inner = VGroup(c_hull, chain)
        self.play(FadeOut(arrows[4]), FadeOut(rtag), k1_inner.animate.next_to(boxes[4], LEFT, buff=0.14), run_time=0.9)
        k1_hull = hull(VGroup(k1_inner, boxes[4]), pad_w=0.4, pad_h=0.42)
        k1_tag = T("kernel #1", font=MONO, font_size=16, color=ACCENT).next_to(k1_hull, DOWN, buff=0.22)
        self.play(ReplacementTransform(c_hull, k1_hull), FadeIn(k1_tag, shift=UP * 0.08), run_time=0.8)
        self.wait(3.5)

        switch("WALL", "reduction 的牆", "mean 把整份 input 收成一個值：全域統計量沒出爐，牆後任何一個元素都動不了筆")
        k1 = VGroup(k1_hull, k1_inner, boxes[4], k1_tag)
        tail = VGroup(*boxes[5:])
        self.play(FadeOut(arrows[5]), k1.animate.shift(LEFT * 0.55), xin.animate.shift(LEFT * 0.55), arrows[0].animate.shift(LEFT * 0.55), tail.animate.shift(RIGHT * 0.75), arrows[6].animate.shift(RIGHT * 0.75), run_time=0.7)
        wall = wall_piece().move_to([(k1_hull.get_right()[0] + boxes[5].get_left()[0]) / 2, ROW_Y, 0])
        wtag = T("(numel, rnumel) 對不上", font=MONO, font_size=16, color=MUTED, t2f={"對不上": CJK}).next_to(wall, UP, buff=0.28)
        self.play(GrowFromEdge(wall, DOWN), FadeIn(wtag, shift=UP * 0.08), run_time=0.7)
        self.play(boxes[5].animate(rate_func=there_and_back).shift(LEFT * 0.4), run_time=0.5)
        self.play(wall[0].animate.set_stroke(ACCENT, 2.2), Flash(wall.get_center(), color=ACCENT, line_length=0.16, flash_radius=0.45), run_time=0.5)
        no_fuse = T("found 0 possible fusions", font=MONO, font_size=16, color=ACCENT).next_to(wall, DOWN, buff=0.35)
        self.play(FadeIn(no_fuse, shift=UP * 0.1), run_time=0.3)
        self.wait(3.5)

        switch("RESTART", "牆後另起一顆", "sub 跟 relu 只好自己開一顆 kernel：x 得重讀一遍，這就是牆的代價")
        self.play(FadeOut(no_fuse), FadeOut(arrows[6]), tail.animate.arrange(RIGHT, buff=0.1).move_to([wall.get_right()[0] + 1.75, ROW_Y, 0]), run_time=0.8)
        k2_hull = hull(tail, pad_w=0.4, pad_h=0.42)
        k2_tag = T("kernel #2", font=MONO, font_size=16, color=ACCENT).next_to(k2_hull, DOWN, buff=0.22)
        self.play(FadeIn(k2_hull), FadeIn(k2_tag, shift=UP * 0.08), Flash(k2_hull.get_center(), color=ACCENT, line_length=0.15, flash_radius=0.5), run_time=0.7)
        self.wait(2.5)

        switch("COUNT", "數牆就是數 kernel", "融合的邊界就是 kernel 的邊界：能融的都融了，剩下的每一道牆就是一次 launch")
        lhs = T("7 ops", font=MONO, font_size=26, color=MUTED)
        arr = T("->", font=MONO, font_size=26, color=DIM)
        rhs = T("2 kernels", font=MONO, font_size=26, color=ACCENT)
        summary = VGroup(lhs, arr, rhs).arrange(RIGHT, buff=0.4).move_to([0, -1.75, 0])
        self.play(FadeIn(lhs, shift=UP * 0.1), run_time=0.4)
        self.play(FadeIn(arr), TransformFromCopy(VGroup(k1_tag, k2_tag), rhs), run_time=0.9)
        self.play(Indicate(rhs, scale_factor=1.08, color=ACCENT), run_time=0.6)
        self.wait(5.5)
