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


def card(name, kind, w=1.9):
    nm = T(name, font=MONO, font_size=16, color=TXT)
    tp = T(kind, font=MONO, font_size=16, color=MUTED)
    inner = VGroup(nm, tp).arrange(DOWN, buff=0.14)
    box = RoundedRectangle(corner_radius=0.14, width=w, height=1.0, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    return VGroup(box, inner.move_to(box))


def chip(name):
    t = T(name, font=MONO, font_size=16, color=MUTED)
    bg = RoundedRectangle(corner_radius=0.12, width=t.width + 0.34, height=0.42, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD_DIM, fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def edge_arrow(a, b):
    return Line(a.get_right(), b.get_left(), stroke_color=DIM, stroke_width=2.4).set_z_index(-2)


ROW_Y = 0.9
XS = [-5.1, -1.7, 1.7, 5.1]


class Scheduler(Scene):
    def construct(self):
        title = T("f(x) -> relu((sin(x) + 1).sum())", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("SETUP", "排程單位", "lowering 完的 IR node 逐顆包成 SchedulerNode，這是排程與融合的基本單位")
        cards = [
            card("op0 · sin", "Pointwise").move_to([XS[0], ROW_Y, 0]),
            card("op1 · add", "Pointwise").move_to([XS[1], ROW_Y, 0]),
            card("op2 · sum", "Reduction").move_to([XS[2], ROW_Y, 0]),
            card("op3 · relu", "Pointwise").move_to([XS[3], ROW_Y, 0]),
        ]
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cards], lag_ratio=0.15), run_time=1.2)
        tlabel = T("memory traffic", font=MONO, font_size=16, color=MUTED)
        tval = T("24 MB", font=MONO, font_size=22, color=TXT)
        traffic = VGroup(tlabel, tval).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(traffic, shift=UP * 0.1), run_time=0.4)
        self.wait(3.5)

        switch("DEPS", "讀寫依賴", "compute_dependencies 只看讀寫，op1 讀 op0 寫的 buf0，一條 MemoryDep 就是一條邊")
        arrows = [edge_arrow(cards[i], cards[i + 1]) for i in range(3)]
        chips = [chip(f"buf{i}").move_to([(XS[i] + XS[i + 1]) / 2, ROW_Y, 0]) for i in range(3)]
        for a, ch in zip(arrows, chips):
            self.play(Create(a), FadeIn(ch, scale=0.8), run_time=0.45)
        self.wait(3.5)

        switch("SCORE", "打分", "score_fusion_memory 把共享的讀寫加總，buf0 的 4MB 就是融合能省下的流量")
        self.play(cards[0][0].animate.set_stroke(ACCENT, 2.2), cards[1][0].animate.set_stroke(ACCENT, 2.2), chips[0][0].animate.set_stroke(ACCENT, 2), chips[0][1].animate.set_color(ACCENT), run_time=0.4)
        score = T("shared data 4 MB", font=MONO, font_size=16, color=ACCENT).next_to(chips[0], UP, buff=0.55)
        self.play(FadeIn(score, shift=UP * 0.1), Indicate(chips[0], scale_factor=1.12, color=ACCENT), run_time=0.6)
        self.wait(2.5)

        switch("FUSE", "融成一顆", "op0 與 op1 融成一顆，buf0 不用寫回記憶體，流量少 8MB")
        fused1 = card("op0_op1", "FusedSchedulerNode", w=2.9).move_to([(XS[0] + XS[1]) / 2, ROW_Y, 0])
        fused1[0].set_stroke(ACCENT, 1.8)
        tval2 = T("16 MB", font=MONO, font_size=22, color=ACCENT).move_to(tval, aligned_edge=LEFT)
        self.play(FadeOut(score), FadeOut(arrows[0]), FadeOut(arrows[1]), chips[0].animate.scale(0.2).set_opacity(0), ReplacementTransform(VGroup(cards[0], cards[1]), fused1), run_time=0.9)
        self.remove(chips[0])
        arr1 = edge_arrow(fused1, cards[2])
        self.play(Create(arr1), chips[1].animate.move_to([(fused1.get_right()[0] + cards[2].get_left()[0]) / 2, ROW_Y, 0]), Transform(tval, tval2), run_time=0.6)
        self.play(Flash(tval, color=ACCENT, line_length=0.12, flash_radius=0.5), run_time=0.4)
        self.wait(3.5)

        switch("ROUND 2", "下一輪", "融完的節點下一輪繼續當候選人，pointwise 整串被吸進 reduction 的迴圈")
        fused2 = card("op0_op1_op2", "Reduction · fused", w=3.4).move_to([(XS[0] + XS[2]) / 2 + 0.85, ROW_Y, 0])
        fused2[0].set_stroke(ACCENT, 1.8)
        tval3 = T("8 MB", font=MONO, font_size=22, color=ACCENT).move_to(tval, aligned_edge=LEFT)
        self.play(FadeOut(arr1), FadeOut(arrows[2]), chips[1].animate.scale(0.2).set_opacity(0), ReplacementTransform(VGroup(fused1, cards[2]), fused2), run_time=0.9)
        self.remove(chips[1])
        arr2 = edge_arrow(fused2, cards[3])
        self.play(Create(arr2), chips[2].animate.move_to([(fused2.get_right()[0] + cards[3].get_left()[0]) / 2, ROW_Y, 0]), Transform(tval, tval3), run_time=0.6)
        self.wait(2.5)

        switch("WALL", "reduction 的牆", "op3 每一格都要等 sum 的最終值，迴圈形狀對不上，融合被擋下")
        g1 = T("group ((), (1048576,))", font=MONO, font_size=16, color=MUTED).next_to(fused2, DOWN, buff=0.35)
        g2 = T("group ((1048576,), ())", font=MONO, font_size=16, color=MUTED).next_to(cards[3], DOWN, buff=0.35)
        self.play(FadeIn(g1, shift=UP * 0.08), FadeIn(g2, shift=UP * 0.08), run_time=0.5)
        self.play(g1[9:19].animate.set_color(ACCENT), g2[6:16].animate.set_color(ACCENT), run_time=0.4)
        cross = VGroup(Line(UL * 0.16, DR * 0.16), Line(UR * 0.16, DL * 0.16)).set_stroke(ACCENT, 4).move_to(chips[2])
        self.play(FadeIn(cross, scale=1.4), chips[2][1].animate.set_color(ACCENT), Flash(chips[2], color=ACCENT, line_length=0.14, flash_radius=0.45), run_time=0.6)
        bx = (fused2.get_right()[0] + cards[3].get_left()[0]) / 2
        wall = DashedLine([bx, ROW_Y + 1.55, 0], [bx, ROW_Y - 1.55, 0], stroke_color=ACCENT, stroke_width=2.6, dash_length=0.14)
        wtag = T("kernel boundary", font=MONO, font_size=16, color=ACCENT).next_to(wall, UP, buff=0.18)
        self.play(FadeOut(cross), FadeOut(chips[2]), FadeOut(arr2), Create(wall), FadeIn(wtag, shift=UP * 0.1), run_time=0.8)
        self.wait(3.5)

        switch("RESULT", "座位表定案", "兩張桌子兩個 kernel，計算一次沒少做，省下的全是記憶體往返")
        f1 = RoundedRectangle(corner_radius=0.18, width=fused2.width + 0.5, height=fused2.height + 0.9, stroke_color=MUTED, stroke_width=1.8, fill_opacity=0).move_to(fused2)
        f2 = RoundedRectangle(corner_radius=0.18, width=cards[3].width + 0.5, height=cards[3].height + 0.9, stroke_color=MUTED, stroke_width=1.8, fill_opacity=0).move_to(cards[3])
        k1 = T("kernel 1", font=MONO, font_size=16, color=MUTED).next_to(f1, UP, buff=0.2)
        k2 = T("kernel 2", font=MONO, font_size=16, color=MUTED).next_to(f2, UP, buff=0.2)
        self.play(FadeOut(g1), FadeOut(g2), FadeOut(wtag), Create(f1), Create(f2), run_time=0.8)
        self.play(FadeIn(k1, shift=UP * 0.1), FadeIn(k2, shift=UP * 0.1), run_time=0.4)
        self.play(Indicate(tval, scale_factor=1.15, color=ACCENT), run_time=0.6)
        self.wait(5.5)
