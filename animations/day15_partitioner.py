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
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    zt = T(zh, font=zh_font, font_size=17, color=BG)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def label(s, size=16, color=MUTED):
    return T(s, font=MONO, font_size=size, color=color)


def node(txt, accent=False):
    t = T(txt, font=MONO, font_size=16, color=TXT)
    r = RoundedRectangle(corner_radius=0.09, width=t.width + 0.5, height=0.55, stroke_color=ACCENT if accent else EDGE, stroke_width=2 if accent else 1.5, fill_color=ACTIVE_FILL if accent else CARD_DIM, fill_opacity=1)
    return VGroup(r, t.move_to(r))


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


def blade(p, q):
    glow = Line(p, q, stroke_color=ACCENT, stroke_width=14, stroke_opacity=0.2)
    core = Line(p, q, stroke_color=ACCENT, stroke_width=4)
    return VGroup(glow, core)


class MinCut(Scene):
    def construct(self):
        title = T("f(x, w) = tanh(x @ w).sum()", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        Y1, Y2 = 1.7, -0.7
        n_pr = node("primals").move_to([-5.3, Y1, 0])
        n_mm = node("mm").move_to([-3.6, Y1, 0])
        n_th = node("tanh").move_to([-2.0, Y1, 0])
        n_sm = node("sum_1").move_to([-0.4, Y1, 0])
        n_d = node("1 - tanh^2").move_to([1.0, Y2, 0])
        n_ml = node("mul_1").move_to([3.05, Y2, 0])
        n_gw = node("grad_w").move_to([4.9, Y2, 0])
        a1 = arrow(n_pr[0].get_right(), n_mm[0].get_left())
        a2 = arrow(n_mm[0].get_right(), n_th[0].get_left())
        a3 = arrow(n_th[0].get_right(), n_sm[0].get_left())
        a_x = arrow(n_th[0].get_bottom(), n_d[0].get_left())
        a4 = arrow(n_d[0].get_right(), n_ml[0].get_left())
        a5 = arrow(n_ml[0].get_right(), n_gw[0].get_left())
        arc = CubicBezier(n_pr[0].get_bottom() + DOWN * 0.06, [-5.3, -1.9, 0], [2.0, -2.4, 0], n_gw[0].get_bottom() + DOWN * 0.06).set_stroke(DIM, 2)
        fw_tag = label("forward", color=DIM).next_to(n_pr, UP, buff=0.25)
        bw_tag = label("backward", color=DIM).next_to(n_gw, UP, buff=0.25)

        switch("SETUP", "共同財產清冊", "joint graph 把 forward 和 backward 畫在同一張圖上，中間沒有邊界，只靠資料流相連")
        self.play(LaggedStart(FadeIn(n_pr), GrowArrow(a1), FadeIn(n_mm), GrowArrow(a2), FadeIn(n_th), GrowArrow(a3), FadeIn(n_sm), lag_ratio=0.15), FadeIn(fw_tag), run_time=1.1)
        self.play(LaggedStart(GrowArrow(a_x), FadeIn(n_d), GrowArrow(a4), FadeIn(n_ml), GrowArrow(a5), FadeIn(n_gw), lag_ratio=0.15), Create(arc), FadeIn(bw_tag), run_time=1.1)
        self.wait(3.5)

        switch("NEED", "backward 要什麼", "backward 要 tanh 算導數、要 primals_1 轉置成 permute：這兩條邊把值從 forward 拉過去")
        self.play(a_x.animate.set_color(ACCENT), Indicate(n_th, color=ACCENT, scale_factor=1.08), run_time=0.6)
        self.play(arc.animate.set_stroke(ACCENT, 2), Indicate(n_pr, color=ACCENT, scale_factor=1.06), run_time=0.6)
        self.wait(2.5)

        t_th = label("16 KB", color=TXT).move_to([-0.3, 0.55, 0])
        t_mm = label("16 KB", color=TXT).move_to([-2.88, 2.15, 0])
        no_re = label("no recompute", color=ACCENT).next_to(n_mm, DOWN, buff=0.2).shift(LEFT * 0.35)
        t_pm = label("permute · 0 B", color=TXT).move_to([0, -2.15, 0])
        switch("COST", "邊上有價格", "每條邊標上保存的價格：保存成本就是 bytes，mm 計算密集禁止重算，permute 只改 metadata 免費")
        self.play(FadeIn(t_th), FadeIn(t_mm), FadeIn(no_re), FadeIn(t_pm), arc.animate.set_stroke(MUTED, 2), run_time=0.5)
        self.wait(3.5)

        bld = blade([-1.1, 2.2, 0], [-2.1, -1.9, 0]).shift(UP * 4.5)
        labA = label("cut A · keep tanh · 16 KB", color=ACCENT).move_to([-1.35, 2.6, 0])
        switch("CUT A", "光刀落下試切", "一刀落在 tanh 之後：割斷 tanh 過去的邊，forward 就得保存 tanh，帳單 16 KB")
        self.play(bld.animate.shift(DOWN * 4.5), run_time=0.5, rate_func=rush_into)
        self.play(Flash([-1.43, 0.8, 0], color=ACCENT, line_length=0.18, flash_radius=0.4), FadeIn(labA), run_time=0.5)
        self.wait(2.5)

        labB = label("cut B · keep mm · 16 KB", color=ACCENT).move_to([-3.0, 2.6, 0])
        n_re = node("tanh", accent=True).move_to([-1.5, Y2, 0])
        re_tag = label("recompute · free", color=ACCENT).move_to([-1.5, -1.2, 0])
        a_sv = arrow(n_mm[0].get_bottom(), n_re[0].get_left(), color=ACCENT, w=2.5)
        a_re = arrow(n_re[0].get_right(), n_d[0].get_left())
        switch("CUT B", "刀往輸入滑", "存 mm 同樣 16 KB，但 tanh 是 pointwise、重算免費：刀滑到 mm 之後，tanh 複製一份歸隊 backward")
        self.play(Transform(bld, blade([-2.7, 2.3, 0], [-3.0, -1.5, 0])), Transform(labA, labB), FadeOut(t_th), FadeOut(t_mm), FadeOut(no_re), run_time=0.8)
        self.play(FadeOut(a_x), TransformFromCopy(n_th, n_re, path_arc=-PI / 3), FadeIn(re_tag), run_time=0.8)
        self.play(GrowArrow(a_sv), GrowArrow(a_re), n_mm[0].animate.set_stroke(ACCENT, width=2).set_fill(ACTIVE_FILL), run_time=0.6)
        self.wait(3.5)

        FX, BX = -4.2, 4.2
        switch("SPLIT", "節點歸隊", "節點沿切線各自歸隊成兩張圖：跨線的 mm 和 permute 就是 forward 的輸出、backward 的輸入")
        self.play(FadeOut(bld), FadeOut(labA), FadeOut(t_pm), FadeOut(re_tag), FadeOut(fw_tag), FadeOut(bw_tag), FadeOut(a1), FadeOut(a2), FadeOut(a3), FadeOut(a4), FadeOut(a5), FadeOut(a_re), run_time=0.4)
        self.play(n_pr.animate.move_to([FX, 2.1, 0]), n_mm.animate.move_to([FX, 1.15, 0]), n_th.animate.move_to([FX, 0.2, 0]), n_sm.animate.move_to([FX, -0.75, 0]), n_re.animate.move_to([BX, 1.15, 0]), n_d.animate.move_to([BX, 0.2, 0]), n_ml.animate.move_to([BX, -0.75, 0]), n_gw.animate.move_to([BX, -1.7, 0]), run_time=1.2)
        n_pm = node("permute").move_to([FX, -1.7, 0])
        fw_h = label("FORWARD").move_to([FX, 2.8, 0])
        bw_h = label("BACKWARD").move_to([BX, 2.8, 0])
        sv_h = label("SAVED").move_to([0, 1.85, 0])
        f_a1 = arrow(n_pr[0].get_bottom(), n_mm[0].get_top())
        f_a2 = arrow(n_mm[0].get_bottom(), n_th[0].get_top())
        f_a3 = arrow(n_th[0].get_bottom(), n_sm[0].get_top())
        b_a1 = arrow(n_re[0].get_bottom(), n_d[0].get_top())
        b_a2 = arrow(n_d[0].get_bottom(), n_ml[0].get_top())
        b_a3 = arrow(n_ml[0].get_bottom(), n_gw[0].get_top())
        self.play(FadeIn(fw_h), FadeIn(bw_h), FadeIn(n_pm), GrowArrow(f_a1), GrowArrow(f_a2), GrowArrow(f_a3), GrowArrow(b_a1), GrowArrow(b_a2), GrowArrow(b_a3), run_time=0.6)
        s1 = arrow([-3.6, 1.15, 0], [3.45, 1.15, 0], color=ACCENT, w=2.5)
        s2 = arrow([-3.0, -1.7, 0], [3.3, -1.7, 0], color=ACCENT, w=2.5)
        c1 = label("mm · 16 KB", color=ACCENT).move_to([0, 1.45, 0])
        c2 = label("permute · 0 B", color=ACCENT).move_to([0, -1.4, 0])
        self.play(Transform(a_sv, s1), Transform(arc, s2), FadeIn(sv_h), FadeIn(c1), FadeIn(c2), run_time=0.9)
        self.wait(3.5)

        ring = Circle(radius=0.45, stroke_color=EDGE, stroke_width=3, fill_color=CARD, fill_opacity=1).move_to([0, -2.55, 0])
        ptr = Line(ring.get_center(), ring.get_center() + rotate_vector(UP * 0.34, 50 * DEGREES), stroke_color=ACCENT, stroke_width=4)
        lab_min = label("1 · min-cut").next_to(ring, LEFT, buff=0.3)
        lab_ck = label("0 · checkpoint", color=ACCENT).next_to(ring, RIGHT, buff=0.3)
        switch("CHECKPOINT", "旋鈕轉到底", "旋鈕轉到 0 就是 checkpoint：只保存 primals，backward 開頭把整段 forward 重播一遍")
        self.play(FadeIn(ring), FadeIn(ptr), FadeIn(lab_min), FadeIn(lab_ck), run_time=0.4)
        self.play(Rotate(ptr, angle=-100 * DEGREES, about_point=ring.get_center()), run_time=1.0)
        c1b = label("primals_1 · 16 KB", color=ACCENT).move_to(c1)
        c2b = label("primals_2 · 16 KB", color=ACCENT).move_to(c2)
        self.play(Transform(c1, c1b), Transform(c2, c2b), run_time=0.6)
        b_mm = node("mm", accent=True).move_to([BX, 2.1, 0])
        b_ar = arrow([BX, 1.825, 0], [BX, 1.425, 0], color=ACCENT)
        self.play(TransformFromCopy(n_mm, b_mm, path_arc=-PI / 5), n_pm.animate.move_to([BX, -2.6, 0]), run_time=0.9)
        rp_line = Line([5.3, 2.38, 0], [5.3, 0.87, 0], stroke_color=ACCENT, stroke_width=3)
        rp_lab = label("replay", color=ACCENT).move_to([5.95, 1.62, 0])
        self.play(GrowArrow(b_ar), n_pm[0].animate.set_stroke(ACCENT, width=2).set_fill(ACTIVE_FILL), Create(rp_line), FadeIn(rp_lab), run_time=0.5)
        self.wait(3.5)

        switch("RULE", "切線就是記憶體帳單", "切線往 forward 靠是多存、往 backward 靠是多算：min-cut 自動找折衷，checkpoint 是推到極端")
        self.wait(5.5)
