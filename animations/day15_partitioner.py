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
    t = T(name, font=SANS, font_size=20, weight=BOLD, color=TXT)
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


def rows(lines, size=12, color=TXT, buff=0.12):
    items = []
    for l in lines:
        if l == "":
            items.append(Rectangle(width=0.01, height=size * 4 / 100 * 0.25, stroke_width=0, fill_opacity=0))
        else:
            items.append(T(l, font=MONO, font_size=size, color=color))
    return VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


def node(txt, w=1.5, accent=False):
    r = panel(w, 0.5, fill=ACTIVE_FILL if accent else CARD_DIM, edge=ACCENT if accent else EDGE, r=0.08, sw=2 if accent else 1.5)
    t = T(txt, font=MONO, font_size=12, color=TXT).move_to(r)
    return VGroup(r, t)


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

        joint = titled(13.2, 3.7, "JOINT GRAPH", "forward 和 backward 先畫成一張").move_to([0, 0.8, 0])
        jlbl = label("JOINT  ·  aot_joint_graph").next_to(joint, UP, buff=0.22).align_to(joint, LEFT)
        switch("SETUP", "共同財產清冊", "AOTAutograd 先把 forward 和 backward trace 成同一張 joint graph，中間值是兩邊共同持有的家當")
        self.play(FadeIn(joint), FadeIn(jlbl), run_time=0.5)

        Y1, Y2 = 1.55, -0.25
        n_pr = node("primals", w=1.5).move_to([-5.2, Y1, 0])
        n_mm = node("mm", w=1.1).move_to([-3.3, Y1, 0])
        n_th = node("tanh", w=1.2).move_to([-1.55, Y1, 0])
        n_sm = node("sum_1", w=1.3).move_to([-0.1, Y1, 0])
        n_d = node("1 - tanh^2", w=1.9).move_to([1.6, Y2, 0])
        n_ml = node("mul_1", w=1.2).move_to([3.35, Y2, 0])
        n_gw = node("mm_1 = grad_w", w=2.0).move_to([5.2, Y2, 0])
        fw_tag = label("forward", size=12).move_to([-5.2, 0.95, 0])
        bw_tag = label("backward", size=12).move_to([5.2, 0.45, 0])
        a1 = arrow(n_pr[0].get_right(), n_mm[0].get_left())
        a2 = arrow(n_mm[0].get_right(), n_th[0].get_left())
        a3 = arrow(n_th[0].get_right(), n_sm[0].get_left())
        a_x = arrow(n_th[0].get_bottom(), n_d[0].get_left())
        a4 = arrow(n_d[0].get_right(), n_ml[0].get_left())
        a5 = arrow(n_ml[0].get_right(), n_gw[0].get_left())
        self.play(FadeIn(n_pr), GrowArrow(a1), FadeIn(n_mm), GrowArrow(a2), FadeIn(n_th), GrowArrow(a3), FadeIn(n_sm), FadeIn(fw_tag), run_time=0.9)
        self.play(GrowArrow(a_x), FadeIn(n_d), GrowArrow(a4), FadeIn(n_ml), GrowArrow(a5), FadeIn(n_gw), FadeIn(bw_tag), run_time=0.9)
        self.wait(3.5)

        switch("NEED", "backward 要什麼", "backward 靠資料流用到 tanh 和 primals_1：這些值要嘛 forward 存下來、要嘛 backward 自己重算")
        self.play(a_x.animate.set_color(ACCENT), Indicate(n_th, color=ACCENT, scale_factor=1.08), run_time=0.6)
        self.wait(2.5)

        tag_th = label("16 KB", size=10, color=TXT).next_to(n_th, DOWN, buff=0.12).shift(LEFT * 0.55)
        tag_mm = VGroup(label("16 KB", size=10, color=TXT), label("no recompute", size=9, color=ACCENT)).arrange(DOWN, buff=0.05).next_to(n_mm, DOWN, buff=0.12)
        switch("COST", "邊上有價格", "min-cut 建模：保存一個值的成本是它的 bytes，mm 這種計算密集的 op 禁止重算，容量無限大")
        self.play(FadeIn(tag_th), FadeIn(tag_mm), run_time=0.4)
        self.wait(3.5)

        cutA = DashedLine([1.2, 2.3, 0], [-0.5, -0.9, 0], color=ACCENT, stroke_width=3, dash_length=0.12)
        cutA_l = T("cut A", font=MONO, font_size=13, color=ACCENT, weight=BOLD).next_to(cutA.get_start(), UP, buff=0.08)
        switch("CUT A", "存 tanh", "先試切在 tanh 之後：保存 tanh，跨線的量是 16 KB")
        self.play(Create(cutA), FadeIn(cutA_l), run_time=0.7)
        self.wait(2.5)

        cutB = DashedLine([-2.5, 2.05, 0], [-1.9, -0.85, 0], color=ACCENT, stroke_width=3, dash_length=0.12)
        cutB_l = T("cut B", font=MONO, font_size=13, color=ACCENT, weight=BOLD).move_to([-2.6, -0.6, 0])
        n_re = node("tanh", w=1.2, accent=True).move_to([-0.9, Y2, 0])
        re_tag = label("recompute", size=10, color=ACCENT).next_to(n_re, DOWN, buff=0.1)
        a_sv = arrow(n_mm[0].get_bottom(), n_re[0].get_left(), color=ACCENT, w=2.5)
        a_re = arrow(n_re[0].get_right(), n_d[0].get_left(), color=MUTED)
        switch("CUT B", "刀往輸入推", "存 mm 也是 16 KB，但 tanh 是 pointwise，重算免費還能融合：tanh 複製一份歸隊 backward")
        self.play(Transform(cutA, cutB), Transform(cutA_l, cutB_l), FadeOut(tag_th), FadeOut(tag_mm), run_time=0.7)
        self.play(FadeOut(a_x), TransformFromCopy(n_th, n_re), FadeIn(re_tag), run_time=0.7)
        self.play(GrowArrow(a_sv), GrowArrow(a_re), n_mm[0].animate.set_stroke(ACCENT, width=2).set_fill(ACTIVE_FILL), run_time=0.6)
        self.wait(3.5)

        joint_all = VGroup(joint, jlbl, n_pr, n_mm, n_th, n_sm, n_d, n_ml, n_gw, fw_tag, bw_tag, a1, a2, a3, a4, a5, a_sv, a_re, n_re, re_tag, cutA, cutA_l)
        fw_card = titled(5.6, 3.9, "FORWARD", "只到切線為止").move_to([-3.85, -0.05, 0])
        bw_card = titled(5.6, 3.9, "BACKWARD", "切線之後 + 重算").move_to([3.85, -0.05, 0])
        fw_lbl = label("FW  ·  saves activations").next_to(fw_card, UP, buff=0.22).align_to(fw_card, LEFT)
        bw_lbl = label("BW  ·  recomputes tanh").next_to(bw_card, UP, buff=0.22).align_to(bw_card, LEFT)
        sv_lbl = label("SAVED", size=12).move_to([0, 1.0, 0])
        switch("SPLIT", "節點歸隊", "沿著切線分成兩張圖：跨線的 mm 和 permute 成為 forward 的輸出、backward 的輸入")
        self.play(FadeOut(joint_all), run_time=0.5)
        self.play(FadeIn(fw_card), FadeIn(bw_card), FadeIn(fw_lbl), FadeIn(bw_lbl), FadeIn(sv_lbl), run_time=0.5)
        fw_body = rows(["mm = mm(primals_1, primals_2)", "tanh = tanh(mm)", "sum_1 = sum(tanh)", "permute = permute(primals_1)", "", "return (sum_1, mm, permute)"], size=11, buff=0.14).next_to(fw_card[1], DOWN, buff=0.3).align_to(fw_card[1], LEFT)
        fw_body[5].set_color(ACCENT)
        bw_body = rows(["tanh = tanh(mm)", "mul = tanh * tanh", "sub = 1 - mul", "mul_1 = expand * sub", "mm_1 = mm(permute, mul_1)", "", "return (None, mm_1)"], size=11, buff=0.14).next_to(bw_card[1], DOWN, buff=0.3).align_to(bw_card[1], LEFT)
        bw_body[0].set_color(ACCENT)
        s1 = arrow([-1.05, 0.35, 0], [1.05, 0.35, 0], color=ACCENT, w=2.5)
        s2 = arrow([-1.05, -0.55, 0], [1.05, -0.55, 0], color=ACCENT, w=2.5)
        c1 = label("mm  ·  16 KB", size=11, color=ACCENT).next_to(s1, UP, buff=0.1)
        c2 = label("permute  ·  0 B", size=11, color=ACCENT).next_to(s2, UP, buff=0.1)
        self.play(FadeIn(fw_body, shift=RIGHT * 0.1), run_time=0.4)
        self.play(GrowArrow(s1), FadeIn(c1), GrowArrow(s2), FadeIn(c2), run_time=0.5)
        self.play(FadeIn(bw_body, shift=RIGHT * 0.1), run_time=0.4)
        self.wait(3.5)

        fw_body2 = rows(["mm = mm(primals_1, primals_2)", "tanh = tanh(mm)", "sum_1 = sum(tanh)", "", "return (sum_1, primals_1, primals_2)"], size=11, buff=0.14).next_to(fw_card[1], DOWN, buff=0.3).align_to(fw_card[1], LEFT)
        fw_body2[4].set_color(ACCENT)
        bw_body2 = rows(["mm = mm(primals_1, primals_2)", "tanh = tanh(mm)", "mul = tanh * tanh", "sub = 1 - mul", "mul_1 = expand * sub", "permute = permute(primals_1)", "mm_1 = mm(permute, mul_1)", "", "return (None, mm_1)"], size=11, buff=0.14).next_to(bw_card[1], DOWN, buff=0.3).align_to(bw_card[1], LEFT)
        for i in (0, 1, 5):
            bw_body2[i].set_color(ACCENT)
        c1b = label("primals_1  ·  16 KB", size=11, color=ACCENT).next_to(s1, UP, buff=0.1)
        c2b = label("primals_2  ·  16 KB", size=11, color=ACCENT).next_to(s2, UP, buff=0.1)
        switch("CHECKPOINT", "旋鈕轉到底", "checkpoint 只存 primals，backward 開頭把整段 forward 重播一遍，用計算換記憶體")
        self.play(Transform(fw_body, fw_body2), Transform(bw_body, bw_body2), Transform(c1, c1b), Transform(c2, c2b), run_time=0.8)
        self.wait(3.5)

        switch("RULE", "切線就是記憶體帳單", "切線往 forward 靠是多存、往 backward 靠是多算：min-cut 自動找折衷，checkpoint 是極端值")
        self.wait(5.5)
