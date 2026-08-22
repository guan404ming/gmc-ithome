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


def rows(lines, size=12, color=TXT, buff=0.12):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


class AOT(Scene):
    def construct(self):
        title = T("AOTAutograd  ·  (x @ w).relu().sum()", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        title[:11].set_color(ACCENT)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        dyn = titled(3.7, 3.0, "DYNAMO", "torch 層的圖").move_to([-5.05, -0.35, 0])
        dyn_lbl = label("INPUT  ·  forward only").next_to(dyn, UP, buff=0.22).align_to(dyn, LEFT)
        db = rows(["matmul = x @ w", "relu = matmul.relu()", "sum_1 = relu.sum()", "return (sum_1,)"], size=12, buff=0.16)
        db.next_to(dyn[1], DOWN, buff=0.35).align_to(dyn[1], LEFT)
        db_note = T("backward？沒有", font=CJK, font_size=12, color=MUTED).next_to(db, DOWN, buff=0.3).align_to(db, LEFT)
        switch("SETUP", "交接", "Dynamo 交出 torch 層的 forward 圖：寫什麼是什麼，但只有前進的路線")
        self.play(FadeIn(dyn), FadeIn(dyn_lbl), FadeIn(db, shift=RIGHT * 0.1), FadeIn(db_note, shift=RIGHT * 0.1), run_time=0.5)
        self.wait(2.5)

        joint = titled(4.3, 4.3, "JOINT GRAPH", "沙盤推演").move_to([-0.6, -0.35, 0])
        joint_lbl = label("TRACE  ·  FakeTensor").next_to(joint, UP, buff=0.22).align_to(joint, LEFT)
        a1 = arrow(dyn[0].get_right(), joint[0].get_left(), color=ACCENT)
        jx = joint[0].get_left()[0] + 0.3
        jy0 = joint[0].get_top()[1] - 0.95

        def place(m, i):
            return m.move_to([jx, jy0 - i * 0.325, 0], aligned_edge=LEFT)

        r_mm = place(T("mm(primals_1, primals_2)", font=MONO, font_size=11, color=TXT), 0)
        r_relu = place(T("relu(mm)", font=MONO, font_size=11, color=TXT), 1)
        r_sum = place(T("sum(relu)", font=MONO, font_size=11, color=TXT), 2)
        r_tan = place(T("tangents_1: f32[]", font=MONO, font_size=11, color=MUTED), 3)
        r_exp = place(T("expand(tangents_1, [4,4])", font=MONO, font_size=11, color=ACCENT), 4)
        r_le = place(T("le(relu, 0)", font=MONO, font_size=11, color=ACCENT), 5)
        r_whr = place(T("where(le, 0.0, expand)", font=MONO, font_size=11, color=ACCENT), 6)
        r_perm = place(T("permute(primals_1, [1,0])", font=MONO, font_size=11, color=ACCENT), 7)
        r_mm1 = place(T("mm(permute, where)", font=MONO, font_size=11, color=ACCENT), 8)

        switch("TRACE", "FakeTensor 重演", "AOTAutograd 拿 FakeTensor 把 forward 重跑一遍：每個 op 經過 dispatcher，落成 ATen 名字")
        self.play(FadeIn(joint), FadeIn(joint_lbl), GrowArrow(a1), run_time=0.5)
        for r in (r_mm, r_relu, r_sum):
            self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.25)
        self.wait(3.5)

        switch("UNFOLD", "autograd 引擎回放", "對推演的輸出呼叫反向傳播：sum 展開成 expand，relu 展開成 le + where，mm 展開成 permute + mm")
        for r in (r_tan, r_exp, r_le, r_whr, r_perm, r_mm1):
            self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=0.25)
        self.wait(3.5)

        switch("PARTITION", "切一刀", "partitioner 決定 le、permute 算在 forward 側存下來：存 b8 的 mask，比存 f32 的 relu 輸出省")
        self.play(r_exp.animate.set_color(TXT), r_whr.animate.set_color(TXT), r_mm1.animate.set_color(TXT), run_time=0.3)
        self.play(place(r_le.animate, 3), place(r_perm.animate, 4), place(r_tan.animate, 5), place(r_exp.animate, 6), place(r_whr.animate, 7), run_time=0.7)
        tag_x = jx + 3.0
        tag1 = T("saved", font=MONO, font_size=10, color=ACCENT).move_to([tag_x, jy0 - 3 * 0.325, 0], aligned_edge=LEFT)
        tag2 = T("saved", font=MONO, font_size=10, color=ACCENT).move_to([tag_x, jy0 - 4 * 0.325, 0], aligned_edge=LEFT)
        ycut = jy0 - 4.5 * 0.325
        cut = DashedLine([jx - 0.05, ycut, 0], [joint[0].get_right()[0] - 0.25, ycut, 0], color=ACCENT, stroke_width=1.5, dash_length=0.08)
        cut_lbl = T("cut", font=MONO, font_size=10, color=ACCENT).next_to(cut, DOWN, buff=0.06).align_to(cut, RIGHT)
        self.play(FadeIn(tag1), FadeIn(tag2), Create(cut), FadeIn(cut_lbl), run_time=0.5)
        self.wait(3.5)

        mid = titled(2.5, 1.6, "AOT", "trace + cut").move_to([-0.6, -0.35, 0])
        fw = titled(4.4, 2.3, "FORWARD", "ATen 層", fill=ACTIVE_FILL, edge=ACCENT, sw=2).move_to([4.35, 1.45, 0])
        bw = titled(4.4, 2.3, "BACKWARD", "ATen 層", fill=ACTIVE_FILL, edge=ACCENT, sw=2).move_to([4.35, -1.45, 0])
        out_lbl = label("OUTPUT  ·  兩張圖").next_to(fw, UP, buff=0.22).align_to(fw, LEFT)
        fw_body = rows(["mm -> relu -> sum_1", "le, permute", "return (sum_1, le, permute)"], size=12, buff=0.12).next_to(fw[1], DOWN, buff=0.3).align_to(fw[1], LEFT)
        fw_body[1].set_color(ACCENT)
        bw_body = rows(["in: le, permute, tangents_1", "expand -> where -> mm_1", "return (None, mm_1)"], size=12, buff=0.12).next_to(bw[1], DOWN, buff=0.3).align_to(bw[1], LEFT)
        bw_body[0].set_color(ACCENT)
        fgrp = VGroup(r_mm, r_relu, r_sum, r_le, r_perm)
        bgrp = VGroup(r_tan, r_exp, r_whr, r_mm1)
        na1 = arrow(dyn[0].get_right(), mid[0].get_left(), color=ACCENT)
        a2 = arrow(mid[0].get_right(), fw[0].get_left(), color=ACCENT)
        a3 = arrow(mid[0].get_right(), bw[0].get_left(), color=ACCENT)
        switch("SPLIT", "一變二", "joint graph 沿著切口分裂：橫跨切口的 le、permute 變成 forward 的額外輸出、backward 的輸入")
        self.play(FadeOut(tag1), FadeOut(tag2), FadeOut(cut), FadeOut(cut_lbl), FadeOut(joint_lbl), run_time=0.3)
        self.play(Transform(joint, mid), Transform(a1, na1), FadeIn(fw), FadeIn(bw), FadeIn(out_lbl), Transform(fgrp, fw_body), Transform(bgrp, bw_body), GrowArrow(a2), GrowArrow(a3), run_time=0.9)
        self.add(fgrp, bgrp)
        sv = arrow(fw[0].get_bottom(), bw[0].get_top(), color=ACCENT, w=2.5)
        sv_lbl = T("saved: le, permute", font=MONO, font_size=12, color=ACCENT).next_to(sv, RIGHT, buff=0.15)
        self.play(GrowArrow(sv), FadeIn(sv_lbl), run_time=0.4)
        self.wait(3.5)

        switch("RUNTIME", "掛回 tape", "兩張圖各自交給 Inductor，再包成一個 autograd.Function：forward 存 le、permute，backward 取出來用")
        self.play(Indicate(sv_lbl, color=ACCENT, scale_factor=1.08), run_time=0.6)
        self.wait(3)

        switch("RULE", "先推演，再切分", "微分是 autograd 引擎跑出來、AOT 錄下來的；存什麼、拿什麼，由 partitioner 全局決定")
        self.wait(5.5)
