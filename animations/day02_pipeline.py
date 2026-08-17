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
EDGE_DIM = "#262a30"
TXT = "#e8e6e3"
MUTED = "#8b8f96"
ACCENT = "#e8622a"
config.background_color = BG
MONO = "Menlo"
SANS = "TASA Orbiter"
CJK = "PingFang TC"


def T(txt, font_size, **kw):
    return Text(txt, font_size=font_size * 4, **kw).scale(0.25)


def label(s, size=15, color=MUTED):
    return T(s, font=MONO, font_size=size, color=color)


def card(title, sub, w=3.05, h=1.7):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    t = T(title, font=SANS, font_size=28, weight=BOLD, color=TXT).move_to(r.get_center() + UP * 0.28)
    s = T(sub, font=CJK, font_size=18, color=MUTED).move_to(r.get_center() + DOWN * 0.38)
    return VGroup(r, t, s)


def ghost(w=3.05, h=1.7):
    return RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=EDGE_DIM, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1)


def artifact(lines, w=3.05):
    t = T("\n".join(lines), font=MONO, font_size=16, color=TXT, line_spacing=0.9)
    r = RoundedRectangle(corner_radius=0.1, width=max(w, t.width + 0.5), height=t.height + 0.6, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    t.move_to(r.get_center())
    return VGroup(r, t)


def pill(name, zh):
    t = VGroup(Dot(radius=0.06, color=ACCENT), T(name, font=SANS, font_size=18, weight=BOLD, color=TXT), T("·", font=MONO, font_size=16, color=MUTED), T(zh, font=CJK, font_size=16, color=TXT)).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    t[1].set_color(BG); t[3].set_color(BG); t[2].set_color("#666")
    return VGroup(bg, t.move_to(bg))


class Pipeline(Scene):
    def construct(self):
        title = T("model = torch.compile(model)", font=MONO, font_size=26, color=TXT).to_corner(UL, buff=0.5)
        title[6:19].set_color(ACCENT)
        self.play(FadeIn(title), run_time=0.5)

        names = [("Dynamo", "擷取"), ("AOTAutograd", "展開、正規化"), ("Inductor", "生成程式碼"), ("Runtime", "執行")]
        cards = VGroup(*[card(n, z) for n, z in names]).arrange(RIGHT, buff=0.45).move_to(DOWN * 1.4)
        ghosts = VGroup(*[ghost().move_to(c) for c in cards])
        sec = label("PIPELINE  ·  流水線").move_to(UP * -0.1).align_to(cards, LEFT)
        sec2 = label("ARTIFACT  ·  中間產物").move_to(UP * 2.5).align_to(cards, LEFT)
        ART_TOP = 2.05
        self.play(FadeIn(ghosts), FadeIn(sec), FadeIn(sec2), run_time=0.5)

        src = artifact(["def f(x):", "  return sin(x)*cos(x)", "         + tanh(x)"]).align_to(ghosts[0], LEFT)
        src.shift(UP * (ART_TOP - src.get_top()[1]))
        self.play(FadeIn(src, shift=UP * 0.15), run_time=0.5)

        arts = [
            artifact(["FX Graph", "sin -> mul -> add", "+ Guards"]),
            artifact(["forward graph", "backward graph", "decomposed ops"]),
            artifact(["@triton.jit", "fused_kernel(...)", "1 launch, 1 pass"]),
            artifact(["cached kernel", "CUDA graph", "run"]),
        ]
        captions = [
            "攔截 Python bytecode，錄成 FX Graph，裝上 Guard",
            "把 backpropagation 一起 trace 出來，正規化、拆解成基本 op",
            "決定誰跟誰融合，生成 Triton（GPU）或 C++（CPU）kernel",
            "載入編好的 kernel、管理快取，必要時掛 CUDA Graph",
        ]
        cur_pill = None
        cur_cap = None
        for i, c in enumerate(cards):
            cap = T(captions[i], font=CJK, font_size=20, color=TXT).to_edge(DOWN, buff=0.55)
            p = pill(names[i][0].upper(), names[i][1]).to_edge(RIGHT, buff=0.5).match_y(title)
            anims = [FadeIn(c), c[0].animate.set_stroke(ACCENT, width=2.5).set_fill("#2b2622")]
            anims.append(FadeIn(p) if cur_pill is None else AnimationGroup(FadeOut(cur_pill), FadeIn(p)))
            anims.append(FadeIn(cap, shift=UP * 0.1) if cur_cap is None else AnimationGroup(FadeOut(cur_cap), FadeIn(cap, shift=UP * 0.1)))
            cur_pill = p
            cur_cap = cap
            self.play(*anims, run_time=0.5)
            if i > 0:
                arrow = Arrow(cards[i - 1].get_right(), c.get_left(), buff=0.06, color=MUTED, stroke_width=2, max_tip_length_to_length_ratio=0.3)
                self.play(GrowArrow(arrow), run_time=0.3)
            a = arts[i].move_to(c)
            a.shift(UP * (ART_TOP - a.get_top()[1]))
            a[0].set_stroke(ACCENT, width=2)
            if i == 0:
                self.play(FadeOut(src, shift=UP * 0.15), FadeIn(a, shift=UP * 0.15), run_time=0.5)
            else:
                self.play(FadeIn(a, shift=UP * 0.15), run_time=0.5)
            self.wait(1.25)
            self.play(c[0].animate.set_stroke(EDGE, width=1.5).set_fill(CARD), a[0].animate.set_stroke(EDGE, width=1.5), run_time=0.3)
        self.wait(3)
