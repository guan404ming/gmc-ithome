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


def chip(label, color=TXT, fill=CARD_DIM):
    t = T(label, font=MONO, font_size=16, color=color)
    bg = RoundedRectangle(corner_radius=0.12, width=t.width + 0.36, height=0.44, stroke_color=EDGE, stroke_width=1.2, fill_color=fill, fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def stage(name, w=1.85):
    t = T(name, font=SANS, font_size=16, weight=BOLD, color=MUTED)
    box = RoundedRectangle(corner_radius=0.14, width=w, height=0.66, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    return VGroup(box, t.move_to(box))


def key_icon(teeth, color=TXT):
    bow = Circle(radius=0.15, stroke_color=color, stroke_width=3.2)
    shaft = Line(ORIGIN, RIGHT * 0.95, stroke_color=color, stroke_width=3.2)
    shaft.next_to(bow, RIGHT, buff=0)
    tg = VGroup()
    for i, h in enumerate(teeth):
        x = shaft.get_start()[0] + 0.42 + i * 0.18
        tg.add(Line([x, 0, 0], [x, -0.1 - 0.08 * h, 0], stroke_color=color, stroke_width=3.2))
    return VGroup(bow, shaft, tg)


def cell(w=2.5, h=1.15):
    return RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1)


STAGE_Y = 1.15
CAB_X = 4.6


class Cache(Scene):
    def construct(self):
        title = T("f(x, y) -> gelu(x @ y + 1).sum()", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("COLD", "冷編譯", "快取是空的，(512, 512) 的請求走完整條產線，每一站都是真金白銀的時間")
        stages = VGroup(stage("Dynamo"), stage("AOTAutograd", w=2.15), stage("Inductor")).arrange(RIGHT, buff=0.55).move_to([-3.2, STAGE_Y, 0])
        arrows = VGroup(*[Line(stages[i].get_right(), stages[i + 1].get_left(), stroke_color=DIM, stroke_width=2.4) for i in range(2)])
        cab_label = T("FXGraphCache", font=MONO, font_size=16, color=MUTED)
        cells = VGroup(cell(), cell()).arrange(DOWN, buff=0.3)
        cab = VGroup(cab_label, cells).arrange(DOWN, buff=0.25).move_to([CAB_X, 0.55, 0])
        tlabel = T("first call", font=MONO, font_size=16, color=MUTED)
        tval = T("...", font=MONO, font_size=22, color=MUTED)
        timer = VGroup(tlabel, tval).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(stages, shift=UP * 0.15), FadeIn(arrows), FadeIn(cab, shift=UP * 0.15), FadeIn(timer), run_time=0.9)
        req = chip("(512, 512) · f32").next_to(stages, UP, buff=0.55).align_to(stages[0], LEFT)
        self.play(FadeIn(req, shift=RIGHT * 0.3), run_time=0.5)
        for s in stages:
            self.play(s[0].animate.set_stroke(ACCENT, 2.2), s[1].animate.set_color(TXT), run_time=0.45)
            self.play(s[0].animate.set_stroke(EDGE, 1.5), run_time=0.25)
        art = chip("kernel.so", fill=CARD).next_to(stages[2], DOWN, buff=0.55)
        tval1 = T("3.75 s", font=MONO, font_size=22, color=ACCENT).move_to(tval, aligned_edge=LEFT)
        self.play(FadeIn(art, shift=DOWN * 0.2), Transform(tval, tval1), run_time=0.6)
        self.wait(3.0)

        switch("KEY", "打一把鑰匙", "圖、shape、dtype、config、torch 版本五塊碎片熔成一把鑰匙，缺一不可")
        frags = VGroup(chip("graph"), chip("(512, 512)"), chip("f32"), chip("config"), chip("v2.8.0")).arrange(RIGHT, buff=0.3).move_to([-2.6, -1.15, 0])
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.15) for f in frags], lag_ratio=0.12), run_time=1.0)
        k1 = key_icon((1, 2, 1), ACCENT).move_to([-2.6, -1.15, 0])
        k1tag = T("fih5y2v3...", font=MONO, font_size=16, color=MUTED).next_to(k1, DOWN, buff=0.25)
        self.wait(1.2)
        self.play(ReplacementTransform(frags, k1), run_time=0.9)
        self.play(FadeIn(k1tag, shift=UP * 0.1), run_time=0.4)
        self.wait(2.5)

        switch("STORE", "存進置物櫃", "成品掛上鑰匙存進磁碟上的櫃子，這一格從此屬於這把鑰匙")
        tag1 = T("fih5y2v3...", font=MONO, font_size=16, color=MUTED).move_to(cells[0]).shift(DOWN * 0.28)
        art1 = art.copy().scale(0.9).move_to(cells[0]).shift(UP * 0.2)
        self.play(art.animate.scale(0.9).move_to(art1), k1.animate.scale(0.5).move_to(cells[0]).set_opacity(0), ReplacementTransform(k1tag, tag1), run_time=1.0)
        self.remove(k1)
        self.play(cells[0].animate.set_stroke(ACCENT, 2.0), Flash(cells[0], color=ACCENT, line_length=0.14, flash_radius=1.1), run_time=0.6)
        self.play(cells[0].animate.set_stroke(EDGE, 1.5), run_time=0.3)
        self.wait(2.5)

        switch("HIT", "同一把鑰匙", "第二個 process 帶著同樣的材料進來，鑰匙齒形一致，開櫃取貨，整段編譯直接略過")
        req2 = chip("(512, 512) · f32").move_to(req)
        self.play(FadeOut(req), FadeIn(req2, shift=RIGHT * 0.3), stages.animate.set_opacity(0.35), arrows.animate.set_opacity(0.35), run_time=0.6)
        k2 = key_icon((1, 2, 1), ACCENT).scale(0.8).next_to(req2, RIGHT, buff=0.7)
        self.play(FadeIn(k2, scale=0.7), run_time=0.5)
        self.play(k2.animate.move_to(cells[0].get_left() + LEFT * 0.7), run_time=0.8)
        tval2 = T("0.79 s", font=MONO, font_size=22, color=ACCENT).move_to(tval, aligned_edge=LEFT)
        out1 = chip("kernel.so", fill=CARD).next_to(cells[0], LEFT, buff=0.9).shift(DOWN * 0.9)
        self.play(cells[0].animate.set_stroke(ACCENT, 2.2), Flash(cells[0], color=ACCENT, line_length=0.14, flash_radius=1.1), run_time=0.5)
        self.play(FadeOut(k2), TransformFromCopy(art1, out1), Transform(tval, tval2), run_time=0.8)
        self.play(Flash(tval, color=ACCENT, line_length=0.12, flash_radius=0.5), run_time=0.4)
        self.wait(3.5)

        switch("MISS", "齒形變了", "shape 換成 (768, 768)，一塊碎片變形，鑰匙插不進鎖孔，這格櫃子不認帳")
        req3 = chip("(768, 768) · f32", color=ACCENT)
        req3.move_to(req2)
        self.play(FadeOut(req2), FadeOut(out1), cells[0].animate.set_stroke(EDGE, 1.5), FadeIn(req3, shift=RIGHT * 0.3), run_time=0.6)
        k3 = key_icon((3, 1, 2), TXT).scale(0.8).next_to(req3, RIGHT, buff=0.7)
        k3[2].set_stroke(ACCENT, 3.2)
        self.play(FadeIn(k3, scale=0.7), run_time=0.5)
        self.play(k3.animate.move_to(cells[0].get_left() + LEFT * 0.7), run_time=0.8)
        cross = VGroup(Line(UL * 0.18, DR * 0.18), Line(UR * 0.18, DL * 0.18)).set_stroke(ACCENT, 4).move_to(cells[0].get_left() + LEFT * 0.15)
        self.play(FadeIn(cross, scale=1.4), Flash(cells[0], color=ACCENT, line_length=0.14, flash_radius=1.1), run_time=0.5)
        self.play(k3.animate.shift(LEFT * 0.35), rate_func=there_and_back, run_time=0.4)
        self.wait(2.0)

        switch("SLOW PATH", "重走慢路", "miss 不是錯誤，只是重付一次編譯費，新成品用新鑰匙住進另一格")
        self.play(FadeOut(cross), FadeOut(k3), stages.animate.set_opacity(1), arrows.animate.set_opacity(1), run_time=0.5)
        for s in stages:
            self.play(s[0].animate.set_stroke(ACCENT, 2.2), run_time=0.3)
            self.play(s[0].animate.set_stroke(EDGE, 1.5), run_time=0.18)
        tval3 = T("3.12 s", font=MONO, font_size=22, color=ACCENT).move_to(tval, aligned_edge=LEFT)
        art2 = chip("kernel.so", fill=CARD).scale(0.9).move_to(cells[1]).shift(UP * 0.2)
        tag2 = T("fcybv4xa...", font=MONO, font_size=16, color=MUTED).move_to(cells[1]).shift(DOWN * 0.28)
        self.play(Transform(tval, tval3), FadeIn(art2, shift=DOWN * 0.15), FadeIn(tag2), run_time=0.8)
        self.play(cells[1].animate.set_stroke(ACCENT, 2.0), Flash(cells[1], color=ACCENT, line_length=0.14, flash_radius=1.1), run_time=0.6)
        self.play(cells[1].animate.set_stroke(EDGE, 1.5), run_time=0.3)
        self.wait(3.0)

        switch("RESULT", "各住一格", "512 和 768 的成品並存，各自等各自的下一次 hit，編譯費每種形狀只付一次")
        rects = VGroup(cells[0].copy(), cells[1].copy()).set_fill(opacity=0)
        rects.set_stroke(ACCENT, 2.0)
        h1 = T("hit  -> 0.79 s", font=MONO, font_size=18, color=TXT)
        h2 = T("miss -> 3.12 s", font=MONO, font_size=18, color=MUTED)
        summary = VGroup(h1, h2).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([-3.2, -1.2, 0])
        self.play(Create(rects[0]), Create(rects[1]), FadeIn(summary, shift=UP * 0.1), run_time=0.9)
        self.play(Indicate(h1, scale_factor=1.1, color=ACCENT), run_time=0.6)
        self.wait(5.5)
