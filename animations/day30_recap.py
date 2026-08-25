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
EDGE_DIM = "#262a30"
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


def mixed(s, size, color=TXT, font=MONO):
    t2f = {ch: CJK for ch in s if "一" <= ch <= "鿿"}
    return T(s, font=font, font_size=size, color=color, t2f=t2f)


def station(name, zh, keys, days):
    r = RoundedRectangle(corner_radius=0.14, width=3.05, height=3.0, stroke_color=EDGE_DIM, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1)
    nm = T(name, font=SANS, font_size=18, weight=BOLD, color=DIM)
    sub = T(zh, font=CJK, font_size=16, color=DIM)
    hd = VGroup(nm, sub).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
    hd.move_to(r.get_top() + DOWN * 0.42)
    dy = T(days, font=MONO, font_size=16, color=DIM)
    dy.next_to(hd, DOWN, buff=0.12)
    rows = VGroup()
    for k in keys:
        d = Dot(radius=0.045, color=ACCENT)
        t = T(k, font=MONO, font_size=16, color=TXT)
        rows.add(VGroup(d, t).arrange(RIGHT, buff=0.16))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    rows.move_to(r.get_center() + DOWN * 0.42).align_to(r.get_left() + RIGHT * 0.35, LEFT)
    rows.set_opacity(0)
    return VGroup(r, nm, sub, dy, rows)


class Recap(Scene):
    def construct(self):
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_corner(UR, buff=0.5)
            c = mixed(caption, 19).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        line = T("model = torch.compile(model)", font=MONO, font_size=26, color=TXT)
        line[6:19].set_color(ACCENT)
        line.move_to(UP * 3.6)
        self.add(line)
        switch("DAY 30", "回到起點", "30 天前落下的這一行，今天沿著它，把整條管線再走一遍")
        self.play(line.animate(rate_func=rate_functions.ease_out_sine).move_to(UP * 0.4), run_time=0.9)
        self.play(Flash(line.get_center(), color=ACCENT, line_length=0.2, flash_radius=1.6), run_time=0.5)
        self.wait(2.5)

        title = T("model = torch.compile(model)", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        title[6:19].set_color(ACCENT)
        self.play(ReplacementTransform(line, title), run_time=0.6)

        specs = [
            ("Dynamo", "擷取", ["eval hook", "Guard", "graph break", "sym shapes"], "Day 3 - 11"),
            ("AOTAutograd", "展開", ["joint graph", "functionalize", "decompose", "min-cut"], "Day 12 - 16"),
            ("Inductor", "生成", ["loop IR", "fusion", "Triton / C++", "autotune"], "Day 17 - 24"),
            ("Runtime", "執行", ["cache", "CUDA Graph", "replay"], "Day 25 - 29"),
        ]
        cards = VGroup(*[station(*s) for s in specs]).arrange(RIGHT, buff=0.45).move_to(DOWN * 0.75)
        self.play(FadeIn(cards, shift=UP * 0.15), run_time=0.6)

        arrows = []

        def light(i):
            c = cards[i]
            anims = [
                c[0].animate.set_stroke(ACCENT, width=2.5).set_fill(ACTIVE_FILL),
                c[1].animate.set_color(TXT),
                c[2].animate.set_color(MUTED),
                c[3].animate.set_color(MUTED),
            ]
            if i > 0:
                ar = Arrow(cards[i - 1][0].get_right(), c[0].get_left(), buff=0.06, color=ACCENT, stroke_width=2.2, max_tip_length_to_length_ratio=0.3)
                arrows.append(ar)
                anims.append(GrowArrow(ar))
            self.play(*anims, run_time=0.5)
            self.play(LaggedStart(*[r.animate.set_opacity(1) for r in c[4]], lag_ratio=0.25), run_time=0.8)

        def dim(i):
            self.play(cards[i][0].animate.set_stroke(EDGE, width=1.5).set_fill(CARD), run_time=0.25)
            if i > 0:
                self.play(arrows[i - 1].animate.set_color(MUTED), run_time=0.2)

        switch("DYNAMO", "第一站，攔截", "eval hook 攔下 bytecode，Guard 記住前提，斷了就切圖，shape 可以是符號")
        light(0)
        self.wait(3.5)
        dim(0)

        switch("AOTAUTOGRAD", "第二站，推演", "forward 和 backward 一起 trace，in-place 消失，min-cut 決定存誰、重算誰")
        light(1)
        self.wait(3.5)
        dim(1)

        switch("INDUCTOR", "第三站，鑄造", "圖攤成迴圈，fusion 省流量，生出 Triton 與 C++，碼表挑冠軍")
        light(2)
        self.wait(3.5)
        dim(2)

        switch("RUNTIME", "第四站，執行", "快取讓編譯費只付一次，CUDA Graph 把整串 launch 收成一次 replay")
        light(3)
        self.wait(3.5)
        dim(3)

        switch("PIPELINE", "全線點亮", "四站接起來，就是那一行背後的完整旅程")
        self.play(
            *[c[0].animate.set_stroke(ACCENT, width=2.2).set_fill(ACTIVE_FILL) for c in cards],
            *[a.animate.set_color(ACCENT) for a in arrows],
            run_time=0.7,
        )
        self.wait(2.5)

        switch("FIN", "收回一行", "從 bytecode 到 kernel 的一切，都收在這一行裡")
        rig = VGroup(cards, *arrows)
        final = T("model = torch.compile(model)", font=MONO, font_size=28, color=TXT).move_to(UP * 0.3)
        final[6:19].set_color(ACCENT)
        self.play(rig.animate.scale(0.05).move_to(UP * 0.3).set_opacity(0), FadeOut(title), run_time=0.9)
        self.remove(rig)
        self.play(FadeIn(final, scale=1.1), Flash(final.get_center(), color=ACCENT, line_length=0.2, flash_radius=1.8), run_time=0.7)
        tag = T("一行背後的 30 天", font=CJK, font_size=20, color=MUTED).next_to(final, DOWN, buff=0.5)
        self.play(FadeIn(tag, shift=UP * 0.1), run_time=0.5)
        self.wait(5.5)
