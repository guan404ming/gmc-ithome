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


def mixed(s, size, color=TXT, font=MONO):
    t2f = {ch: CJK for ch in s if "一" <= ch <= "鿿"}
    return T(s, font=font, font_size=size, color=color, t2f=t2f)


def card(tag, lines, w=6.9, tag_color=ACCENT, size=16):
    tg = mixed(tag, 16, color=tag_color)
    body = VGroup(*[mixed(s, size) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    inner = VGroup(tg, body).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
    box = RoundedRectangle(corner_radius=0.16, width=w, height=inner.height + 0.6, stroke_color=EDGE, stroke_width=1.6, fill_color=CARD, fill_opacity=1)
    inner.move_to(box).align_to(box.get_left() + RIGHT * 0.4, LEFT)
    g = VGroup(box, inner)
    g.set_z_index(2)
    return g


def chip(s, edge=EDGE, color=TXT, fill=CARD_DIM):
    t = mixed(s, 16, color=color)
    r = RoundedRectangle(corner_radius=0.24, width=t.width + 0.55, height=0.58, stroke_color=edge, stroke_width=1.5, fill_color=fill, fill_opacity=1)
    return VGroup(r, t.move_to(r))


class Backend(Scene):
    def construct(self):
        title = T("torch.compile(f, backend=my_backend)", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = mixed(caption, 19).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("PIPELINE", "三站一線", "Dynamo 抓圖、AOTAutograd 展開，第三站把圖變成可執行的成品")
        st = VGroup(chip("Dynamo"), chip("AOTAutograd"), chip("Inductor", edge=ACCENT)).arrange(RIGHT, buff=1.0).move_to([0, 2.15, 0])
        rails = VGroup(*[Arrow(st[i].get_right(), st[i + 1].get_left(), buff=0.1, color=MUTED, stroke_width=2.2, tip_length=0.14) for i in range(2)])
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.15) for m in (st[0], rails[0], st[1], rails[1], st[2])], lag_ratio=0.15), run_time=1.0)
        self.play(Indicate(st[2], scale_factor=1.06, color=ACCENT), run_time=0.6)
        self.wait(3.5)

        switch("SWAP", "拔掉插上", "第三站可以換，拔下 Inductor 的廠房，插上自己的小工作坊")
        mine = chip("my_backend", edge=ACCENT, fill=ACTIVE_FILL).move_to(st[2]).shift(DOWN * 1.1)
        mine.set_opacity(0)
        self.play(st[2].animate.shift(UP * 1.1).set_opacity(0), mine.animate.shift(UP * 1.1).set_opacity(1), run_time=0.8)
        shop = RoundedRectangle(corner_radius=0.2, width=8.6, height=4.3, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD_DIM, fill_opacity=1).move_to([0, -0.95, 0])
        hdr = mixed("my_backend · 工作坊", 17, color=MUTED).move_to(shop.get_corner(UL) + RIGHT * 0.35 + DOWN * 0.32, aligned_edge=UL)
        drop = DashedLine(mine.get_bottom(), shop.get_top(), dash_length=0.1, color=ACCENT, stroke_width=2)
        self.play(FadeIn(shop), FadeIn(hdr), Create(drop), run_time=0.6)
        self.wait(2.5)

        switch("INPUT", "原料進廠", "一張 FX Graph 流進來，附上 example inputs，這是契約的前半")
        g = card("GraphModule · from Dynamo", [
            "%x   : placeholder",
            "%y   : call_function[target=torch.relu]",
            "%out : call_function[target=add(%y, 1)]",
            "return (%out,)",
        ])
        inp = chip("example inputs · (4, 8) f32")
        pack = VGroup(g, inp).arrange(DOWN, buff=0.3).move_to([0, -1.35, 0])
        pack.shift(LEFT * 12)
        self.add(pack)
        self.play(pack.animate.shift(RIGHT * 12), run_time=0.9)
        self.wait(2.5)

        switch("INSPECT", "逐節點檢視", "GraphModule 是普通的資料結構，走訪每個 node，看 op、看 target")
        lines = g[1][1]
        for i in range(4):
            self.play(lines[i].animate.set_color(ACCENT), run_time=0.25)
            self.play(lines[i].animate.set_color(TXT), run_time=0.2)
        self.wait(2.5)

        switch("REWRITE", "換掉一顆 op", "node.target 改掉再 recompile，這張圖裡的 relu 從此變 sigmoid")
        new_line = mixed("%y   : call_function[target=torch.sigmoid]", 16, color=ACCENT).move_to(lines[1], aligned_edge=LEFT)
        self.play(lines[1].animate.set_color(ACCENT), run_time=0.3)
        self.play(Transform(lines[1], new_line), Flash(lines[1].get_center(), color=ACCENT, line_length=0.18, flash_radius=0.9), run_time=0.7)
        self.wait(3.0)

        switch("OUTPUT", "交回 callable", "回傳一個 callable，契約的後半，水流繼續往下走")
        out = chip("callable · gm.forward", edge=ACCENT, fill=ACTIVE_FILL).move_to([0, -1.35, 0])
        self.play(ReplacementTransform(pack, out), run_time=0.7)
        self.play(out.animate.move_to([5.15, -0.95, 0]), run_time=0.7)
        self.play(Indicate(out, scale_factor=1.05, color=ACCENT), run_time=0.5)
        self.wait(2.5)

        switch("RULE", "契約就一句話", "收 GraphModule 與 example inputs，回一個 callable，上游 Guard 與快取照常")
        self.play(FadeOut(shop), FadeOut(hdr), FadeOut(drop), FadeOut(st[0]), FadeOut(st[1]), FadeOut(mine), FadeOut(rails), FadeOut(out), run_time=0.4)
        row = VGroup(chip("FX Graph"), chip("你的工作坊", edge=ACCENT, fill=ACTIVE_FILL), chip("callable")).arrange(RIGHT, buff=1.05).move_to([0, 0.25, 0])
        arrows = VGroup()
        labels = VGroup()
        for i, name in enumerate(("IN", "OUT")):
            a2 = Arrow(row[i].get_right(), row[i + 1].get_left(), buff=0.1, color=ACCENT, stroke_width=2.5, tip_length=0.16)
            arrows.add(a2)
            labels.add(T(name, font=MONO, font_size=16, color=MUTED).next_to(a2, DOWN, buff=0.22))
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.15) for m in (row[0], arrows[0], row[1], arrows[1], row[2])], lag_ratio=0.15), run_time=1.2)
        self.play(FadeIn(labels), run_time=0.4)
        self.wait(5.5)
