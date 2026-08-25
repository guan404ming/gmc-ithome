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


def ball(label):
    t = T(label, font=MONO, font_size=16, color=TXT)
    bg = RoundedRectangle(corner_radius=0.25, width=t.width + 0.5, height=0.52, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1)
    g = VGroup(bg, t.move_to(bg))
    g.set_z_index(3)
    return g


def entry_cell():
    return RoundedRectangle(corner_radius=0.1, width=1.62, height=1.0, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1)


def entry_content(i):
    g = T(f"g{i}", font=MONO, font_size=16, color=TXT)
    gd = T(f"step == {i}", font=MONO, font_size=16, color=MUTED)
    v = VGroup(g, gd).arrange(DOWN, buff=0.14)
    v.set_z_index(1)
    return v


GRID_C = np.array([3.1, 0.75, 0])
LANE_Y = 0.75
BAR_W = 3.4


class Recompile(Scene):
    def construct(self):
        title = T("poly(x) -> ((x * step).sin() + ...).sum()", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            runs = re.findall(r"[一-鿿，、。！？]+|[^一-鿿，、。！？]+", caption)
            cs = [T(r, font=CJK if re.search(r"[一-鿿]", r) else MONO, font_size=19, color=TXT) for r in runs]
            c = VGroup(*cs).arrange(RIGHT, buff=0.12).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        cells = VGroup(*[entry_cell() for _ in range(8)]).arrange_in_grid(rows=2, cols=4, buff=0.28).move_to(GRID_C)
        cab_label = T("code object · cache entries", font=MONO, font_size=16, color=MUTED).next_to(cells, UP, buff=0.3)
        track = RoundedRectangle(corner_radius=0.09, width=BAR_W, height=0.3, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1).move_to([GRID_C[0], -1.55, 0])
        bar_label = T("entries 0 / 8", font=MONO, font_size=16, color=MUTED).next_to(track, UP, buff=0.2)
        comp = RoundedRectangle(corner_radius=0.14, width=2.1, height=0.66, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1).move_to([-3.4, -1.55, 0])
        comp_t = T("COMPILE", font=SANS, font_size=16, weight=BOLD, color=MUTED).move_to(comp)
        comp_cost = T("~0.8 s / graph", font=MONO, font_size=16, color=DIM).next_to(comp, DOWN, buff=0.22)
        bar = [None]
        state = [0]

        def bar_fill(n):
            w = BAR_W * n / 8
            r = Rectangle(width=w, height=0.3, stroke_width=0, fill_color=ACCENT if n >= 8 else MUTED, fill_opacity=0.9)
            r.move_to(track.get_left() + RIGHT * w / 2)
            return r

        def count_to(n):
            state[0] = n
            lbl = T(f"entries {n} / 8", font=MONO, font_size=16, color=ACCENT if n >= 8 else MUTED).move_to(bar_label)
            nb = bar_fill(n)
            if bar[0] is None:
                bar[0] = nb
                return [FadeIn(nb), Transform(bar_label, lbl)]
            return [Transform(bar[0], nb), Transform(bar_label, lbl)]

        switch("CALL 1", "第一次編譯", "step=0 進來，快取還是空的，走一遍完整編譯，成品塞進第一格 entry")
        self.play(FadeIn(cells, shift=UP * 0.15), FadeIn(cab_label), FadeIn(track), FadeIn(bar_label), FadeIn(comp), FadeIn(comp_t), FadeIn(comp_cost), run_time=0.8)
        b0 = ball("step=0").move_to([-8, LANE_Y, 0])
        self.play(b0.animate.move_to([-4.6, LANE_Y, 0]), run_time=0.8)
        self.play(comp.animate.set_stroke(ACCENT, 2.2), comp_t.animate.set_color(TXT), b0.animate.move_to(comp.get_top() + UP * 0.45), run_time=0.7)
        contents = []
        c0 = entry_content(0).move_to(cells[0])
        contents.append(c0)
        self.play(FadeOut(b0), comp.animate.set_stroke(EDGE, 1.5), FadeIn(c0, shift=UP * 0.1), cells[0].animate.set_stroke(ACCENT, 2.0), *count_to(1), run_time=0.9)
        self.play(cells[0].animate.set_stroke(EDGE, 1.5), run_time=0.3)
        self.wait(3.5)

        switch("GUARD FAIL", "驗票失敗", "step 變成 1，g0 的 Guard 接不住，沒人接得住就再編一張，塞進下一格")
        b1 = ball("step=1").move_to([-8, LANE_Y, 0])
        self.play(b1.animate.move_to([-4.6, LANE_Y, 0]), run_time=0.7)
        fail = T("G['step'] == 0", font=MONO, font_size=16, color=ACCENT).next_to(cells[0], LEFT, buff=0.45)
        cross = VGroup(Line(UL * 0.14, DR * 0.14), Line(UR * 0.14, DL * 0.14)).set_stroke(ACCENT, 3.5).move_to(cells[0].get_left() + LEFT * 0.15)
        self.play(cells[0].animate.set_stroke(ACCENT, 2.2), FadeIn(cross, scale=1.4), FadeIn(fail, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.6)
        self.play(FadeOut(fail), FadeOut(cross), cells[0].animate.set_stroke(EDGE, 1.5), b1.animate.move_to(comp.get_top() + UP * 0.45), comp.animate.set_stroke(ACCENT, 2.2), run_time=0.7)
        c1 = entry_content(1).move_to(cells[1])
        contents.append(c1)
        self.play(FadeOut(b1), comp.animate.set_stroke(EDGE, 1.5), FadeIn(c1, shift=UP * 0.1), cells[1].animate.set_stroke(ACCENT, 2.0), *count_to(2), run_time=0.8)
        self.play(cells[1].animate.set_stroke(EDGE, 1.5), run_time=0.3)
        self.wait(2.5)

        switch("EXPLOSION", "連鎖反應", "每次呼叫 step 都不一樣，一次驗票全滅換一張新圖，八格一路被塞滿")
        for i in range(2, 8):
            bi = ball(f"step={i}").move_to([-8, LANE_Y, 0])
            self.play(bi.animate.move_to([-4.6, LANE_Y, 0]), run_time=0.3)
            ci = entry_content(i).move_to(cells[i])
            contents.append(ci)
            self.play(FadeOut(bi, target_position=comp.get_top()), FadeIn(ci, shift=UP * 0.1), cells[i].animate.set_stroke(ACCENT, 2.0), *count_to(i + 1), run_time=0.4)
            self.play(cells[i].animate.set_stroke(EDGE, 1.5), run_time=0.15)
        self.play(Flash(track, color=ACCENT, line_length=0.14, flash_radius=1.9), run_time=0.5)
        self.wait(2.5)

        switch("LIMIT", "保險絲斷了", "第九次失敗撞上 recompile_limit，Dynamo 放棄這個函式，閘門拉下")
        b8 = ball("step=8").move_to([-8, LANE_Y, 0])
        self.play(b8.animate.move_to([-4.6, LANE_Y, 0]), run_time=0.6)
        banner_t = T("hit config.recompile_limit (8)", font=MONO, font_size=17, color=ACCENT)
        banner = VGroup(RoundedRectangle(corner_radius=0.12, width=banner_t.width + 0.5, height=0.52, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD, fill_opacity=1), banner_t).move_to(cells)
        banner.set_z_index(4)
        gate = Rectangle(width=cells.width + 0.35, height=cells.height + 0.35, stroke_color=ACCENT, stroke_width=2.5, fill_color=BG, fill_opacity=0.55).move_to(cells)
        gate.set_z_index(2)
        self.play(FadeIn(gate), VGroup(*[c for c in cells]).animate.set_stroke(DIM, 1.2), run_time=0.6)
        self.play(FadeIn(banner, shift=DOWN * 0.15), run_time=0.5)
        self.wait(3.5)

        switch("EAGER", "水流改道", "之後的每一次呼叫連編譯的門都不敲，整條改走 eager，加速歸零")
        path = VMobject(stroke_color=DIM, stroke_width=2.4)
        path.set_points_as_corners([[-4.6, LANE_Y, 0], [-4.6, -2.75, 0], [7.8, -2.75, 0]])
        dpath = DashedVMobject(path, num_dashes=40)
        elabel = T("eager", font=MONO, font_size=17, color=MUTED).next_to([-4.0, -2.75, 0], UP, buff=0.18).align_to([-3.9, 0, 0], LEFT)
        self.play(Create(dpath), FadeIn(elabel), run_time=0.7)
        self.play(MoveAlongPath(b8, path), run_time=1.4, rate_func=linear)
        self.remove(b8)
        b9 = ball("step=999").move_to([-8, LANE_Y, 0])
        self.play(b9.animate.move_to([-4.6, LANE_Y, 0]), run_time=0.5)
        self.play(MoveAlongPath(b9, path), run_time=1.1, rate_func=linear)
        self.remove(b9)
        et = T("0.17 ms · no compile", font=MONO, font_size=16, color=DIM).next_to(elabel, RIGHT, buff=0.6)
        self.play(FadeIn(et, shift=RIGHT * 0.1), run_time=0.4)
        self.wait(3.0)

        switch("FIX", "常數搬進 tensor", "把 step 包成 tensor 再傳，數值變成資料不是圖的一部分，Guard 不驗數值")
        old = VGroup(cells, cab_label, gate, banner, dpath, elabel, et, comp, comp_t, comp_cost, *contents)
        title2 = T("poly_t(x, s)   s = torch.tensor(step)", font=MONO, font_size=24, color=TXT).move_to(title, aligned_edge=LEFT)
        big = RoundedRectangle(corner_radius=0.12, width=3.0, height=1.35, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1).move_to([3.1, 0.75, 0])
        bg1 = T("g0", font=MONO, font_size=17, color=TXT)
        bg2 = T("TENSOR_MATCH s", font=MONO, font_size=16, color=MUTED)
        bigc = VGroup(bg1, bg2).arrange(DOWN, buff=0.16).move_to(big)
        big_label = T("cache entries", font=MONO, font_size=16, color=MUTED).next_to(big, UP, buff=0.3)
        self.play(FadeOut(old), Transform(title, title2), run_time=0.7)
        self.play(FadeIn(big, shift=UP * 0.15), FadeIn(bigc), FadeIn(big_label), *count_to(1), run_time=0.7)
        self.wait(2.5)

        switch("ALL HIT", "全部命中一格", "同一串呼叫再來一遍，s=0 到 s=9 全部驗票通過，一張圖接住所有值")
        for i in [0, 1, 2, 9]:
            bi = ball(f"s=t({i})").move_to([-8, LANE_Y, 0])
            self.play(bi.animate.move_to([big.get_left()[0] - bi.width / 2 - 0.4, LANE_Y, 0]), run_time=0.45)
            self.play(bg2.animate.set_color(ACCENT), run_time=0.15)
            self.play(bg2.animate.set_color(MUTED), bi.animate.move_to([8.5, LANE_Y, 0]), Flash([big.get_right()[0] + 0.4, LANE_Y, 0], color=ACCENT, line_length=0.12, flash_radius=0.3), run_time=0.5)
            self.remove(bi)
        hit = T("graphs compiled: 1", font=MONO, font_size=16, color=MUTED).next_to(big, DOWN, buff=0.35)
        self.play(FadeIn(hit, shift=UP * 0.1), run_time=0.4)
        self.wait(3.0)

        switch("RESULT", "兩張帳單", "爆炸的版本八張圖付了十秒還退回 eager，修好之後一張圖 0.8 秒收工")
        h1 = T("before  8 graphs · 10.19 s -> eager", font=MONO, font_size=18, color=MUTED)
        h2 = T("after   1 graph  ·  0.82 s -> all hits", font=MONO, font_size=18, color=TXT)
        summary = VGroup(h1, h2).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([-3.3, 0.75, 0])
        self.play(FadeOut(hit), FadeIn(summary, shift=UP * 0.1), run_time=0.7)
        self.play(Indicate(h2, scale_factor=1.08, color=ACCENT), run_time=0.6)
        self.wait(5.5)
