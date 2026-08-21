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


def titled(w, h, name, sub):
    r = panel(w, h)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
    return VGroup(r, hdr)


def pill(name, zh):
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    body = T(f"{name}  ·  {zh}", font_size=17, font=SANS, color=BG, t2f={name: SANS, "·": MONO, zh: zh_font}, t2w={name: BOLD}, t2c={"·": "#666"})
    t = VGroup(Dot(radius=0.06, color=ACCENT), body).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def chip(main, sub, w, main_color=TXT, edge=EDGE, fill=CARD_DIM):
    m = T(main, font=MONO, font_size=12, color=main_color)
    s = T(sub, font=MONO, font_size=10, color=MUTED)
    body = VGroup(m, s).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    r = panel(w, body.height + 0.3, fill=fill, edge=edge, r=0.08)
    body.move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, body)


def arrow(a, b, color=MUTED, w=1.8):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


ROWS = [
    ("x", "Tensor f32[8]", "L['x']", "TensorVariable", "as_proxy() 進圖", "graph input L_x_", "TENSOR_MATCH  shape / dtype", ACCENT, "Tensor 進圖當輸入，Guard 只守 shape 與 dtype，不守數值"),
    ("n", "int  3", "L['n']", "ConstantVariable", "as_python_constant()", "bake:  y * 3", "EQUALS_MATCH  n == 3", TXT, "int 不當輸入，直接 bake 成圖裡的常數，代價是一條 n == 3 的 Guard"),
    ("items[0]", "list  [1, 2]", "L['items'][0]", "ListVariable -> ConstantVariable", "翻譯期取元素", "bake:  + 1", "EQUALS_MATCH  items[0] == 1", TXT, "list 在符號世界被拆開，取出的元素同樣 bake 進圖"),
    ("cfg.scale", "attr  int 2", "L['cfg'].scale", "UserDefinedObjectVariable", "var_getattr() 逐屬性", "bake:  + 2", "EQUALS_MATCH  cfg.scale == 2", TXT, "物件逐屬性追蹤，Source 鏈記著 L['cfg'].scale 怎麼取"),
    ("helper", "function", "L['helper']", "UserFunctionVariable", "call_function() -> inline", "inline:  y = L_x_ * 3", "ID_MATCH  helper.__code__", TXT, "自己寫的函式不斷圖，被 inline 攤進同一張圖，守它的 __code__"),
]


class VT(Scene):
    def construct(self):
        title = T("f(x, n, items, cfg)", font=MONO, font_size=26, color=TXT).to_corner(UL, buff=0.5).shift(UP * 0.15)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        TOP, BOT = 2.45, -2.85
        H = TOP - BOT
        W = 4.07
        XS = [-4.575, 0.0, 4.575]
        cards = [titled(W, H, "VALUE", "外面世界進來的值").move_to([XS[0], (TOP + BOT) / 2, 0]),
                 titled(W, H, "VARIABLETRACKER", "替身").move_to([XS[1], (TOP + BOT) / 2, 0]),
                 titled(W, H, "EFFECT", "進圖 / bake / inline").move_to([XS[2], (TOP + BOT) / 2, 0])]
        lbls = [label("INPUT  ·  Source").next_to(cards[0], UP, buff=0.22).align_to(cards[0], LEFT),
                label("WRAP  ·  variables/").next_to(cards[1], UP, buff=0.22).align_to(cards[1], LEFT),
                label("OUTPUT  ·  graph + guard").next_to(cards[2], UP, buff=0.22).align_to(cards[2], LEFT)]
        switch("SETUP", "每個值都要包", "Dynamo 手上沒有真值，每個進來的 Python 值都先包成一個知道自己是什麼的替身")
        self.play(*[FadeIn(c) for c in cards], *[FadeIn(l) for l in lbls], run_time=0.5)

        row_top = TOP - 0.85
        row_h = 0.9
        cw = W - 0.5
        for i, (name, typ, src, vt, how, eff, guard, col, cap) in enumerate(ROWS):
            y = row_top - i * row_h - 0.25
            c1 = chip(f"{name}   {typ}", src, cw).move_to([XS[0], y, 0])
            c2 = chip(vt, how, cw, main_color=col).move_to([XS[1], y, 0])
            c3 = chip(eff, guard, cw, main_color=col).move_to([XS[2], y, 0])
            a1 = arrow(c1[0].get_right(), c2[0].get_left())
            a2 = arrow(c2[0].get_right(), c3[0].get_left())
            switch(f"ROW {i + 1}", name, cap)
            self.play(FadeIn(c1, shift=RIGHT * 0.1), run_time=0.3)
            self.play(GrowArrow(a1), FadeIn(c2, shift=RIGHT * 0.1), run_time=0.35)
            self.play(GrowArrow(a2), FadeIn(c3, shift=RIGHT * 0.1), run_time=0.35)
            self.wait(2.5)
        switch("RULE", "Guard 守邊界", "有 Source 的值走 VariableBuilder 並裝 Guard；追蹤中途生出的值走 SourcelessBuilder，不需要 Guard")
        self.wait(4.5)
