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


def rows(lines, size=12, color=TXT, buff=0.1):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def entry_card(w, title, guards, sub=""):
    body = rows(guards, size=11, buff=0.08)
    r = panel(w, body.height + 0.65, fill=CARD_DIM, edge=EDGE, r=0.08)
    hd = VGroup(T(title, font=MONO, font_size=12, color=TXT), T(sub, font=MONO, font_size=10, color=MUTED)).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
    inner = VGroup(hd, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, inner)


class Guards(Scene):
    def construct(self):
        title = T("f(x, n)  ->  x * n * cfg_scale", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        TOP, BOT = 2.45, -2.9
        H = TOP - BOT
        W = 4.07
        XS = [-4.575, 0.0, 4.575]
        call_card = titled(W, H, "CALL", "這次呼叫的輸入").move_to([XS[0], (TOP + BOT) / 2, 0])
        cache_card = titled(W, H, "CACHE ENTRIES", "f.__code__ 上掛的圖").move_to([XS[1], (TOP + BOT) / 2, 0])
        out_card = titled(W, H, "OUTCOME", "驗票結果").move_to([XS[2], (TOP + BOT) / 2, 0])
        lbls = [label("INPUT  ·  frame locals").next_to(call_card, UP, buff=0.22).align_to(call_card, LEFT),
                label("GUARD TREE  ·  C++").next_to(cache_card, UP, buff=0.22).align_to(cache_card, LEFT),
                label("RESULT  ·  reuse / recompile").next_to(out_card, UP, buff=0.22).align_to(out_card, LEFT)]
        switch("SETUP", "先驗票再重用", "每個 code object 掛著若干 cache entry，每個 entry 是一組 Guard 樹加編好的圖")
        self.play(FadeIn(call_card), FadeIn(cache_card), FadeIn(out_card), *[FadeIn(l) for l in lbls], run_time=0.5)

        cw = W - 0.5
        G0 = ["GLOBAL_STATE   grad on", "L['x']  TENSOR_MATCH  f32 [4,4]", "L['n']  EQUALS_MATCH  n == 3"]
        e0 = entry_card(cw, "entry 0", G0, "n == 3 的那張圖").next_to(cache_card[1], DOWN, buff=0.3).align_to(cache_card[1], LEFT)
        self.play(FadeIn(e0), run_time=0.4)

        def call_rows(lines):
            g = rows(lines, size=13, buff=0.22).next_to(call_card[1], DOWN, buff=0.4).align_to(call_card[1], LEFT)
            return g

        def outcome(lines, color=TXT):
            g = rows(lines, size=13, buff=0.22).next_to(out_card[1], DOWN, buff=0.4).align_to(out_card[1], LEFT)
            g.set_color(color)
            return g

        def check(entry, idx_fail=None, hold=0.35):
            body = entry[1][1]
            for i, r in enumerate(body):
                self.play(r.animate.set_color(ACCENT), run_time=0.15)
                if idx_fail == i:
                    x = T("x", font=MONO, font_size=13, color=ACCENT, weight=BOLD).next_to(r, RIGHT, buff=0.15)
                    self.play(FadeIn(x), run_time=0.15)
                    self.wait(hold)
                    self.play(*[rr.animate.set_color(TXT) for rr in body], FadeOut(x), run_time=0.15)
                    return False
                self.play(r.animate.set_color(MUTED), run_time=0.08)
            self.wait(hold)
            self.play(*[r.animate.set_color(TXT) for r in body], run_time=0.15)
            return True

        # ---- call 1: hit ----
        c1 = call_rows(["x: f32 [4,4]  (new values)", "n: 3", "grad: on"])
        switch("CALL 1", "f(x, 3)", "同 shape 同 dtype 的新 Tensor，n 還是 3：整棵樹全過")
        self.play(FadeIn(c1, shift=RIGHT * 0.1), run_time=0.3)
        check(e0)
        o1 = outcome(["reuse entry 0", "no recompile", "guard eval ~10 us"], color=TXT)
        self.play(FadeIn(o1, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(1.5)
        self.play(FadeOut(c1), FadeOut(o1), run_time=0.25)

        # ---- call 2: fail on n -> recompile, new entry prepended ----
        c2 = call_rows(["x: f32 [4,4]", "n: 4", "grad: on"])
        switch("CALL 2", "f(x, 4)", "被 bake 的 int 變了：EQUALS_MATCH 失敗，沒有別的 entry，只能重編")
        self.play(FadeIn(c2, shift=RIGHT * 0.1), run_time=0.3)
        check(e0, idx_fail=2)
        o2 = outcome(["entry 0: n == 3  x", "no entry left", "-> RECOMPILE"], color=ACCENT)
        self.play(FadeIn(o2, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(0.8)
        G1 = ["GLOBAL_STATE   grad on", "L['x']  TENSOR_MATCH  f32 [4,4]", "L['n']  TYPE_MATCH  int"]
        e1 = entry_card(cw, "entry 0", G1, "n 改成符號整數").next_to(cache_card[1], DOWN, buff=0.3).align_to(cache_card[1], LEFT)
        self.play(e0.animate.next_to(e1, DOWN, buff=0.15).align_to(e1, LEFT), run_time=0.4)
        t = T("entry 1", font=MONO, font_size=12, color=TXT).move_to(e0[1][0][0], aligned_edge=LEFT)
        self.play(FadeIn(e1, shift=DOWN * 0.1), Transform(e0[1][0][0], t), run_time=0.4)
        switch("CALL 2", "recompile", "新圖掛到最前面。注意 Dynamo 這次不再 bake n，改押符號整數：automatic dynamic")
        self.wait(1.5)
        self.play(FadeOut(c2), FadeOut(o2), run_time=0.25)

        # ---- call 3: no_grad -> both fail ----
        c3 = call_rows(["x: f32 [4,4]", "n: 3", "grad: OFF  (no_grad)"])
        switch("CALL 3", "no_grad 下呼叫", "參數都沒變，但 grad mode 是隱形前提：GLOBAL_STATE 在每個 entry 都失敗")
        self.play(FadeIn(c3, shift=RIGHT * 0.1), run_time=0.3)
        check(e1, idx_fail=0, hold=0.3)
        check(e0, idx_fail=0, hold=0.3)
        o3 = outcome(["entry 0: GLOBAL_STATE  x", "entry 1: GLOBAL_STATE  x", "-> RECOMPILE (3rd graph)"], color=ACCENT)
        self.play(FadeIn(o3, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(0.8)
        G2 = ["GLOBAL_STATE   grad OFF", "L['x']  TENSOR_MATCH  f32 [4,4]", "L['n']  EQUALS_MATCH  n == 3"]
        e2 = entry_card(cw, "entry 0", G2, "no_grad 的那張圖").next_to(cache_card[1], DOWN, buff=0.3).align_to(cache_card[1], LEFT)
        self.play(e1.animate.next_to(e2, DOWN, buff=0.15).align_to(e2, LEFT), e0.animate.next_to(e2, DOWN, buff=0.15 * 2 + e1.height).align_to(e2, LEFT), run_time=0.4)
        t1 = T("entry 1", font=MONO, font_size=12, color=TXT).move_to(e1[1][0][0], aligned_edge=LEFT)
        t2 = T("entry 2", font=MONO, font_size=12, color=TXT).move_to(e0[1][0][0], aligned_edge=LEFT)
        self.play(FadeIn(e2, shift=DOWN * 0.1), Transform(e1[1][0][0], t1), Transform(e0[1][0][0], t2), run_time=0.4)
        self.wait(1.2)
        self.play(FadeOut(c3), FadeOut(o3), run_time=0.25)

        # ---- call 4: hits older entry, moved to front ----
        c4 = call_rows(["x: f32 [4,4]", "n: 3", "grad: on"])
        switch("CALL 4", "f(x, 3) 又來", "逐個 entry 驗：no_grad 那張先失敗，符號整數那張全過。命中的 entry 會被搬到最前面")
        self.play(FadeIn(c4, shift=RIGHT * 0.1), run_time=0.3)
        check(e2, idx_fail=0, hold=0.3)
        check(e1)
        o4 = outcome(["entry 0: GLOBAL_STATE  x", "entry 1: all pass", "-> reuse entry 1", "-> move to front"], color=TXT)
        self.play(FadeIn(o4, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(0.6)
        self.play(e1.animate.next_to(cache_card[1], DOWN, buff=0.3).align_to(cache_card[1], LEFT), e2.animate.next_to(cache_card[1], DOWN, buff=0.3 + e1.height + 0.15).align_to(cache_card[1], LEFT), run_time=0.5)
        t3 = T("entry 0", font=MONO, font_size=12, color=TXT).move_to(e1[1][0][0], aligned_edge=LEFT)
        t4 = T("entry 1", font=MONO, font_size=12, color=TXT).move_to(e2[1][0][0], aligned_edge=LEFT)
        self.play(Transform(e1[1][0][0], t3), Transform(e2[1][0][0], t4), run_time=0.3)
        self.wait(1.5)
        switch("RULE", "圖有多特化，Guard 就有多少條", "Guard 是翻譯時押注的帳單：全過才重用、全敗才重編、超過 recompile_limit 就放棄改跑 eager")
        self.wait(3.5)
