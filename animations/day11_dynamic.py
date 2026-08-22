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
    hd = VGroup(T(title, font=MONO, font_size=12, color=TXT), T(sub, font=CJK, font_size=10, color=MUTED)).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
    inner = VGroup(hd, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, inner)


class Dynamic(Scene):
    def construct(self):
        title = T("f(x, y)  ->  x @ y", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
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
        cache_card = titled(W, H, "CACHE + STATE", "f.__code__").move_to([XS[1], (TOP + BOT) / 2, 0])
        out_card = titled(W, H, "OUTCOME", "驗票結果").move_to([XS[2], (TOP + BOT) / 2, 0])
        lbls = [label("INPUT  ·  x shape").next_to(call_card, UP, buff=0.22).align_to(call_card, LEFT),
                label("GUARDS  ·  frame_state").next_to(cache_card, UP, buff=0.22).align_to(cache_card, LEFT),
                label("RESULT  ·  reuse / recompile").next_to(out_card, UP, buff=0.22).align_to(out_card, LEFT)]
        switch("SETUP", "預設 static", "assume_static_by_default：第一次編譯把 shape 按具體值特化，frame_state 記著每維見過的值")
        self.play(FadeIn(call_card), FadeIn(cache_card), FadeIn(out_card), *[FadeIn(l) for l in lbls], run_time=0.5)

        cw = W - 0.5

        def call_rows(lines):
            return rows(lines, size=13, buff=0.22).next_to(call_card[1], DOWN, buff=0.4).align_to(call_card[1], LEFT)

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

        c1 = call_rows(["x: f32 (4, 4)", "y: f32 (4, 8)"])
        switch("ACT 1", "第一幕：押死", "快取是空的，第一次編譯全部特化：(4, 4) 就是 (4, 4)，只認 batch = 4")
        self.play(FadeIn(c1, shift=RIGHT * 0.1), run_time=0.3)
        o1 = outcome(["no cache entry", "-> COMPILE #1", "   static: size=[4, 4]"], color=ACCENT)
        self.play(FadeIn(o1, shift=RIGHT * 0.1), run_time=0.3)
        G0 = ["L['x']  TENSOR_MATCH [4, 4]", "L['y']  TENSOR_MATCH [4, 8]"]
        e0 = entry_card(cw, "entry 0", G0, "特化的圖").next_to(cache_card[1], DOWN, buff=0.3).align_to(cache_card[1], LEFT)
        fs = entry_card(cw, "frame_state", ["x: dim0 = 4   dim1 = 4"], "每維上次見過的值")
        fs.move_to([XS[1], BOT + fs.height / 2 + 0.25, 0]).align_to(cache_card[1], LEFT)
        self.play(FadeIn(e0, shift=DOWN * 0.1), FadeIn(fs, shift=UP * 0.1), run_time=0.4)
        self.wait(3.5)
        self.play(FadeOut(c1), FadeOut(o1), run_time=0.25)

        c2 = call_rows(["x: f32 (8, 4)", "y: f32 (4, 8)"])
        switch("ACT 2", "第二幕：賭輸", "batch 從 4 換成 8：TENSOR_MATCH 失敗，size mismatch at index 0")
        self.play(FadeIn(c2, shift=RIGHT * 0.1), run_time=0.3)
        check(e0, idx_fail=0)
        o2 = outcome(["size mismatch at index 0", "expected 4, actual 8", "-> RECOMPILE"], color=ACCENT)
        self.play(FadeIn(o2, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(2.5)
        switch("ACT 2", "frame_state 比對", "重編前先翻小冊子：dim 0 上次 4 這次 8，它會變，這個維度改發符號 s0")
        fs2 = entry_card(cw, "frame_state", ["x: dim0 = 4 -> 8  會變", "   dim0 升級成符號 s0"], "自動升級")
        fs2.move_to([XS[1], BOT + fs2.height / 2 + 0.25, 0]).align_to(cache_card[1], LEFT)
        fs2[1][1][1].set_color(ACCENT)
        self.play(FadeOut(fs), FadeIn(fs2), run_time=0.4)
        self.wait(2.5)
        G1 = ["L['x']  TENSOR_MATCH [None, 4]", "L['y']  TENSOR_MATCH [4, 8]", "LAMBDA_GUARD  2 <= s0"]
        e1 = entry_card(cw, "entry 0", G1, "改押 SymInt 的圖").next_to(cache_card[1], DOWN, buff=0.3).align_to(cache_card[1], LEFT)
        e1[1][1][0][18:26].set_color(ACCENT)
        e1[1][1][2].set_color(ACCENT)
        self.play(e0.animate.next_to(e1, DOWN, buff=0.2).align_to(e1, LEFT), run_time=0.4)
        t = T("entry 1", font=MONO, font_size=12, color=TXT).move_to(e0[1][0][0], aligned_edge=LEFT)
        self.play(FadeIn(e1, shift=DOWN * 0.1), Transform(e0[1][0][0], t), run_time=0.4)
        switch("ACT 2", "recompile", "第二次編譯改押符號：size=[None, 4]，換來一條 2 <= s0 的範圍約束")
        self.wait(3.5)
        self.play(FadeOut(c2), FadeOut(o2), run_time=0.25)

        c3 = call_rows(["x: f32 (16, 4)", "y: f32 (4, 8)"])
        switch("ACT 3", "第三幕：通吃", "(16, 4) 進來：size 第一格不看具體值，範圍檢查 2 <= 16 也過")
        self.play(FadeIn(c3, shift=RIGHT * 0.1), run_time=0.3)
        check(e1)
        o3 = outcome(["entry 0: all pass", "reuse, no recompile"], color=TXT)
        self.play(FadeIn(o3, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(2.5)
        self.play(FadeOut(c3), FadeOut(o3), run_time=0.25)

        c4 = call_rows(["x: f32 (100, 4)", "y: f32 (4, 8)"])
        switch("ACT 3", "(100, 4) 也一樣", "recompiles 完全安靜：一張圖吃下所有 batch size")
        self.play(FadeIn(c4, shift=RIGHT * 0.1), run_time=0.3)
        check(e1, hold=0.3)
        o4 = outcome(["entry 0: all pass", "reuse, no recompile", "guard eval ~58 us"], color=TXT)
        self.play(FadeIn(o4, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(2.5)
        switch("RULE", "被逼才 dynamic", "兩次編譯是固定成本，mark_dynamic 可以省掉第一次；0 和 1 永遠特化，所以下界是 2")
        self.wait(5.5)
