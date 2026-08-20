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


def rows(lines, size=12, color=TXT, buff=0.12):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


class SideFx(Scene):
    def construct(self):
        title = T("forward:  self.calls += 1;  log.append(...);  return x * 2", font=MONO, font_size=20, color=TXT).to_corner(UL, buff=0.5)
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
        src_card = titled(W, H, "TRACE", "翻譯到的每一行").move_to([XS[0], (TOP + BOT) / 2, 0])
        ledger_card = titled(W, H, "SIDEEFFECTS", "帳本").move_to([XS[1], (TOP + BOT) / 2, 0])
        graph_card = titled(W, H, "FX GRAPH", "純函數").move_to([XS[2], (TOP + BOT) / 2, 0])
        lbls = [label("INPUT  ·  python").next_to(src_card, UP, buff=0.22).align_to(src_card, LEFT),
                label("LEDGER  ·  不真的改").next_to(ledger_card, UP, buff=0.22).align_to(ledger_card, LEFT),
                label("OUTPUT  ·  只算值").next_to(graph_card, UP, buff=0.22).align_to(graph_card, LEFT)]
        switch("SETUP", "圖只算值，不碰世界", "翻譯期碰到的每一筆修改都不真的做，先記進帳本；Tensor 運算照常進圖")
        self.play(FadeIn(src_card), FadeIn(ledger_card), FadeIn(graph_card), *[FadeIn(l) for l in lbls], run_time=0.5)

        cw = W - 0.5
        src = rows(["self.calls += 1", "log.append(self.calls)", "return x * 2"], size=13, buff=0.35).next_to(src_card[1], DOWN, buff=0.4).align_to(src_card[1], LEFT)
        self.play(FadeIn(src), run_time=0.4)

        led = []
        led_base = ledger_card[1]

        def ledger_add(txt, cap_args):
            switch(*cap_args)
            e = rows(txt, size=12, buff=0.08)
            r = panel(cw, e.height + 0.3, fill=ACTIVE_FILL, edge=ACCENT, r=0.08)
            e.move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
            g = VGroup(r, e)
            if led:
                g.next_to(led[-1], DOWN, buff=0.15).align_to(led[-1], LEFT)
            else:
                g.next_to(led_base, DOWN, buff=0.4).align_to(led_base, LEFT)
            led.append(g)
            return g

        # step 1
        self.play(src[0].animate.set_color(ACCENT), run_time=0.25)
        g1 = ledger_add(["AttributeMutationExisting", "self.calls -> 1"], ("STEP 1", "記帳，不寫入", "self.calls += 1：真實物件不動，帳本記下新值。之後讀 self.calls 讀到的是帳本裡的 1"))
        self.play(FadeIn(g1, shift=RIGHT * 0.1), run_time=0.35)
        self.wait(1.5)
        self.play(src[0].animate.set_color(TXT), g1[0].animate.set_stroke(EDGE).set_fill(CARD_DIM), run_time=0.25)
        # step 2
        self.play(src[1].animate.set_color(ACCENT), run_time=0.25)
        g2 = ledger_add(["ValueMutationExisting", "log -> [1]"], ("STEP 2", "list 也記帳", "log.append(1)：讀的是帳本裡的 calls，list 的修改同樣進帳本"))
        self.play(FadeIn(g2, shift=RIGHT * 0.1), run_time=0.35)
        self.wait(1.5)
        self.play(src[1].animate.set_color(TXT), g2[0].animate.set_stroke(EDGE).set_fill(CARD_DIM), run_time=0.25)
        # step 3
        self.play(src[2].animate.set_color(ACCENT), run_time=0.25)
        switch("STEP 3", "Tensor 運算進圖", "x * 2 是 Tensor 運算，直接進 FX Graph。圖裡沒有 calls 也沒有 log")
        gr = rows(["def forward(L_x_):", "    mul = L_x_ * 2", "    return (mul,)"], size=13, buff=0.14).next_to(graph_card[1], DOWN, buff=0.4).align_to(graph_card[1], LEFT)
        self.play(FadeIn(gr, shift=RIGHT * 0.1), run_time=0.4)
        self.wait(1.5)
        self.play(src[2].animate.set_color(TXT), run_time=0.2)

        # replay
        switch("REPLAY", "圖跑完才結帳", "生成的 bytecode 先呼叫純圖，再把帳本的最終狀態一筆一筆寫回真實世界")
        replay = rows(["call __compiled_fn_1(x)", "object.__setattr__(self,", "  'calls', 1)", "log[:] = [1]", "RETURN"], size=12, buff=0.12)
        rp = panel(cw, replay.height + 0.35, fill=CARD_DIM, edge=ACCENT, r=0.08)
        replay.move_to(rp).align_to(rp.get_left() + RIGHT * 0.15, LEFT)
        grp = VGroup(rp, replay).next_to(gr, DOWN, buff=0.35).align_to(gr, LEFT)
        self.play(FadeIn(grp, shift=UP * 0.1), run_time=0.4)
        a1 = Arrow(g1[0].get_right(), rp.get_left() + UP * 0.3, buff=0.08, color=ACCENT, stroke_width=2, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)
        a2 = Arrow(g2[0].get_right(), rp.get_left() + DOWN * 0.1, buff=0.08, color=ACCENT, stroke_width=2, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)
        self.play(GrowArrow(a1), GrowArrow(a2), run_time=0.4)
        self.wait(1.5)
        switch("RULE", "重播最終狀態", "重播的是最終狀態不是過程：append 十次只補一次；整張圖的執行對外界是原子的")
        self.wait(3.5)
