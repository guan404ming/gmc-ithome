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


def arrow(a, b, color=ACCENT, w=2):
    return Arrow(a, b, buff=0.08, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


def phase_chip(name, zh):
    n = T(name, font=SANS, font_size=13, weight=BOLD, color=MUTED)
    z = T(zh, font=CJK, font_size=11, color=MUTED)
    body = VGroup(n, z).arrange(RIGHT, buff=0.15)
    r = panel(2.5, 0.5, fill=CARD_DIM, edge=EDGE, r=0.25, sw=1.2)
    return VGroup(r, body.move_to(r))


def ledger_entry(w, typ, content):
    body = rows([typ, content], size=11, buff=0.08)
    body[0].set_color(ACCENT)
    r = panel(w, body.height + 0.3, fill=ACTIVE_FILL, edge=ACCENT, r=0.08, sw=1.2)
    body.move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, body)


def badge(en, zh, edge=EDGE, color=MUTED):
    e = T(en, font=MONO, font_size=12, color=color)
    z = T(zh, font=CJK, font_size=11, color=color)
    body = VGroup(e, z).arrange(RIGHT, buff=0.15)
    r = panel(body.width + 0.5, 0.5, fill=CARD_DIM, edge=edge, r=0.25, sw=1.2)
    return VGroup(r, body.move_to(r))


class SideFx(Scene):
    def construct(self):
        title = T("forward:  self.calls += 1;  log.append(...);  return x * 2", font=MONO, font_size=18, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        chips = VGroup(phase_chip("TRACE", "追蹤"), phase_chip("RUN", "圖執行"), phase_chip("REPLAY", "重播")).arrange(RIGHT, buff=0.35).move_to([0, 2.82, 0])

        def set_phase(idx):
            anims = []
            for i, ch in enumerate(chips):
                if i == idx:
                    anims += [ch[0].animate.set_stroke(ACCENT).set_fill(ACTIVE_FILL), ch[1][0].animate.set_color(TXT), ch[1][1].animate.set_color(TXT)]
                else:
                    anims += [ch[0].animate.set_stroke(EDGE).set_fill(CARD_DIM), ch[1][0].animate.set_color(MUTED), ch[1][1].animate.set_color(MUTED)]
            self.play(*anims, run_time=0.3)

        TOP, BOT = 2.35, -2.9
        H = TOP - BOT
        WL, WR = 4.4, 8.45
        XL, XR = -4.4, 2.375
        real_card = titled(WL, H, "REAL WORLD", "真實世界").move_to([XL, (TOP + BOT) / 2, 0])
        sym_card = titled(WR, H, "SYMBOLIC WORLD", "追蹤期").move_to([XR, (TOP + BOT) / 2, 0])

        lx = XL - WL / 2 + 0.25
        r1 = T("self.calls   0", font=MONO, font_size=14, color=TXT).move_to([lx, 1.15, 0], aligned_edge=LEFT)
        r2 = T("log          []", font=MONO, font_size=14, color=TXT).move_to([lx, 0.05, 0], aligned_edge=LEFT)
        frozen = badge("FROZEN", "追蹤期間凍結").move_to([XL, -2.25, 0])

        src = rows(["self.calls += 1", "log.append(self.calls)", "return x * 2"], size=13, buff=0.15).next_to(sym_card[1], DOWN, buff=0.3).align_to(sym_card[1], LEFT)

        PW, PH = 3.775, 2.75
        PX1 = XR - WR / 2 + 0.25 + PW / 2
        PX2 = PX1 + PW + 0.4
        PY = -1.125
        led_panel = panel(PW, PH, fill=CARD_DIM, r=0.1)
        led_panel.move_to([PX1, PY, 0])
        led_lbl = label("SIDEEFFECTS  ·  帳本", size=11).move_to(led_panel.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.2, aligned_edge=UL)
        graph_panel = panel(PW, PH, fill=CARD_DIM, r=0.1)
        graph_panel.move_to([PX2, PY, 0])
        graph_lbl = label("FX GRAPH  ·  純函數", size=11).move_to(graph_panel.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.2, aligned_edge=UL)

        switch("SETUP", "圖只算值", "左邊是真實世界，右邊是 Dynamo 的符號世界：追蹤期間所有修改只進帳本，真實物件凍結")
        self.play(FadeIn(real_card), FadeIn(sym_card), FadeIn(chips), FadeIn(r1), FadeIn(r2), FadeIn(frozen), FadeIn(src), FadeIn(led_panel), FadeIn(led_lbl), FadeIn(graph_panel), FadeIn(graph_lbl), run_time=0.6)
        self.wait(2.5)

        ew = PW - 0.5
        set_phase(0)
        self.play(src[0].animate.set_color(ACCENT), run_time=0.25)
        e1 = ledger_entry(ew, "AttributeMutationExisting", "self.calls -> 1").next_to(led_lbl, DOWN, buff=0.25).set_x(PX1)
        switch("STEP 1", "記帳，不寫入", "self.calls += 1：真實世界的 calls 還是 0，帳本記下新值 1，之後讀 calls 讀到的是帳本")
        self.play(FadeIn(e1, shift=RIGHT * 0.1), run_time=0.35)
        self.play(Indicate(frozen, color=MUTED, scale_factor=1.05), run_time=0.5)
        self.wait(3.5)
        self.play(src[0].animate.set_color(TXT), e1[0].animate.set_stroke(EDGE), run_time=0.25)

        self.play(src[1].animate.set_color(ACCENT), run_time=0.25)
        self.play(Indicate(e1, color=ACCENT, scale_factor=1.03), run_time=0.5)
        e2 = ledger_entry(ew, "ValueMutationExisting", "log -> [1]").next_to(e1, DOWN, buff=0.2).set_x(PX1)
        switch("STEP 2", "list 也記帳", "log.append 讀到帳本裡的 1；list 的修改同樣進帳本，記的是最終內容不是過程")
        self.play(FadeIn(e2, shift=RIGHT * 0.1), run_time=0.35)
        self.wait(3.5)
        self.play(src[1].animate.set_color(TXT), e2[0].animate.set_stroke(EDGE), run_time=0.25)

        self.play(src[2].animate.set_color(ACCENT), run_time=0.25)
        gr = rows(["def forward(L_x_):", "    mul = L_x_ * 2", "    return (mul,)"], size=12, buff=0.15).next_to(graph_lbl, DOWN, buff=0.25).align_to(graph_lbl, LEFT)
        switch("STEP 3", "Tensor 運算進圖", "x * 2 是 Tensor 運算，直接進 FX Graph。圖裡沒有 calls 也沒有 log，是一張純函數")
        self.play(FadeIn(gr, shift=RIGHT * 0.1), run_time=0.4)
        self.wait(3.5)
        self.play(src[2].animate.set_color(TXT), run_time=0.2)

        set_phase(1)
        run_line = T("call __compiled_fn_1(x)", font=MONO, font_size=11, color=ACCENT).next_to(gr, DOWN, buff=0.2).align_to(gr, LEFT)
        switch("RUN", "圖先跑，世界不動", "改寫後的 bytecode 先呼叫純圖：圖在 GPU 上算多久，真實世界就凍結多久")
        self.play(src.animate.set_color(DIM), graph_panel.animate.set_stroke(ACCENT), FadeIn(run_line), run_time=0.4)
        self.play(Indicate(frozen, color=MUTED, scale_factor=1.05), run_time=0.5)
        self.wait(2.5)

        set_phase(2)
        switch("REPLAY", "逐筆結帳", "圖跑完才結帳：帳本的最終狀態由生成的 bytecode 一筆一筆寫回真實世界")
        self.play(graph_panel.animate.set_stroke(EDGE), run_line.animate.set_color(MUTED), run_time=0.3)
        a1 = arrow(e1[0].get_left(), [XL + WL / 2 - 0.1, 1.15, 0])
        self.play(e1[0].animate.set_stroke(ACCENT), GrowArrow(a1), run_time=0.4)
        n1 = T("self.calls   1", font=MONO, font_size=14, color=TXT).move_to([lx, 1.15, 0], aligned_edge=LEFT)
        s1 = T("object.__setattr__(self,'calls',1)", font=MONO, font_size=10, color=MUTED).move_to([lx, 0.78, 0], aligned_edge=LEFT)
        self.play(Transform(r1, n1), FadeIn(s1), run_time=0.4)
        self.play(FadeOut(a1), e1[0].animate.set_stroke(EDGE), run_time=0.25)
        self.wait(2)
        a2 = arrow(e2[0].get_left(), [XL + WL / 2 - 0.1, 0.05, 0])
        self.play(e2[0].animate.set_stroke(ACCENT), GrowArrow(a2), run_time=0.4)
        n2 = T("log          [1]", font=MONO, font_size=14, color=TXT).move_to([lx, 0.05, 0], aligned_edge=LEFT)
        s2 = T("log[:] = [1]", font=MONO, font_size=10, color=MUTED).move_to([lx, -0.32, 0], aligned_edge=LEFT)
        self.play(Transform(r2, n2), FadeIn(s2), run_time=0.4)
        self.play(FadeOut(a2), e2[0].animate.set_stroke(EDGE), run_time=0.25)
        synced = badge("SYNCED", "已同步", edge=ACCENT, color=TXT).move_to([XL, -2.25, 0])
        self.play(Transform(frozen, synced), run_time=0.4)
        self.wait(3)

        switch("RULE", "重播最終狀態", "append 十次只補一次；existing 必重播、沒逃出去的 new 整筆勾銷；對外界而言整張圖是原子的")
        self.wait(5.5)
