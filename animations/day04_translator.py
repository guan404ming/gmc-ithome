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


def panel(w, h, fill=CARD, edge=EDGE, r=0.12):
    return RoundedRectangle(corner_radius=r, width=w, height=h, stroke_color=edge, stroke_width=1.5, fill_color=fill, fill_opacity=1)


def header(name, sub):
    t = T(name, font=SANS, font_size=21, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=14, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub):
    r = panel(w, h)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
    return VGroup(r, hdr)


def pill(name, zh):
    t = VGroup(Dot(radius=0.06, color=ACCENT), T(name, font=SANS, font_size=18, weight=BOLD, color=BG), T("·", font=MONO, font_size=16, color="#666"), T(zh, font=CJK, font_size=16, color=BG)).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def chip(txt, w, color=TXT, fill=CARD_DIM, edge=EDGE):
    r = panel(w, 0.4, fill=fill, edge=edge, r=0.08)
    t = T(txt, font=MONO, font_size=12, color=color).move_to(r)
    return VGroup(r, t)


def node(txt, w=1.5, accent=False):
    r = panel(w, 0.46, fill=ACTIVE_FILL if accent else CARD_DIM, edge=ACCENT if accent else EDGE, r=0.08)
    t = T(txt, font=MONO, font_size=12, color=TXT).move_to(r)
    return VGroup(r, t)


BC = ["LOAD_FAST    x", "LOAD_FAST    y", "BINARY_OP    *", "STORE_FAST   z", "LOAD_FAST    z", "LOAD_CONST   1", "BINARY_OP    +", "RETURN_VALUE"]


class Translator(Scene):
    def construct(self):
        title = T("InstructionTranslator", font=MONO, font_size=26, color=TXT).to_corner(UL, buff=0.5)
        title[11:21].set_color(ACCENT)
        self.play(FadeIn(title), run_time=0.4)

        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        # ---- layout ----
        TOP, BOT = 2.45, -2.75
        H = TOP - BOT
        LX, LW = -4.75, 3.7
        MX, MW = -0.85, 3.3
        RX, RW = 3.7, 5.3

        bc_card = titled(LW, H, "BYTECODE", "dispatch_table 查表").move_to([LX, (TOP + BOT) / 2, 0])
        rows = VGroup(*[T(l, font=MONO, font_size=14, color=TXT) for l in BC]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(bc_card[1], DOWN, buff=0.4).align_to(bc_card[1], LEFT)
        ptr = panel(LW - 0.4, 0.46, fill=ACTIVE_FILL, edge=ACCENT, r=0.06).set_opacity(0)

        st_h = 3.0
        st_card = titled(MW, st_h, "STACK", "符號值").move_to([MX, TOP - st_h / 2, 0])
        lc_h = H - st_h - 0.25
        lc_card = titled(MW, lc_h, "LOCALS", "symbolic_locals").move_to([MX, BOT + lc_h / 2, 0])
        loc_rows = VGroup(*[T(s, font=MONO, font_size=12, color=TXT) for s in ["x: LazyVT", "y: LazyVT"]]).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(lc_card[1], DOWN, buff=0.22).align_to(lc_card[1], LEFT)

        fx_card = titled(RW, H, "FX GRAPH", "只有 Tensor 運算留下").move_to([RX, (TOP + BOT) / 2, 0])
        lbl_l = label("INPUT  ·  bytecode").next_to(bc_card, UP, buff=0.22).align_to(bc_card, LEFT)
        lbl_m = label("STATE  ·  stack machine").next_to(st_card, UP, buff=0.22).align_to(st_card, LEFT)
        lbl_r = label("OUTPUT  ·  graph").next_to(fx_card, UP, buff=0.22).align_to(fx_card, LEFT)

        switch("SETUP", "同一台 stack machine", "InstructionTranslator 把 CPython 的 stack machine 搬到 Python 層，stack 裡放的是符號替身")
        self.play(FadeIn(lbl_l), FadeIn(bc_card), FadeIn(rows), FadeIn(lbl_m), FadeIn(st_card), FadeIn(lc_card), FadeIn(loc_rows), FadeIn(lbl_r), FadeIn(fx_card), run_time=0.6)
        self.add(ptr)

        # graph nodes positions
        gx0 = fx_card[0].get_left()[0] + 0.35
        n_x = node("L_x_  f32[8]").move_to([gx0 + 0.9, TOP - 0.95, 0])
        n_y = node("L_y_  f32[8]").move_to([gx0 + 2.6, TOP - 0.95, 0])
        n_mul = node("mul", w=1.2).move_to([gx0 + 1.75, TOP - 2.05, 0])
        n_c1 = node("1", w=0.6).move_to([gx0 + 3.6, TOP - 2.05, 0])
        n_add = node("add", w=1.2).move_to([gx0 + 2.65, TOP - 3.15, 0])
        n_out = node("output", w=1.5).move_to([gx0 + 2.65, TOP - 4.25, 0])

        def edge(a, b):
            return Line(a.get_bottom(), b.get_top(), color=MUTED, stroke_width=1.5, buff=0.05)

        # stack helpers
        stack = []
        stack_base = st_card[0].get_bottom() + UP * 0.35
        chip_w = MW - 0.6

        def push(txt, color=TXT, edge_c=EDGE, run_time=0.35):
            c = chip(txt, chip_w, color=color, edge=edge_c)
            c.move_to(stack_base + UP * (0.45 * len(stack) + 0.2))
            stack.append(c)
            self.play(FadeIn(c, shift=UP * 0.15), run_time=run_time)
            return c

        def pop(run_time=0.3):
            c = stack.pop()
            self.play(FadeOut(c, shift=UP * 0.15), run_time=run_time)
            return c

        def point(i):
            self.play(ptr.animate.set_opacity(1).move_to(rows[i]).align_to(bc_card[0].get_left() + RIGHT * 0.2, LEFT), run_time=0.25)
            self.bring_to_front(rows[i])

        HOLD = 1.5

        # 1 LOAD_FAST x
        point(0); switch("STEP 1", "LOAD_FAST x", "查 symbolic_locals，把 x 的替身推上 stack。還沒人用它，所以先是 Lazy 的殼")
        push("x  LazyVariableTracker", color=MUTED); self.wait(HOLD)
        # 2 LOAD_FAST y
        point(1); switch("STEP 2", "LOAD_FAST y", "同樣，y 的替身推上 stack。圖上什麼都沒有")
        push("y  LazyVariableTracker", color=MUTED); self.wait(HOLD)
        # 3 BINARY_OP *
        point(2); switch("STEP 3", "BINARY_OP *", "彈出兩個運算元，發現都是 Tensor：不算，往 FX Graph 加 mul 節點，推回結果的替身")
        pop(0.2); pop(0.2)
        self.play(FadeIn(n_x), FadeIn(n_y), run_time=0.3)
        e1, e2 = edge(n_x, n_mul), edge(n_y, n_mul)
        self.play(FadeIn(n_mul), Create(e1), Create(e2), run_time=0.4)
        n_mul[0].set_stroke(ACCENT).set_fill(ACTIVE_FILL)
        push("z  TensorVariable", color=ACCENT, edge_c=ACCENT); self.wait(HOLD)
        n_mul[0].set_stroke(EDGE).set_fill(CARD_DIM)
        # 4 STORE_FAST z
        point(3); switch("STEP 4", "STORE_FAST z", "彈出頂端，存進 symbolic_locals[\"z\"]。一行 dict 賦值，圖上沒事")
        c = pop(0.25)
        z_row = T("z: TensorVariable", font=MONO, font_size=12, color=ACCENT).next_to(loc_rows, DOWN, aligned_edge=LEFT, buff=0.12)
        self.play(FadeIn(z_row, shift=LEFT * 0.2), run_time=0.3); self.wait(HOLD)
        # 5 LOAD_FAST z
        point(4); switch("STEP 5", "LOAD_FAST z", "再把 z 的替身推回 stack")
        push("z  TensorVariable"); self.wait(HOLD)
        # 6 LOAD_CONST 1
        point(5); switch("STEP 6", "LOAD_CONST 1", "推一個常數替身 ConstantVariable(1)。Python 的 int 就這樣被帶著走")
        push("1  ConstantVariable(int)", color=MUTED); self.wait(HOLD)
        # 7 BINARY_OP +
        point(6); switch("STEP 7", "BINARY_OP +", "Tensor 加常數：往圖上加 add 節點，常數直接 bake 進去")
        pop(0.2); pop(0.2)
        e3, e4 = edge(n_mul, n_add), edge(n_c1, n_add)
        self.play(FadeIn(n_c1), FadeIn(n_add), Create(e3), Create(e4), run_time=0.4)
        push("add  TensorVariable", color=ACCENT, edge_c=ACCENT); self.wait(HOLD)
        # 8 RETURN_VALUE
        point(7); switch("STEP 8", "RETURN_VALUE", "翻譯結束：stack 頂端就是輸出，收圖交給後端。8 條指令，只有 2 條進了圖")
        e5 = edge(n_add, n_out)
        self.play(FadeIn(n_out), Create(e5), run_time=0.4)
        self.play(*[r.animate.set_color(DIM) for i, r in enumerate(rows) if i not in (2, 6)], rows[2].animate.set_color(ACCENT), rows[6].animate.set_color(ACCENT), ptr.animate.set_opacity(0), run_time=0.5)
        self.wait(3.5)
