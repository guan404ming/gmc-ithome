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

COL_W = 6.3
LX, RX = -3.55, 3.55


def T(txt, font_size, **kw):
    return Text(txt, font_size=font_size * 4, **kw).scale(0.25)


def label(s, size=15, color=MUTED):
    return T(s, font=MONO, font_size=size, color=color)


def panel(w, h, fill=CARD, edge=EDGE):
    return RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=edge, stroke_width=1.5, fill_color=fill, fill_opacity=1)


def header(name, sub):
    t = T(name, font=SANS, font_size=21, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=14, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub, fill=CARD):
    r = panel(w, h, fill)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
    return VGroup(r, hdr)


def code_lines(lines, size=12, color=TXT, buff=0.085):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def pill(name, zh):
    t = VGroup(Dot(radius=0.06, color=ACCENT), T(name, font=SANS, font_size=18, weight=BOLD, color=BG), T("·", font=MONO, font_size=16, color="#666"), T(zh, font=CJK, font_size=16, color=BG)).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def varrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.08, color=color, stroke_width=w, max_tip_length_to_length_ratio=0.5, max_stroke_width_to_length_ratio=10)


def mini(name, lines, w):
    body = code_lines(lines, size=11, buff=0.09)
    r = panel(max(w, body.width + 0.35), body.height + 0.5, fill=CARD_DIM, edge=EDGE)
    hd = T(name, font=MONO, font_size=11, color=MUTED)
    inner = VGroup(hd, body).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, inner)


BC = ["LOAD_DEREF   torch", "LOAD_ATTR    sin", "LOAD_FAST    x", "CALL         1", "LOAD_CONST   1", "BINARY_OP    +", "RETURN_VALUE"]
BC_NEW = ["LOAD_GLOBAL  __compiled_fn_1", "LOAD_FAST    x", "CALL         1", "RETURN_VALUE"]

Y_HDR, Y1, Y2, Y3 = 2.78, 2.1, 0.1, -2.2
H1, H2, H3 = 0.9, 2.6, 1.55


def column(x, tag, tag_zh, ptr_text, ptr_color):
    hdr = label(f"{tag}  ·  {tag_zh}").move_to([x - COL_W / 2, Y_HDR, 0], aligned_edge=LEFT)
    c1 = titled(COL_W, H1, "eval_frame", "函式指標").move_to([x, Y1, 0])
    val = T(ptr_text, font=MONO, font_size=14, color=ptr_color).move_to(c1[0].get_right() + LEFT * 0.25, aligned_edge=RIGHT)
    c2 = titled(COL_W, H2, "FRAME", "f 的 frame，code object 裡的 bytecode").move_to([x, Y2, 0])
    rows = code_lines(BC).next_to(c2[1], DOWN, buff=0.22).align_to(c2[1], LEFT)
    a12 = varrow(c1[0].get_bottom(), c2[0].get_top())
    a23 = varrow(c2[0].get_bottom(), [x, Y3 + H3 / 2, 0])
    return hdr, c1, val, c2, rows, a12, a23


class EvalFrame(Scene):
    def construct(self):
        title = T("PEP 523 · eval_frame", font=MONO, font_size=26, color=TXT).to_corner(UL, buff=0.5)
        title[7:17].set_color(ACCENT)
        self.play(FadeIn(title), run_time=0.4)

        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        # ---- column A ----
        hdrA, a1, valA, a2, rowsA, aa12, aa23 = column(LX, "A", "沒有 torch.compile", "= _PyEval_EvalFrameDefault", TXT)
        a3 = titled(COL_W, H3, "執行", "一條一條跑").move_to([LX, Y3, 0])
        a3_body = code_lines(["sin(x) + 1  ->  Tensor", "7 條指令，7 次 dispatch"], size=13, color=MUTED).next_to(a3[1], DOWN, buff=0.2).align_to(a3[1], LEFT)

        switch("CPYTHON", "預設路徑", "沒有 torch.compile 時，eval_frame 指向預設的 evaluator，frame 交給它一條一條執行")
        self.play(FadeIn(hdrA), FadeIn(a1), FadeIn(valA), run_time=0.4)
        self.play(GrowArrow(aa12), FadeIn(a2), FadeIn(rowsA), run_time=0.4)
        self.play(GrowArrow(aa23), FadeIn(a3), FadeIn(a3_body), run_time=0.4)
        for r in rowsA:
            self.play(r.animate.set_color(ACCENT), run_time=0.1)
            self.play(r.animate.set_color(TXT), run_time=0.1)
        self.wait(0.6)

        # ---- column B ----
        colA = VGroup(hdrA, a1, valA, a2, rowsA, aa12, aa23, a3, a3_body)
        hdrB, b1, valB, b2, rowsB, ba12, ba23 = column(RX, "B", "torch.compile(f)(x)", "= dynamo_eval_frame", ACCENT)
        switch("DYNAMO", "接手", "torch.compile 把 eval_frame 指標換成 Dynamo 的 evaluator，同一個 frame 改送進 Dynamo")
        self.play(colA.animate.set_opacity(0.45), FadeIn(hdrB), FadeIn(b1), FadeIn(valB), b1[0].animate.set_stroke(ACCENT, width=2.5).set_fill(ACTIVE_FILL), run_time=0.5)
        self.play(GrowArrow(ba12), FadeIn(b2), FadeIn(rowsB), run_time=0.4)

        switch("DYNAMO", "符號執行", "Dynamo 不真的算，拿符號值走過每一條 bytecode，把 Tensor 運算記下來")
        for r in rowsB:
            self.play(r.animate.set_color(ACCENT), run_time=0.1)
            self.play(r.animate.set_color(TXT), run_time=0.1)

        b3 = titled(COL_W, H3, "Dynamo", "產物").move_to([RX, Y3, 0])
        m1 = mini("FX GRAPH", ["sin = sin(x)", "add = sin + 1"], w=1.85)
        m2 = mini("GUARDS", ["x: f32[8] cuda", "sin is same obj"], w=1.85)
        m3 = mini("NEW BYTECODE", ["LOAD_GLOBAL cfn_1", "CALL 1, RETURN"], w=1.85)
        minis = VGroup(m1, m2, m3).arrange(RIGHT, buff=0.12, aligned_edge=UP)
        minis.next_to(b3[1], DOWN, buff=0.18).align_to(b3[1], LEFT)
        self.play(GrowArrow(ba23), FadeIn(b3), run_time=0.4)
        for m, (n, z, c) in zip(minis, [("DYNAMO", "FX Graph", "走過的 Tensor 運算收成一張 FX Graph，交給後端編譯"), ("DYNAMO", "Guards", "同時記下這張圖成立的前提：輸入的 dtype、shape，torch.sin 還是同一個物件"), ("DYNAMO", "改寫 bytecode", "生成一份新的 bytecode：整段運算換成呼叫編好的函式")]):
            switch(n, z, c)
            self.play(FadeIn(m, shift=UP * 0.1), run_time=0.35)
            self.wait(0.7)

        # ---- hand back ----
        switch("CPYTHON", "執行新 bytecode", "新的 code object 交回 CPython 執行；下次 Guard 通過就直接重用，Dynamo 不再介入")
        rows_new = code_lines(BC_NEW, color=ACCENT).move_to(rowsB, aligned_edge=UL)
        self.play(m3[0].animate.set_stroke(ACCENT, width=2), run_time=0.25)
        self.play(FadeOut(rowsB), TransformFromCopy(m3[1][1], rows_new), run_time=0.6)
        self.play(m3[0].animate.set_stroke(EDGE, width=1.5), run_time=0.25)
        self.wait(3)
