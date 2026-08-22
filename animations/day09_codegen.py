from pathlib import Path

import manimpango
from manim import *

FONT_DIR = Path(__file__).parent / "fonts"
for f in FONT_DIR.glob("*.ttf"):
    manimpango.register_font(str(f))

BG = "#161719"
CARD = "#23272e"
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
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    body = T(f"{name}  ·  {zh}", font_size=17, font=SANS, color=BG, t2f={name: SANS, "·": MONO, zh: zh_font}, t2w={name: BOLD}, t2c={"·": "#666"})
    t = VGroup(Dot(radius=0.06, color=ACCENT), body).arrange(RIGHT, buff=0.15)
    bg = RoundedRectangle(corner_radius=0.3, width=t.width + 0.6, height=0.6, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def line(s, color=TXT, size=15):
    return T(s, font=MONO, font_size=size, color=color)


ORIG = [
    "LOAD_FAST     x",
    "LOAD_FAST     n",
    "BINARY_OP     *",
    "LOAD_CONST    1",
    "BINARY_OP     +",
    "RETURN_VALUE",
]

NEW = [
    ("0", "LOAD_GLOBAL   __compiled_fn_1"),
    ("10", "LOAD_FAST     x"),
    ("12", "CALL          1"),
    ("20", "STORE_FAST    graph_out_0"),
    ("22", "LOAD_FAST     graph_out_0"),
    ("24", "LOAD_CONST    0"),
    ("26", "BINARY_SUBSCR"),
    ("30", "DELETE_FAST   graph_out_0"),
    ("32", "RETURN_VALUE"),
]

OLX = -2.35
OY0, ODY = 1.6, 0.55
NY0, NDY = 2.05, 0.5


def slot(i):
    return [OLX, NY0 - i * NDY, 0]


class Codegen(Scene):
    def construct(self):
        title = T("f(x, n)  ->  x * n + 1", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("SETUP", "原 bytecode", "f 的六條 bytecode：兩條載入、三條計算、一條 return，改寫的原料就這些")
        olines = VGroup(*[line(s).move_to([OLX, OY0 - i * ODY, 0], aligned_edge=LEFT) for i, s in enumerate(ORIG)])
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in olines], lag_ratio=0.15), run_time=1.2)
        self.wait(2.5)

        switch("ABSORB", "計算被吸走", "乘與加整段收進編譯產物，bytecode 層從此不再有任何計算")
        cap_label = T("__compiled_fn_1", font=MONO, font_size=15, color=ACCENT)
        cap_box = RoundedRectangle(corner_radius=0.3, width=cap_label.width + 0.8, height=0.85, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD, fill_opacity=1)
        capsule = VGroup(cap_box, cap_label.move_to(cap_box)).move_to([3.6, OY0 - 3 * ODY, 0])
        self.play(FadeIn(capsule, shift=LEFT * 0.2), run_time=0.5)
        for k in (2, 3, 4):
            self.play(olines[k].animate.set_color(ACCENT), run_time=0.25)
            self.play(olines[k].animate.move_to(capsule.get_center()).scale(0.2).set_opacity(0), Indicate(capsule, scale_factor=1.04, color=ACCENT), run_time=0.6)
            self.remove(olines[k])
        self.wait(3.5)

        switch("BAKE", "n 變常數", "n 是 Python int，Day 5 已被 bake 進圖裡，圖的輸入只剩 x，這條不用留")
        bake = T("-> baked", font=MONO, font_size=12, color=ACCENT).next_to(olines[1], RIGHT, buff=0.3)
        self.play(olines[1].animate.set_color(DIM), FadeIn(bake), run_time=0.4)
        self.wait(1.2)
        self.play(FadeOut(olines[1]), FadeOut(bake), run_time=0.5)
        self.wait(2.5)

        switch("REWRITE", "原地改寫", "剩下兩條原指令原樣沿用，先排進新的位置，中間的洞交給新的搬運碼")
        self.play(olines[0].animate.move_to(slot(1), aligned_edge=LEFT).set_color(DIM), olines[5].animate.move_to(slot(8), aligned_edge=LEFT).set_color(DIM), run_time=0.9)
        self.wait(2)

        switch("STEP 1", "載入編譯產物", "__compiled_fn_1 是 Day 8 塞進 globals 的名字，一條 LOAD_GLOBAL 就載上 stack")
        n0 = line(NEW[0][1], color=ACCENT).move_to(slot(0), aligned_edge=LEFT)
        self.play(TransformFromCopy(cap_label, n0), run_time=0.8)
        self.wait(3.5)
        self.play(n0.animate.set_color(TXT), run_time=0.15)

        switch("STEP 2", "按 Source 擺輸入", "x 有 Source：L['x'].reconstruct() 從原位置載，不必讓圖多輸出一份")
        self.play(olines[0].animate.set_color(ACCENT), run_time=0.3)
        self.play(Indicate(olines[0], scale_factor=1.06, color=ACCENT), run_time=0.6)
        self.wait(3.5)
        self.play(olines[0].animate.set_color(TXT), run_time=0.15)

        switch("STEP 3", "一條 CALL 吃掉整段計算", "被吸走的三條運算指令，等價物就是這一條 CALL，計算全部發生在膠囊裡")
        n2 = line(NEW[2][1], color=ACCENT).move_to(slot(2), aligned_edge=LEFT)
        self.play(ReplacementTransform(capsule, n2), run_time=0.9)
        self.play(Flash(n2.get_left() + LEFT * 0.25, color=ACCENT, line_length=0.15, flash_radius=0.3), run_time=0.5)
        self.wait(3.5)
        self.play(n2.animate.set_color(TXT), run_time=0.15)

        switch("STEP 4", "拆輸出", "圖的輸出永遠是 tuple：存進 graph_out_0、取下標 0、用完 DELETE_FAST 歸還引用")
        unpack = VGroup(*[line(NEW[i][1], color=ACCENT).move_to(slot(i), aligned_edge=LEFT) for i in range(3, 8)])
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.25) for m in unpack], lag_ratio=0.2), run_time=1.4)
        self.wait(3.5)
        self.play(*[m.animate.set_color(TXT) for m in unpack], run_time=0.15)

        switch("STEP 5", "RETURN", "RETURN_VALUE 原樣收尾；有 side effect 的話，Day 7 的重播碼會插在它前面")
        self.play(olines[5].animate.set_color(ACCENT), run_time=0.3)
        self.play(Indicate(olines[5], scale_factor=1.06, color=ACCENT), run_time=0.6)
        self.wait(2.5)
        self.play(olines[5].animate.set_color(TXT), run_time=0.15)

        newcol = VGroup(n0, olines[0], n2, *unpack, olines[5])

        switch("ASSEMBLE", "組回 code object", "transform_code_object 組裝：offset 這時才算出來、EXTENDED_ARG 掃到收斂、stacksize 重算")
        offs = VGroup(*[T(NEW[i][0], font=MONO, font_size=12, color=ACCENT).move_to([OLX - 0.4, slot(i)[1], 0], aligned_edge=RIGHT).match_y(newcol[i]) for i in range(9)])
        self.play(LaggedStart(*[FadeIn(o, shift=RIGHT * 0.15) for o in offs], lag_ratio=0.12), run_time=1.4)
        self.play(offs.animate.set_color(MUTED), run_time=0.4)
        frame = RoundedRectangle(corner_radius=0.18, width=VGroup(newcol, offs).width + 0.7, height=VGroup(newcol, offs).height + 0.6, stroke_color=ACCENT, stroke_width=1.8, fill_opacity=0).move_to(VGroup(newcol, offs))
        tag = T("code object  ->  eval hook", font=MONO, font_size=13, color=MUTED).next_to(frame, DOWN, buff=0.25)
        self.play(Create(frame), run_time=0.8)
        self.play(FadeIn(tag, shift=UP * 0.1), run_time=0.3)
        self.wait(3.5)

        switch("RULE", "只生搬運碼", "值有 Source 從原位置載、是圖輸出從 graph_out_0 取、常數直接 LOAD_CONST，能省的絕不多生")
        self.wait(5.5)
