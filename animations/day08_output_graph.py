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


def mini(title, lines, w, active=False, ghost=False):
    body = rows(lines, size=11, buff=0.08, color=DIM if ghost else TXT)
    fill = CARD_DIM if ghost else (ACTIVE_FILL if active else CARD_DIM)
    hd_c = DIM if ghost else (ACCENT if active else MUTED)
    r = panel(w, body.height + 0.62, fill=fill, edge=ACCENT if active and not ghost else EDGE, r=0.08, sw=2 if active and not ghost else 1.5)
    hd = T(title, font=MONO, font_size=11, color=hd_c)
    inner = VGroup(hd, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, inner)


def arrow(a, b, color=MUTED, w=1.8):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


def node(txt, w=1.4, kind="op"):
    fill = ACTIVE_FILL if kind == "op" else CARD_DIM
    edge = ACCENT if kind == "op" else EDGE
    r = panel(w, 0.46, fill=fill, edge=edge, r=0.08, sw=1.5)
    t = T(txt, font=MONO, font_size=11, color=TXT if kind == "op" else MUTED).move_to(r)
    return VGroup(r, t)


def ev(main, sub, w=3.0, color=TXT):
    m = T(main, font=MONO, font_size=12, color=color)
    s = T(sub, font=CJK, font_size=10, color=MUTED if color is TXT else DIM)
    body = VGroup(m, s).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
    r = panel(w, body.height + 0.3, fill=CARD_DIM, edge=EDGE, r=0.08)
    body.move_to(r).align_to(r.get_left() + RIGHT * 0.15, LEFT)
    return VGroup(r, body)


def shelf(w, h, name):
    r = panel(w, h, fill=CARD_DIM, edge=EDGE, r=0.08)
    hd = T(name, font=MONO, font_size=10, color=MUTED).move_to(r.get_corner(UL) + RIGHT * 0.15 + DOWN * 0.15, aligned_edge=UL)
    return VGroup(r, hd)


class OG(Scene):
    def construct(self):
        title = T("f(x, y, unused)  ->  (x @ y + bias).relu()", font=MONO, font_size=22, color=TXT).to_corner(UL, buff=0.5)
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
        MID = (TOP + BOT) / 2
        trace = titled(3.5, H, "TRACE", "逐條翻譯").move_to([-5.1, MID, 0])
        og = titled(5.6, H, "OutputGraph", "倉庫", fill=CARD, edge=ACCENT, sw=2).move_to([0, MID, 0])
        lbl_l = label("PRODUCER  ·  symbolic_convert").next_to(trace, UP, buff=0.22).align_to(trace, LEFT)
        lbl_c = label("STATE  ·  正在長大").next_to(og, UP, buff=0.22).align_to(og, LEFT)

        events = VGroup(
            ev("x @ y", "matmul 進圖"),
            ev("+ bias", "外面抓進來的 Tensor"),
            ev(".relu()", "method 呼叫進圖"),
            ev("RETURN", "翻完了，該收圖"),
            ev("unused", "從頭到尾沒被用到", color=DIM),
        ).arrange(DOWN, buff=0.2).next_to(trace[1], DOWN, buff=0.3).align_to(trace[1], LEFT)

        g_shelf = shelf(2.3, 1.3, "guards").move_to([-1.3, -2.05, 0])
        s_shelf = shelf(2.3, 1.3, "side_effects").move_to([1.3, -2.05, 0])
        s_row = T("(empty)", font=MONO, font_size=9, color=DIM).next_to(s_shelf[1], DOWN, buff=0.08).align_to(s_shelf[1], LEFT)

        ghosts = VGroup(
            mini("call_user_compiler", ["GraphModule -> backend", "拿回 compiled_fn"], 3.5, ghost=True),
            mini("install_global", ["globals['__compiled_fn_1']"], 3.5, ghost=True),
            mini("CheckFunctionManager", ["guards -> C++ tree"], 3.5, ghost=True),
        ).arrange(DOWN, buff=0.5).move_to([5.15, MID + 0.1, 0])
        lbl_r = label("COMPILE_SUBGRAPH  ·  收圖時").next_to(ghosts, UP, buff=0.22).align_to(ghosts, LEFT)

        switch("SETUP", "一個 frame 一張紙", "一個 frame 的一次編譯只有一個 OutputGraph，環境 Guard 在 init_ambient_guards 出生就裝好")
        self.play(FadeIn(trace), FadeIn(og), FadeIn(lbl_l), FadeIn(lbl_c), FadeIn(events), FadeIn(lbl_r), FadeIn(ghosts), run_time=0.5)
        self.play(FadeIn(g_shelf), FadeIn(s_shelf), FadeIn(s_row), run_time=0.4)
        g_rows = [T("ambient x5 (init)", font=MONO, font_size=9, color=TXT).next_to(g_shelf[1], DOWN, buff=0.08).align_to(g_shelf[1], LEFT)]
        self.play(FadeIn(g_rows[0], shift=LEFT * 0.1), run_time=0.3)
        self.wait(2.5)

        hi = [None]

        def focus(e):
            anims = [e[0].animate.set_stroke(color=ACCENT, width=2).set_fill(ACTIVE_FILL)]
            if hi[0] is not None:
                anims.append(hi[0][0].animate.set_stroke(color=EDGE, width=1.5).set_fill(CARD_DIM))
            self.play(*anims, run_time=0.25)
            hi[0] = e

        def fly(n, src, dst):
            n.move_to(dst)
            n.save_state()
            n.scale(0.4).move_to(src)
            self.play(Restore(n), run_time=0.5)

        def add_guard(txt):
            r = T(txt, font=MONO, font_size=9, color=TXT).next_to(g_rows[-1], DOWN, buff=0.08).align_to(g_rows[-1], LEFT)
            g_rows.append(r)
            self.play(FadeIn(r, shift=LEFT * 0.1), run_time=0.25)

        ph_y = 1.35
        n_lx = node("L_x_", kind="ph")
        n_ly = node("L_y_", kind="ph")
        n_lb = node("L_bias_", kind="ph")
        n_mm = node("matmul")
        n_ad = node("add")
        n_re = node("relu")

        switch("STEP 1", "x @ y", "輸入用到才登記：x 與 y 這一刻才 create_graph_input，matmul 節點帶著出生證明進圖")
        focus(events[0])
        src1 = events[0].get_right()
        fly(n_lx, src1, [-1.7, ph_y, 0])
        fly(n_ly, src1, [0.0, ph_y, 0])
        add_guard("TENSOR_MATCH L['x']")
        add_guard("TENSOR_MATCH L['y']")
        fly(n_mm, src1, [-0.85, 0.6, 0])
        e1 = arrow(n_lx.get_bottom(), n_mm.get_top())
        e2 = arrow(n_ly.get_bottom(), n_mm.get_top())
        self.play(GrowArrow(e1), GrowArrow(e2), run_time=0.35)
        self.wait(3.5)

        switch("STEP 2", "+ bias", "bias 不是參數，是外面抓進來的 Tensor：一樣 lift 成圖的輸入，值永遠不 bake")
        focus(events[1])
        src2 = events[1].get_right()
        fly(n_lb, src2, [1.7, ph_y, 0])
        add_guard("TENSOR_MATCH L['bias']")
        fly(n_ad, src2, [0.0, -0.15, 0])
        e3 = arrow(n_mm.get_bottom(), n_ad.get_top())
        e4 = arrow(n_lb.get_bottom(), n_ad.get_top())
        self.play(GrowArrow(e3), GrowArrow(e4), run_time=0.35)
        self.wait(3.5)

        switch("STEP 3", ".relu()", "unused 從頭到尾沒被用到，永遠不會有 placeholder：圖的輸入不看函式簽名")
        focus(events[2])
        fly(n_re, events[2].get_right(), [-0.85, -0.9, 0])
        e5 = arrow(n_ad.get_bottom(), n_re.get_top())
        self.play(GrowArrow(e5), run_time=0.35)
        self.wait(3.5)

        switch("RETURN", "收圖時機", "收圖時機只有兩種：RETURN 或 Graph Break，殊途同歸走進 compile_subgraph")
        focus(events[3])
        self.wait(2)

        switch("COMPILE 1", "接上 output", "活著的值接上 output 節點，remove_unused_graphargs 再清一輪沒用到的輸入")
        n_out = node("output")
        fly(n_out, events[3].get_right(), [0.85, -0.9, 0])
        e6 = arrow(n_re.get_right(), n_out.get_left())
        self.play(GrowArrow(e6), run_time=0.35)
        self.wait(2.5)

        dag = VGroup(n_lx, n_ly, n_lb, n_mm, n_ad, n_re, n_out, e1, e2, e3, e4, e5, e6)
        gm = VGroup(panel(2.3, 0.6, fill=ACTIVE_FILL, edge=ACCENT, r=0.1, sw=2), T("GraphModule", font=MONO, font_size=13, color=TXT))
        gm[1].move_to(gm[0])
        gm.move_to([0, 0.2, 0])

        outs = VGroup(
            mini("call_user_compiler", ["GraphModule -> backend", "拿回 compiled_fn"], 3.5, active=True),
            mini("install_global", ["globals['__compiled_fn_1']"], 3.5, active=True),
            mini("CheckFunctionManager", ["guards -> C++ tree"], 3.5),
        )
        for i in range(3):
            outs[i].move_to(ghosts[i])

        switch("COMPILE 2", "call_user_compiler", "整張圖包成 GraphModule 交給後端，拿回 compiled_fn；空圖直接跳過這一步")
        self.play(ReplacementTransform(dag, gm), run_time=0.7)
        self.play(FadeOut(ghosts[0]), FadeIn(outs[0]), run_time=0.4)
        a1 = arrow([og[0].get_right()[0], outs[0].get_left()[1], 0], outs[0].get_left(), color=ACCENT)
        self.play(GrowArrow(a1), run_time=0.35)
        self.wait(3.5)

        switch("COMPILE 3", "install_global", "編譯結果以 __compiled_fn_1 塞進 globals，Guard 交給 CheckFunctionManager 編成 C++ 樹")
        self.play(FadeOut(ghosts[1]), FadeOut(ghosts[2]), FadeIn(outs[1]), FadeIn(outs[2]), run_time=0.4)
        a2 = arrow([og[0].get_right()[0], outs[1].get_left()[1], 0], outs[1].get_left(), color=ACCENT)
        a3 = arrow([og[0].get_right()[0], outs[2].get_left()[1], 0], outs[2].get_left(), color=MUTED)
        self.play(GrowArrow(a2), GrowArrow(a3), run_time=0.35)
        self.wait(3.5)

        switch("RULE", "散落的產出收成一張圖", "翻譯期逐筆累積、收圖時一次收攏：明天輪到 PyCodegen 寫出呼叫它的新 bytecode")
        self.wait(5.5)
