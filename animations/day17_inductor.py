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


def mixed(s, size, color=TXT, font=MONO):
    t2f = {ch: CJK for ch in s if "一" <= ch <= "鿿"}
    return T(s, font=font, font_size=size, color=color, t2f=t2f)


def card(tag, lines, w=6.8, tag_color=ACCENT, size=16):
    tg = mixed(tag, 16, color=tag_color)
    body = VGroup()
    indents = []
    for s in lines:
        stripped = s.lstrip("  ")
        indents.append(len(s) - len(stripped))
        body.add(mixed(stripped, size))
    body.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    for m, ind in zip(body, indents):
        m.shift(RIGHT * 0.17 * ind)
    inner = VGroup(tg, body).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
    box = RoundedRectangle(corner_radius=0.16, width=w, height=inner.height + 0.6, stroke_color=EDGE, stroke_width=1.6, fill_color=CARD, fill_opacity=1)
    inner.move_to(box).align_to(box.get_left() + RIGHT * 0.4, LEFT)
    g = VGroup(box, inner)
    g.set_z_index(2)
    return g


def gate(label):
    t = mixed(label, 17, color=TXT)
    bar = RoundedRectangle(corner_radius=0.12, width=9.2, height=0.66, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD_DIM, fill_opacity=1)
    g = VGroup(bar, t.move_to(bar))
    g.set_z_index(0)
    return g


def chip(s, edge=EDGE, color=TXT, fill=CARD_DIM):
    t = mixed(s, 16, color=color)
    r = RoundedRectangle(corner_radius=0.24, width=t.width + 0.55, height=0.58, stroke_color=edge, stroke_width=1.5, fill_color=fill, fill_opacity=1)
    return VGroup(r, t.move_to(r))


CX = 0.6
CY = 0.25


class Inductor(Scene):
    def construct(self):
        title = T("f(x, y)  ->  relu(x + y) * 2", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = mixed(caption, 19).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        steps = VGroup(*[T(s, font=MONO, font_size=16, color=DIM) for s in ("ATen", "loop IR", "fused", "kernel")]).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([-5.9, CY, 0], aligned_edge=LEFT)
        rail = Line(steps.get_top() + UP * 0.25, steps.get_bottom() + DOWN * 0.25, stroke_color=EDGE, stroke_width=2).next_to(steps, LEFT, buff=0.25)

        def light(i):
            self.play(*[s.animate.set_color(ACCENT if j == i else (TXT if j < i else DIM)) for j, s in enumerate(steps)], run_time=0.3)

        def drop(g, morph):
            g.move_to([CX, -3.4, 0])
            self.add(g)
            self.play(g.animate(rate_func=linear).move_to([CX, CY, 0]), run_time=0.7)
            self.play(morph, Flash([CX, CY, 0], color=ACCENT, line_length=0.2, flash_radius=1.1), run_time=0.9)
            self.play(g.animate(rate_func=linear).move_to([CX, 3.6, 0]), run_time=0.7)
            self.remove(g)

        switch("INPUT", "ATen 圖", "AOTAutograd 交出的成品：純函數式、shape 齊全的 ATen 圖，Inductor 從這裡接手")
        a = card("fx graph · ATen", [
            "add  = aten.add(x, y)",
            "relu = aten.relu(add)",
            "mul  = aten.mul(relu, 2)",
        ]).move_to([CX, 3.8, 0])
        self.play(FadeIn(steps), FadeIn(rail), run_time=0.4)
        self.play(a.animate.move_to([CX, CY, 0]), run_time=0.9)
        light(0)
        self.wait(3.5)

        switch("LOWER", "查表攤開", "GraphLowering 逐 node 查 lowering 表，每個 op 攤開成描述迴圈內容的 body 函式")
        b = card("loop-level IR · define-by-run", [
            "def body(ops):",
            "  a = ops.load('x', i)",
            "  b = ops.load('y', i)",
            "  t = ops.relu(ops.add(a, b))",
            "  m = ops.mul(t, 2.0)",
            "  ops.store('buf0', i, m)",
        ]).move_to([CX, CY, 0])
        drop(gate("LOWER · GraphLowering.run()"), ReplacementTransform(a, b))
        light(1)
        self.wait(3.5)

        switch("IR", "顯式讀寫", "讀了誰、寫了誰、迴圈多長，全部攤在檯面上，排程需要的資訊都齊了")
        loads = VGroup(b[1][1][1], b[1][1][2], b[1][1][5])
        self.play(loads.animate.set_color(ACCENT), run_time=0.3)
        self.play(Indicate(b[1][1][5], scale_factor=1.05, color=ACCENT), run_time=0.6)
        self.wait(2.5)
        self.play(loads.animate.set_color(TXT), run_time=0.2)

        switch("SCHEDULE", "決定融合", "Scheduler 建依賴、排順序：三個 op 讀寫相連、迭代空間相同，融成一組")
        c = card("SchedulerNode · op0", [
            "fused: add + relu + mul",
            "reads : x, y",
            "writes: buf0",
            "iteration: (1024,)",
        ]).move_to([CX, CY, 0])
        drop(gate("SCHEDULE · Scheduler"), ReplacementTransform(b, c))
        light(2)
        self.wait(3.5)

        switch("CODEGEN", "按裝置分流", "同一個 node 交給不同生成器：GPU 生 Triton kernel，CPU 生 C++，今天走 CPU")
        gpu = chip("GPU -> Triton", color=DIM).move_to([-3.2, -2.35, 0])
        cpu = chip("CPU -> C++ / OpenMP", edge=ACCENT, color=TXT).move_to([3.2, -2.35, 0])
        self.play(FadeIn(gpu, shift=UP * 0.1), FadeIn(cpu, shift=UP * 0.1), run_time=0.4)
        self.play(Indicate(cpu, scale_factor=1.05, color=ACCENT), run_time=0.6)
        self.wait(2.5)

        switch("KERNEL", "一個迴圈", "add、relu、mul 進了同一個迴圈：讀兩次、寫一次，名字就叫 fused")
        d = card("cpp_fused_add_mul_relu_0", [
            "for(x0=0; x0<1024; x0+=4){",
            "  tmp2 = tmp0 + tmp1;",
            "  tmp3 = clamp_min(tmp2, 0);",
            "  tmp6 = tmp3 * 2.0;",
            "  tmp6.store(out_ptr0 + x0);",
            "}",
        ]).move_to([CX, CY, 0])
        drop(gate("CODEGEN · cpp.py"), ReplacementTransform(c, d))
        light(3)
        self.play(FadeOut(gpu), FadeOut(cpu), run_time=0.3)
        self.wait(3.5)

        switch("WRAPPER", "組裝出貨", "wrapper 配好 buffer、按順序呼叫 kernel，組成 call()，交還給改寫後的 bytecode")
        self.play(d.animate.move_to([CX, 1.35, 0]), run_time=0.6)
        w = card("wrapper · call(args)", [
            "buf0 = empty_strided_cpu((1024,))",
            "cpp_fused_add_mul_relu_0(x, y, buf0)",
            "return (buf0,)",
        ]).move_to([CX, -1.35, 0])
        ar = Arrow(d.get_bottom(), w.get_top(), buff=0.12, color=ACCENT, stroke_width=2.5, tip_length=0.16)
        self.play(FadeIn(w, shift=UP * 0.2), GrowArrow(ar), run_time=0.7)
        self.wait(2.5)

        switch("RULE", "一條產線", "lower 攤開、schedule 融合、codegen 分流：FX Graph 進來，kernel 出去")
        self.play(FadeOut(d), FadeOut(w), FadeOut(ar), FadeOut(steps), FadeOut(rail), run_time=0.4)
        row = VGroup(chip("ATen 圖"), chip("loop IR"), chip("op0 · fused"), chip("cpp kernel", edge=ACCENT))
        arrows = VGroup()
        labels = VGroup()
        row.arrange(RIGHT, buff=1.05).move_to([0, CY, 0])
        for i, name in enumerate(("LOWER", "SCHEDULE", "CODEGEN")):
            ar2 = Arrow(row[i].get_right(), row[i + 1].get_left(), buff=0.1, color=ACCENT, stroke_width=2.5, tip_length=0.16)
            arrows.add(ar2)
            labels.add(T(name, font=MONO, font_size=16, color=MUTED).next_to(ar2, DOWN, buff=0.22))
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.15) for m in (row[0], arrows[0], row[1], arrows[1], row[2], arrows[2], row[3])], lag_ratio=0.15), run_time=1.4)
        self.play(FadeIn(labels), run_time=0.4)
        self.wait(5.5)
