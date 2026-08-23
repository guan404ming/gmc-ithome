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
    zh_font = CJK if any("一" <= ch <= "鿿" for ch in zh) else MONO
    nm = T(name, font=SANS, font_size=17, weight=BOLD, color=BG)
    sep = T("·", font=MONO, font_size=17, color="#666")
    zt = T(zh, font=zh_font, font_size=17, color=BG)
    t = VGroup(Dot(radius=0.06, color=ACCENT), nm, sep, zt).arrange(RIGHT, buff=0.18)
    bg = RoundedRectangle(corner_radius=0.26, width=t.width + 0.6, height=0.52, stroke_width=0, fill_color="#eceae6", fill_opacity=1)
    return VGroup(bg, t.move_to(bg))


def launch_block(w=0.52):
    return Rectangle(width=w, height=0.34, stroke_width=0, fill_color=ACCENT, fill_opacity=1)


def kernel_block(w=0.3):
    return Rectangle(width=w, height=0.34, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD, fill_opacity=1)


CPU_Y = 1.5
GPU_Y = 0.3
BAR_Y = -1.05
X0 = -5.35
STEP = 1.5
DELAY = 0.62
N = 6


class CudaGraph(Scene):
    def construct(self):
        title = T("32 x Linear(256, 256) · batch=8", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)
        cur = [None, None]

        def switch(name, zh, caption):
            p = pill(name, zh).to_edge(RIGHT, buff=0.5).match_y(title)
            c = T(caption, font=CJK, font_size=19, color=TXT).to_edge(DOWN, buff=0.45)
            if cur[0] is not None:
                self.play(FadeOut(cur[0]), FadeOut(cur[1]), run_time=0.2)
            self.play(FadeIn(p), FadeIn(c, shift=UP * 0.1), run_time=0.3)
            cur[0], cur[1] = p, c

        switch("LAUNCH", "逐次發射", "eager 下每個 kernel 都由 CPU 各發射一次，每次發射都付一段固定的 overhead")
        cpu_lab = T("CPU", font=MONO, font_size=17, color=MUTED).move_to([-6.35, CPU_Y, 0])
        gpu_lab = T("GPU", font=MONO, font_size=17, color=MUTED).move_to([-6.35, GPU_Y, 0])
        cpu_line = Line([-5.8, CPU_Y - 0.32, 0], [4.4, CPU_Y - 0.32, 0], stroke_color=EDGE, stroke_width=1.6)
        gpu_line = Line([-5.8, GPU_Y - 0.32, 0], [4.4, GPU_Y - 0.32, 0], stroke_color=EDGE, stroke_width=1.6)
        self.play(FadeIn(cpu_lab), FadeIn(gpu_lab), Create(cpu_line), Create(gpu_line), run_time=0.6)

        launches, kernels, links = VGroup(), VGroup(), VGroup()
        for i in range(N):
            lx = X0 + i * STEP
            kx = X0 + i * STEP + DELAY
            lb = launch_block().move_to([lx + 0.26, CPU_Y, 0])
            kb = kernel_block().move_to([kx + 0.15, GPU_Y, 0])
            ln = DashedLine([lx + 0.26, CPU_Y - 0.17, 0], [kx + 0.15, GPU_Y + 0.17, 0], stroke_color=DIM, stroke_width=1.6, dash_length=0.08)
            launches.add(lb)
            kernels.add(kb)
            links.add(ln)
        for i in range(N):
            self.play(FadeIn(launches[i], shift=DOWN * 0.08), Create(links[i]), FadeIn(kernels[i], scale=0.8), run_time=0.32)
        ltag = T("launch overhead", font=MONO, font_size=16, color=ACCENT).next_to(launches[1], UP, buff=0.22)
        self.play(FadeIn(ltag, shift=UP * 0.08), run_time=0.3)
        self.wait(3)

        switch("STARVE", "空轉的 GPU", "kernel 本身只跑幾 us，發射卻要更久，GPU 軌道上大半是空隙，都在等 CPU")
        idles = VGroup()
        for i in (1, 3):
            a = kernels[i].get_right()[0]
            b = kernels[i + 1].get_left()[0]
            seg = Line([a + 0.06, GPU_Y, 0], [b - 0.06, GPU_Y, 0], stroke_color=ACCENT, stroke_width=2.2)
            lab = T("idle", font=MONO, font_size=16, color=ACCENT).next_to(seg, DOWN, buff=0.16)
            idles.add(VGroup(seg, lab))
        self.play(*[FadeIn(g) for g in idles], run_time=0.5)
        bar1 = Line([X0, BAR_Y, 0], [X0 + (N - 1) * STEP + DELAY + 0.3, BAR_Y, 0], stroke_color=MUTED, stroke_width=3)
        b1lab = T("one step · 1.124 ms", font=MONO, font_size=16, color=MUTED).next_to(bar1, DOWN, buff=0.2)
        self.play(Create(bar1), FadeIn(b1lab, shift=UP * 0.08), run_time=0.6)
        self.wait(3.5)

        switch("CAPTURE", "錄成一張圖", "torch.cuda.CUDAGraph 側錄整串發射，kernel 的順序、參數、位址全部固定下來")
        rec = RoundedRectangle(corner_radius=0.14, width=3.6, height=1.1, stroke_color=ACCENT, stroke_width=1.8, fill_color=CARD_DIM, fill_opacity=1).move_to([1.6, BAR_Y - 1.05, 0])
        rlab = T("CUDAGraph", font=MONO, font_size=17, color=TXT).next_to(rec, LEFT, buff=0.35)
        rdot = Dot(radius=0.07, color=ACCENT).move_to(rec.get_corner(UR) + [-0.25, -0.25, 0])
        self.play(FadeOut(ltag), FadeOut(idles), Create(rec), FadeIn(rlab), FadeIn(rdot), run_time=0.7)
        packed = VGroup(*[kernel_block(0.42).set_stroke(ACCENT, 1.2) for _ in range(N)]).arrange(RIGHT, buff=0.08).move_to(rec)
        ghosts = VGroup(*[k.copy() for k in kernels])
        self.play(ReplacementTransform(ghosts, packed), FadeOut(links), launches.animate.set_fill(opacity=0.25), run_time=1.1)
        self.play(Flash(rdot, color=ACCENT, line_length=0.12, flash_radius=0.3), run_time=0.4)
        self.wait(3.5)

        switch("REPLAY", "一鍵重播", "之後每一步 CPU 只按一次 replay，整串 kernel 原樣重播，中間再沒有 CPU 的事")
        self.play(FadeOut(launches), FadeOut(kernels), FadeOut(bar1), FadeOut(b1lab), run_time=0.5)
        rp = launch_block(0.52).move_to([X0 + 0.26, CPU_Y, 0])
        rplab = T("replay()", font=MONO, font_size=16, color=ACCENT).next_to(rp, UP, buff=0.2)
        tight = VGroup(*[kernel_block() for _ in range(N)]).arrange(RIGHT, buff=0.07)
        tight.move_to([X0 + DELAY + tight.width / 2, GPU_Y, 0])
        rlink = DashedLine([X0 + 0.26, CPU_Y - 0.17, 0], [tight[0].get_left()[0] + 0.1, GPU_Y + 0.24, 0], stroke_color=DIM, stroke_width=1.6, dash_length=0.08)
        self.play(FadeIn(rp, shift=DOWN * 0.08), FadeIn(rplab), Create(rlink), run_time=0.5)
        echo = packed.copy()
        self.play(ReplacementTransform(echo, tight), run_time=0.9)
        bar2 = Line([X0, BAR_Y, 0], [X0 + DELAY + tight.width + 0.1, BAR_Y, 0], stroke_color=ACCENT, stroke_width=3)
        b2lab = T("one step · 0.119 ms", font=MONO, font_size=16, color=ACCENT).next_to(bar2, DOWN, buff=0.2).align_to(bar2, LEFT)
        self.play(Create(bar2), FadeIn(b2lab, shift=UP * 0.08), run_time=0.6)
        self.wait(3.5)

        switch("STATIC", "位址押死", "代價是位址全是死的，新資料得先 copy_ 進固定的 input buffer，輸出也活在專屬 memory pool")
        xin = VGroup()
        xt = T("x_new", font=MONO, font_size=16, color=TXT)
        xbg = RoundedRectangle(corner_radius=0.12, width=xt.width + 0.4, height=0.46, stroke_color=EDGE, stroke_width=1.2, fill_color=CARD, fill_opacity=1)
        xin.add(xbg, xt.move_to(xbg))
        xin.move_to([-4.3, BAR_Y - 1.05, 0])
        slot = RoundedRectangle(corner_radius=0.12, width=1.6, height=0.5, stroke_color=ACCENT, stroke_width=1.4, fill_color=CARD_DIM, fill_opacity=1).next_to(rec, LEFT, buff=1.15)
        slab = T("static x", font=MONO, font_size=16, color=MUTED).move_to(slot)
        carrow = Arrow(xin.get_right(), slot.get_left(), stroke_color=DIM, stroke_width=2.4, buff=0.12, max_tip_length_to_length_ratio=0.12)
        clab = T("copy_", font=MONO, font_size=16, color=MUTED).next_to(carrow, UP, buff=0.12)
        self.play(FadeOut(rlab), FadeIn(xin), FadeIn(slot), FadeIn(slab), run_time=0.5)
        self.play(Create(carrow), FadeIn(clab), run_time=0.5)
        self.play(xin.animate.move_to(slot).set_opacity(0), Indicate(slot, scale_factor=1.1, color=ACCENT), run_time=0.7)
        self.remove(xin)
        self.wait(3.5)

        switch("RESULT", "省的是 overhead", "batch 8 差 9.45 倍，batch 8192 只剩 1.01 倍，kernel 一大 overhead 就被淹沒，賺頭全在小 kernel 密集的模型")
        keep = VGroup(cpu_lab, gpu_lab, cpu_line, gpu_line, rp, rplab, rlink, tight, bar2, b2lab, rec, rdot, packed, slot, slab, carrow, clab)
        self.play(FadeOut(keep), run_time=0.5)
        rows = VGroup()
        data = [("eager", 1.124, MUTED), ("compile default", 0.960, MUTED), ("reduce-overhead", 0.119, ACCENT)]
        ys = [1.6, 0.75, -0.1]
        for (name, ms, col), y in zip(data, ys):
            lab = T(name, font=MONO, font_size=17, color=TXT)
            lab.move_to([-2.5 - lab.width / 2, y, 0])
            bar = Rectangle(width=5.2 * ms / 1.124, height=0.3, stroke_width=0, fill_color=col, fill_opacity=1)
            bar.move_to([-2.1 + bar.width / 2, y, 0])
            val = T(f"{ms:.3f} ms", font=MONO, font_size=17, color=col).next_to(bar, RIGHT, buff=0.3)
            rows.add(VGroup(lab, bar, val))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.12) for r in rows], lag_ratio=0.25), run_time=1.2)
        sp = T("9.45x", font=MONO, font_size=30, color=ACCENT)
        sp.move_to([-2.1 + sp.width / 2, -1.05, 0])
        self.play(FadeIn(sp, scale=0.8), run_time=0.4)
        self.play(Flash(sp, color=ACCENT, line_length=0.16, flash_radius=0.7), run_time=0.5)
        big = T("batch=8192  1.585 -> 1.567 ms · 1.01x", font=MONO, font_size=17, color=MUTED).next_to(sp, DOWN, buff=0.45).align_to(sp, LEFT)
        self.play(FadeIn(big, shift=UP * 0.08), run_time=0.4)
        self.wait(5.5)
