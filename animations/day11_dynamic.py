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


def lock_icon(color=MUTED):
    body = RoundedRectangle(corner_radius=0.05, width=0.34, height=0.26, stroke_width=0, fill_color=color, fill_opacity=1)
    sh = Arc(radius=0.1, angle=PI, stroke_color=color, stroke_width=3)
    sh.next_to(body, UP, buff=-0.03)
    return VGroup(sh, body)


def guard_card(lines):
    lg = VGroup(*[T(s, font=MONO, font_size=15, color=TXT) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    inner = VGroup(lock_icon(), lg).arrange(RIGHT, buff=0.32)
    box = RoundedRectangle(corner_radius=0.14, width=inner.width + 0.75, height=inner.height + 0.55, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    return VGroup(box, inner.move_to(box))


def ball(shape):
    t = T(shape, font=MONO, font_size=15, color=TXT)
    bg = RoundedRectangle(corner_radius=0.25, width=t.width + 0.5, height=0.52, stroke_color=EDGE, stroke_width=1.5, fill_color=CARD_DIM, fill_opacity=1)
    g = VGroup(bg, t.move_to(bg))
    g.set_z_index(-1)
    return g


BALL_Y = -0.55
TOKEN_Y = 1.35


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

        switch("CALL 1", "第一次呼叫", "(4, 4) 第一次進來，快取是空的：assume_static_by_default，shape 按具體值押死")
        b1 = ball("(4, 4)").move_to([-8, BALL_Y, 0])
        self.play(b1.animate.move_to([-3.3, BALL_Y, 0]), run_time=0.9)
        tag1 = T("no entry -> COMPILE #1", font=MONO, font_size=16, color=ACCENT).next_to(b1, UP, buff=0.3)
        self.play(FadeIn(tag1, shift=UP * 0.1), run_time=0.3)
        self.wait(1)
        token = T("x = (4, 4)", font=MONO, font_size=34, color=TXT).move_to([0, TOKEN_Y, 0])
        token[0].set_color(MUTED)
        token[1].set_color(MUTED)
        glabel = T("graph #1 · static", font=MONO, font_size=16, color=MUTED).next_to(token, UP, buff=0.32)
        self.play(ReplacementTransform(b1, token), FadeOut(tag1), run_time=0.8)
        self.play(FadeIn(glabel, shift=UP * 0.08), run_time=0.3)
        card = guard_card(["TENSOR_MATCH [4, 4]"]).move_to([0, BALL_Y, 0])
        seed = VGroup(token[3].copy(), token[5].copy())
        self.play(FadeIn(card[0]), FadeIn(card[1][0]), Transform(seed, card[1][1][0]), run_time=0.9)
        self.remove(seed)
        self.add(card[1][1][0])
        fs = T("frame_state   x: dim0 = 4  dim1 = 4", font=MONO, font_size=16, color=DIM).next_to(card, DOWN, buff=0.45)
        self.play(FadeIn(fs, shift=UP * 0.08), run_time=0.3)
        self.wait(3.5)

        l1 = card[1][1][0]
        switch("GUARD", "押死的量尺", "編出來的圖把 batch 押死在 4：TENSOR_MATCH 量到 4 就是 4，差一格都不行")
        self.play(l1[12:17].animate.set_color(ACCENT), token[3].animate.set_color(ACCENT), run_time=0.3)
        self.play(Indicate(l1[13], scale_factor=1.3, color=ACCENT), run_time=0.6)
        self.wait(2.5)
        self.play(l1[12:17].animate.set_color(TXT), token[3].animate.set_color(TXT), run_time=0.2)

        switch("CALL 2", "賭輸", "batch 換成 8：驗票時 4 對不上 8，TENSOR_MATCH 失敗，Guard 亮紅")
        b2 = ball("(8, 4)").move_to([-8, BALL_Y, 0])
        stop2 = card.get_left()[0] - b2.width / 2 - 0.4
        self.play(b2.animate.move_to([stop2, BALL_Y, 0]), run_time=0.9)
        self.play(b2[1][1].animate.set_color(ACCENT), l1[13].animate.set_color(ACCENT), run_time=0.3)
        self.play(b2.animate(rate_func=there_and_back).shift(RIGHT * 0.25), run_time=0.4)
        cross = VGroup(Line(UL * 0.13, DR * 0.13), Line(UR * 0.13, DL * 0.13)).set_stroke(ACCENT, 3.5).move_to(l1[13])
        lock = card[1][0]
        mm = VGroup(T("size mismatch at index 0", font=MONO, font_size=16, color=ACCENT), T("expected 4, actual 8", font=MONO, font_size=16, color=ACCENT)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(card, RIGHT, buff=0.45)
        self.play(card[0].animate.set_stroke(ACCENT, 2.2), lock[0].animate.set_stroke(ACCENT, 3), lock[1].animate.set_fill(ACCENT), FadeIn(cross), Flash(l1[13], color=ACCENT, line_length=0.14, flash_radius=0.32), run_time=0.5)
        self.play(FadeIn(mm, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(3.5)

        switch("FRAME_STATE", "翻小冊子", "重編前 _automatic_dynamic 先翻 frame_state：dim 0 上次 4 這次 8，它會變")
        fsm = T("frame_state   x: dim0 = 4 -> 8", font=MONO, font_size=16, color=MUTED)
        fsm[18:22].set_color(ACCENT)
        fsz = T("會變", font=CJK, font_size=16, color=ACCENT)
        fs2 = VGroup(fsm, fsz).arrange(RIGHT, buff=0.25).move_to(fs)
        self.play(ReplacementTransform(fs, fs2), FadeOut(mm), run_time=0.5)
        self.play(Indicate(fs2, scale_factor=1.05, color=ACCENT), run_time=0.6)
        self.wait(2.5)

        switch("MORPH", "4 變身 s0", "Dynamo 不再押死：dim 0 的 4 當場變身成符號 s0，吃下這個維度的所有值")
        ghost = token.copy().set_color(DIM)
        self.add(ghost)
        s0t = T("s0", font=MONO, font_size=34, color=ACCENT).move_to(token[3])
        dx = s0t.width - token[3].width
        self.play(Wiggle(token[3], scale_value=1.3, rotation_angle=0.04 * TAU), run_time=0.9)
        gl2 = T("graph #2 · dynamic", font=MONO, font_size=16, color=MUTED).move_to(glabel)
        self.play(Transform(token[3], s0t, path_arc=PI / 2), VGroup(token[0], token[1], token[2]).animate.shift(LEFT * dx / 2), VGroup(token[4], token[5], token[6]).animate.shift(RIGHT * dx / 2), FadeOut(ghost, shift=UP * 0.7, scale=0.85), ReplacementTransform(glabel, gl2), run_time=1.1)
        self.play(Flash(token[3], color=ACCENT, line_length=0.18, flash_radius=0.5), run_time=0.5)
        self.wait(3.5)

        switch("GUARD", "鎖也跟著換", "Guard 變成 [None, 4] 加一條 2 <= s0：0 和 1 永遠特化，下界才是 2")
        card2 = guard_card(["TENSOR_MATCH [None, 4]", "LAMBDA_GUARD  2 <= s0"]).move_to([0, BALL_Y, 0])
        lines = card2[1][1]
        lines[0][13:17].set_color(ACCENT)
        lines[1][12:17].set_color(ACCENT)
        self.play(ReplacementTransform(card, card2), FadeOut(cross), fs2.animate.set_color(DIM).shift(DOWN * 0.2), run_time=0.9)
        echo = token[3].copy()
        self.play(Transform(echo, lines[1][15:17].copy()), run_time=0.7)
        self.remove(echo)
        self.wait(3.5)

        def verify(b, exit_rt=0.9):
            self.play(lines[0].animate.set_color(ACCENT), run_time=0.2)
            self.play(lines[0].animate.set_color(MUTED), run_time=0.12)
            self.play(lines[1].animate.set_color(ACCENT), run_time=0.2)
            self.play(lines[1].animate.set_color(MUTED), run_time=0.12)
            self.play(b.animate.move_to([8.5, BALL_Y, 0]), Flash([card2.get_right()[0] + 0.45, BALL_Y, 0], color=ACCENT, line_length=0.12, flash_radius=0.28), *[l.animate.set_color(TXT) for l in lines], run_time=exit_rt)
            self.remove(b)

        switch("RETRY", "重新驗票", "(8, 4) 拿新的票根重驗：第一格不看具體值，2 <= 8 過，放行")
        self.play(b2[1][1].animate.set_color(TXT), run_time=0.2)
        verify(b2)
        self.wait(2)

        switch("CALL 3", "同一張圖", "(16, 4) 進來：TENSOR_MATCH 不看第一格，範圍檢查 2 <= 16 也過，直接重用")
        b3 = ball("(16, 4)").move_to([-8, BALL_Y, 0])
        stop3 = card2.get_left()[0] - b3.width / 2 - 0.4
        self.play(b3.animate.move_to([stop3, BALL_Y, 0]), run_time=0.8)
        verify(b3)
        rt = T("reuse, no recompile", font=MONO, font_size=16, color=MUTED).next_to(card2, RIGHT, buff=0.45)
        self.play(FadeIn(rt, shift=RIGHT * 0.1), run_time=0.3)
        self.wait(2.5)

        switch("CALL 4", "recompiles 安靜", "(100, 4) 也一樣放行：不再重編，一張圖吃下所有 batch size")
        b4 = ball("(100, 4)").move_to([-8, BALL_Y, 0])
        stop4 = card2.get_left()[0] - b4.width / 2 - 0.4
        self.play(b4.animate.move_to([stop4, BALL_Y, 0]), run_time=0.7)
        verify(b4, exit_rt=0.7)
        self.wait(2.5)

        switch("RULE", "被逼才 dynamic", "預設 static、被逼才 dynamic：兩次編譯是固定成本，mark_dynamic 能省掉那次賭輸")
        self.play(FadeOut(rt), run_time=0.2)
        self.wait(5.5)
