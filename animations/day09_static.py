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
    t = T(name, font=SANS, font_size=20, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=14, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub, edge=EDGE, sw=1.5, fill=CARD):
    r = panel(w, h, edge=edge, sw=sw, fill=fill)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
    return VGroup(r, hdr)


def chiprow(card, lines, size=12, buff=0.28):
    g = VGroup(*[T(l, font=MONO, font_size=size, color=TXT) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return g.next_to(card[1], DOWN, buff=0.4).align_to(card[1], LEFT)


class Jump(Scene):
    def construct(self):
        title = T("jump target:  offset  vs  reference", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        title[13:19].set_color(ACCENT)
        self.add(title)
        self.add(label("bytecode_transformation.py 的跳轉虛擬化", size=14).next_to(title, DOWN, buff=0.15).align_to(title, LEFT))

        W, H = 6.05, 4.2
        TOPY = -0.5
        left = titled(W, H, "頁碼", "跳到第 84 個 byte", edge=EDGE).move_to([-3.35, TOPY, 0])
        right = titled(W, H, "書籤", "target 指著 Instruction 物件", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([3.35, TOPY, 0])
        self.add(label("BEFORE  ·  數字 offset，插一條全作廢").next_to(left, UP, buff=0.22).align_to(left, LEFT))
        self.add(label("AFTER  ·  參照，怎麼插都指得到").next_to(right, UP, buff=0.22).align_to(right, LEFT))

        lrows = chiprow(left, ["10  POP_JUMP_IF_TRUE  -> 84", "12  ...", "..  (插入一條新指令)", "84  RETURN_CONST  <- 位移成 86"], size=13, buff=0.3)
        lrows[2].set_color(ACCENT)
        lrows[3].set_color(MUTED)
        lnote = chiprow(left, ["所有寫死的數字同時作廢，", "只能全部重算一遍"], size=13, buff=0.2).set_color(MUTED)
        lnote.next_to(lrows, DOWN, buff=0.6).align_to(lrows, LEFT)
        self.add(left, lrows, lnote)

        rrows = chiprow(right, ["POP_JUMP_IF_TRUE", "  .target = <Instruction RETURN_CONST>", "...", "(插入任何指令都不斷鏈)"], size=13, buff=0.3)
        rrows[1].set_color(ACCENT)
        rrows[3].set_color(MUTED)
        rnote = chiprow(right, ["組裝時才把參照換算回 offset，", "EXTENDED_ARG 反覆掃描直到收斂"], size=13, buff=0.2).set_color(MUTED)
        rnote.next_to(rrows, DOWN, buff=0.6).align_to(rrows, LEFT)
        self.add(right, rrows, rnote)

        self.add(T("Dynamo 要在 bytecode 中間大量增刪指令，頁碼式的跳轉一動就碎；書籤式的參照讓生成器隨便插，最後一次結算。", font=CJK, font_size=14, color=MUTED).to_edge(DOWN, buff=0.25))
