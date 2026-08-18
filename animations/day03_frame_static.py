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
    t = T(name, font=SANS, font_size=21, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=14, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub, fill=CARD, edge=EDGE, sw=1.5):
    r = panel(w, h, fill=fill, edge=edge, sw=sw)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
    return VGroup(r, hdr)


def rows(lines, size=12, color=TXT, buff=0.1):
    return VGroup(*[T(l, font=MONO, font_size=size, color=color) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def kv(pairs, size=12, buff=0.1, kw=1.05):
    g = VGroup()
    for k, v, c in pairs:
        kk = T(k, font=MONO, font_size=size, color=MUTED)
        vv = T(v, font=MONO, font_size=size, color=c)
        vv.next_to(kk, RIGHT, buff=0).align_to(kk, DOWN)
        vv.shift(RIGHT * (kw - kk.width))
        g.add(VGroup(kk, vv))
    return g.arrange(DOWN, aligned_edge=LEFT, buff=buff)


def arrow(a, b, color=MUTED, w=1.8):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.16, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


class FrameStatic(Scene):
    def construct(self):
        title = T("def f(x): return torch.sin(x) + 1", font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
        self.add(title)
        self.add(label("呼叫 f(x) 的那一刻，CPython 手上有這幾樣東西", size=14).next_to(title, DOWN, buff=0.15).align_to(title, LEFT))

        TOP, BOT = 2.2, -2.85
        H = TOP - BOT
        CY = (TOP + BOT) / 2

        # function object (left, small)
        W = 4.07
        fn = titled(W, 2.3, "function", "f").move_to([-4.575, TOP - 1.15, 0])
        fn_body = kv([("__code__", "-> code object", ACCENT), ("__globals__", "-> globals dict", TXT), ("__closure__", "(torch,)", MUTED)], kw=1.45, buff=0.16).next_to(fn[1], DOWN, buff=0.22).align_to(fn[1], LEFT)
        self.add(fn, fn_body)

        # globals dict (left bottom)
        gl = titled(W, 2.15, "globals", "模組層字典").move_to([-4.575, BOT + 1.075, 0])
        gl_body = kv([("torch", "<module>", TXT), ("f", "<function f>", TXT), ("__name__", "'__main__'", MUTED)], kw=1.1, buff=0.16).next_to(gl[1], DOWN, buff=0.22).align_to(gl[1], LEFT)
        self.add(gl, gl_body)

        # code object (center)
        co = titled(W, H, "code object", "編譯一次就固定")
        co.move_to([0, CY, 0])
        co_meta = kv([("co_name", "'f'", TXT), ("co_varnames", "('x',)", TXT), ("co_consts", "(None, 1)", TXT), ("co_names", "('sin',)", TXT), ("co_freevars", "('torch',)", TXT)], kw=1.35, buff=0.13).next_to(co[1], DOWN, buff=0.22).align_to(co[1], LEFT)
        self.add(co, co_meta)
        bc_lbl = T("co_code  bytecode", font=MONO, font_size=12, color=MUTED).next_to(co_meta, DOWN, buff=0.32).align_to(co_meta, LEFT)
        bc = rows(["LOAD_DEREF   torch", "LOAD_ATTR    sin", "LOAD_FAST    x", "CALL         1", "LOAD_CONST   1", "BINARY_OP    +", "RETURN_VALUE"], size=12, buff=0.11)
        bc_box = panel(W - 0.5, bc.height + 0.3, fill=CARD_DIM, r=0.08).next_to(bc_lbl, DOWN, buff=0.12).align_to(co[0].get_left() + RIGHT * 0.25, LEFT)
        bc.move_to(bc_box).align_to(bc_box.get_left() + RIGHT * 0.15, LEFT)
        self.add(bc_box, bc, bc_lbl)

        # frame (right)
        fr = titled(W, H, "frame", "每次呼叫新建一個", edge=ACCENT, sw=2, fill=ACTIVE_FILL)
        fr.move_to([4.575, CY, 0])
        fr_meta = kv([("f_code", "-> code object", ACCENT), ("f_globals", "-> globals dict", TXT), ("f_back", "-> caller frame", MUTED), ("f_lasti", "28  (CALL)", TXT)], kw=1.35, buff=0.14).next_to(fr[1], DOWN, buff=0.22).align_to(fr[1], LEFT)
        self.add(fr, fr_meta)
        loc_lbl = T("f_locals  區域變數", font=MONO, font_size=12, color=MUTED).next_to(fr_meta, DOWN, buff=0.32).align_to(fr_meta, LEFT)
        loc = kv([("x", "Tensor f32[8]", TXT)], kw=0.6)
        loc_box = panel(W - 0.5, loc.height + 0.3, fill=CARD_DIM, r=0.08).next_to(loc_lbl, DOWN, buff=0.12).align_to(fr[0].get_left() + RIGHT * 0.25, LEFT)
        loc.move_to(loc_box).align_to(loc_box.get_left() + RIGHT * 0.15, LEFT)
        self.add(loc_lbl, loc_box, loc)
        st_lbl = T("value stack  執行到 CALL 之前", font=MONO, font_size=12, color=MUTED).next_to(loc_box, DOWN, buff=0.32).align_to(loc_lbl, LEFT)
        st = rows(["x", "torch.sin", "NULL"], size=12, buff=0.12)
        st_box = panel(W - 0.5, st.height + 0.3, fill=CARD_DIM, r=0.08).next_to(st_lbl, DOWN, buff=0.12).align_to(fr[0].get_left() + RIGHT * 0.25, LEFT)
        st.move_to(st_box).align_to(st_box.get_left() + RIGHT * 0.15, LEFT)
        top_tag = T("<- top", font=MONO, font_size=11, color=ACCENT).next_to(st[0], RIGHT, buff=0.5)
        self.add(st_lbl, st_box, st, top_tag)

        # arrows
        y_code = fn_body[0].get_y()
        self.add(arrow([fn[0].get_right()[0], y_code, 0], [co[0].get_left()[0], y_code, 0], color=ACCENT))
        y_glob = fn_body[1].get_y()
        gpath = VMobject(stroke_color=MUTED, stroke_width=1.8).set_points_as_corners([[fn[0].get_right()[0], y_glob, 0], [fn[0].get_right()[0] + 0.16, y_glob, 0], [fn[0].get_right()[0] + 0.16, gl_body[0].get_y(), 0]])
        self.add(gpath, Arrow([fn[0].get_right()[0] + 0.16, gl_body[0].get_y(), 0], [gl[0].get_right()[0], gl_body[0].get_y(), 0], buff=0, color=MUTED, stroke_width=1.8, tip_length=0.16, max_tip_length_to_length_ratio=1))
        y_fcode = fr_meta[0].get_y()
        self.add(arrow([fr[0].get_left()[0], y_fcode, 0], [co[0].get_right()[0], y_fcode, 0], color=ACCENT))
        y_fglob = fr_meta[1].get_y()
        gx = (co[0].get_right()[0] + fr[0].get_left()[0]) / 2
        yb = BOT - 0.3
        p2 = [gl[0].get_right()[0], gl_body[1].get_y(), 0]
        path = VMobject(stroke_color=MUTED, stroke_width=1.8).set_points_as_corners([[fr[0].get_left()[0], y_fglob, 0], [gx, y_fglob, 0], [gx, yb, 0], [p2[0] + 0.34, yb, 0], [p2[0] + 0.34, p2[1], 0]])
        tip = Arrow([p2[0] + 0.34, p2[1], 0], p2, buff=0, color=MUTED, stroke_width=1.8, tip_length=0.16, max_tip_length_to_length_ratio=1)
        self.add(path, tip)

        foot = T("code object 是靜態的：bytecode、常數、變數名，編譯一次就固定。frame 是動態的：這一次呼叫的區域變數、value stack、執行到第幾條。", font=CJK, font_size=13, color=MUTED).to_edge(DOWN, buff=0.22)
        self.add(foot)
