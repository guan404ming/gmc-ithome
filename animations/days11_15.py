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
    t = T(name, font=SANS, font_size=20, weight=BOLD, color=TXT)
    s = T(sub, font=CJK, font_size=14, color=MUTED)
    return VGroup(t, s).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def titled(w, h, name, sub, edge=EDGE, sw=1.5, fill=CARD):
    r = panel(w, h, edge=edge, sw=sw, fill=fill)
    hdr = header(name, sub).move_to(r.get_corner(UL) + RIGHT * 0.25 + DOWN * 0.22, aligned_edge=UL)
    return VGroup(r, hdr)


def rows(lines, size=12, color=TXT, buff=0.12):
    items = []
    for l in lines:
        if l == "":
            items.append(Rectangle(width=0.01, height=size * 4 / 100 * 0.25, stroke_width=0, fill_opacity=0))
        else:
            items.append(T(l, font=MONO, font_size=size, color=color))
    return VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def body(card, lines, size=12, buff=0.14):
    g = rows(lines, size=size, buff=buff)
    return g.next_to(card[1], DOWN, buff=0.35).align_to(card[1], LEFT)


def arrow(a, b, color=MUTED, w=2):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.14, max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=20)


def head(scene, main, sub, hi=None):
    t = T(main, font=MONO, font_size=24, color=TXT).to_corner(UL, buff=0.5)
    if hi:
        t[hi[0]:hi[1]].set_color(ACCENT)
    scene.add(t)
    scene.add(label(sub, size=14).next_to(t, DOWN, buff=0.15).align_to(t, LEFT))


def foot(scene, s):
    scene.add(T(s, font=CJK, font_size=14, color=MUTED).to_edge(DOWN, buff=0.25))


class D11(Scene):
    def construct(self):
        head(self, "automatic dynamic", "賭輸一次之後，Dynamo 改押符號", (10, 17))
        W, H, Y = 4.07, 4.35, -0.35
        c1 = titled(W, H, "CALL 1", "(4, 4)  static").move_to([-4.575, Y, 0])
        b1 = body(c1, ["第一次編譯：全部特化", "", "TENSOR_MATCH", "  size=[4, 4]", "", "特化的圖好最佳化", "但只認 batch = 4"], size=12, buff=0.16)
        for i in (2, 3):
            b1[i].set_color(TXT)
        b1[0].set_color(MUTED); b1[5].set_color(MUTED); b1[6].set_color(MUTED)
        c2 = titled(W, H, "CALL 2", "(8, 4)  recompile", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([0.0, Y, 0])
        b2 = body(c2, ["size mismatch: 4 -> 8", "", "frame_state:", "  dim 0 上次 4 這次 8", "  它會變 -> 用符號 s0", "", "第二次編譯改押 SymInt"], size=12, buff=0.16)
        b2[0].set_color(ACCENT); b2[6].set_color(MUTED)
        c3 = titled(W, H, "CALL 3+", "(16,4) (100,4) ...").move_to([4.575, Y, 0])
        b3 = body(c3, ["TENSOR_MATCH", "  size=[None, 4]", "2 <= L['x'].size()[0]", "", "no recompile", "一張圖吃下所有 batch"], size=12, buff=0.16)
        b3[2].set_color(ACCENT); b3[4].set_color(TXT); b3[5].set_color(MUTED)
        self.add(label("STATIC").next_to(c1, UP, buff=0.22).align_to(c1, LEFT))
        self.add(label("RECOMPILE  ·  止損").next_to(c2, UP, buff=0.22).align_to(c2, LEFT))
        self.add(label("DYNAMIC  ·  s0").next_to(c3, UP, buff=0.22).align_to(c3, LEFT))
        self.add(c1, b1, c2, b2, c3, b3)
        self.add(arrow(c1[0].get_right(), c2[0].get_left(), color=ACCENT))
        self.add(arrow(c2[0].get_right(), c3[0].get_left(), color=ACCENT))
        foot(self, "預設 static、被逼才 dynamic：兩次編譯是固定成本，mark_dynamic 可以省掉第一次；0 和 1 永遠特化。")


class D12(Scene):
    def construct(self):
        head(self, "AOTAutograd", "一張 torch 層 forward 圖，變成兩張 ATen 層的圖", (0, 3))
        LW, LH = 3.6, 3.3
        dyn = titled(LW, LH, "DYNAMO 圖", "torch 層").move_to([-4.8, -0.3, 0])
        db = body(dyn, ["matmul = x @ w", "relu = matmul.relu()", "sum_1 = relu.sum()", "", "只有 forward"], size=12, buff=0.16)
        db[4].set_color(MUTED)
        self.add(label("INPUT").next_to(dyn, UP, buff=0.22).align_to(dyn, LEFT), dyn, db)

        mid = titled(2.6, 2.0, "AOT", "FakeTensor 重跑").move_to([-1.05, -0.3, 0])
        mb = body(mid, ["autograd 引擎", "展開 backward"], size=11, buff=0.12)
        mb.set_color(MUTED)
        self.add(mid, mb)
        self.add(arrow(dyn[0].get_right(), mid[0].get_left(), color=ACCENT))

        RW, RH = 5.7, 2.3
        fw = titled(RW, RH, "FORWARD", "ATen 層", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([3.85, 1.15, 0])
        fb = body(fw, ["mm -> relu -> sum", "return (sum_1, le, permute)"], size=12, buff=0.14)
        fb[1][:6].set_color(TXT)
        self.add(label("OUTPUT  ·  兩張圖").next_to(fw, UP, buff=0.22).align_to(fw, LEFT), fw, fb)
        bw = titled(RW, RH, "BACKWARD", "也交給 Inductor", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([3.85, -1.75, 0])
        bb = body(bw, ["inputs: le, permute, tangents_1", "where -> mm_1", "return (None, grad_w)"], size=12, buff=0.14)
        self.add(bw, bb)
        self.add(arrow(mid[0].get_right(), fw[0].get_left(), color=ACCENT))
        self.add(arrow(mid[0].get_right(), bw[0].get_left(), color=ACCENT))
        sv = arrow(fw[0].get_bottom(), bw[0].get_top(), color=ACCENT, w=2.5)
        self.add(sv, T("saved: le, permute", font=MONO, font_size=12, color=ACCENT).next_to(sv, RIGHT, buff=0.15))
        foot(self, "forward 多輸出一批要保存的中間值，backward 拿著它們和上游梯度算出對輸入的梯度；微分規則來自 autograd 引擎，AOT 只是把過程錄下來。")


class D13(Scene):
    def construct(self):
        head(self, "functionalization", "in-place 換成 out-of-place，修改在邊界一次結清", (0, 17))
        W, H, Y = 6.05, 4.5, -0.4
        left = titled(W, H, "使用者寫的", "graph_code").move_to([-3.35, Y, 0])
        lb = body(left, ["y = x.view(2, 8)", "y.add_(1)      # 就地改", "y.relu_()      # 就地改", "return x * 2   # x 被改過", "", "aliasing + mutation", "後端不能自由重排"], size=12, buff=0.18)
        lb[1][9:].set_color(ACCENT); lb[2][9:].set_color(ACCENT)
        lb[5].set_color(MUTED); lb[6].set_color(MUTED)
        self.add(label("BEFORE  ·  torch 層").next_to(left, UP, buff=0.22).align_to(left, LEFT), left, lb)

        right = titled(W, H, "Functionalized", "aot_graphs", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([3.35, Y, 0])
        rb = body(right, ["view   = view(x, [2,8])", "add    = add(view, 1)", "view_1 = view(add, [4,4])   # 寫回 base", "view_2 = view(view_1, [2,8])# 再長出 view", "relu   = relu(view_2)", "mul    = mul(view_3, 2)", "copy_(x, view_3)  # 唯一倖存的 mutation", "return (mul,)"], size=12, buff=0.14)
        rb[6].set_color(ACCENT)
        self.add(label("AFTER  ·  ATen 層，圖內純函數").next_to(right, UP, buff=0.22).align_to(right, LEFT), right, rb)
        self.add(arrow(left[0].get_right(), right[0].get_left(), color=ACCENT))
        foot(self, "add_ 變 add、view 用重放維持一致；對輸入的修改集中成圖尾端的一條 copy_，跟 Day 7 的 SideEffects 同一套哲學：最後一刻結算。")


class D14(Scene):
    def construct(self):
        head(self, "decomposition", "拆是為了讓後端用自己的方式拼回去", (0, 13))
        W, Y = 3.6, -0.4
        left = titled(W, 3.0, "高階 op", "使用者看到的").move_to([-4.8, Y, 0])
        lb = body(left, ["LayerNorm(8)", "F.gelu(...)", "", "2000+ 個 op", "後端寫不完"], size=12, buff=0.18)
        lb[3].set_color(MUTED); lb[4].set_color(MUTED)
        self.add(label("OPS").next_to(left, UP, buff=0.22).align_to(left, LEFT), left, lb)

        mid = titled(4.6, 4.6, "基本運算", "decomposition_table", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([-0.15, Y, 0])
        mb = body(mid, ["LayerNorm ->", "  var_mean, rsqrt,", "  sub, mul, add", "", "GELU ->", "  mul(x, 0.5)", "  erf(x * 0.7071)", "  add, mul", "", "matmul 不拆（戰略 op）"], size=12, buff=0.12)
        mb[9].set_color(MUTED)
        self.add(label("DECOMPOSE  ·  1123 條規則").next_to(mid, UP, buff=0.22).align_to(mid, LEFT), mid, mb)

        right = titled(W, 3.0, "Inductor", "融合回去").move_to([4.65, Y, 0])
        rb = body(right, ["pointwise 全部", "融回一個 kernel", "", "拆解 + 融合 =", "免手寫 fused kernel"], size=12, buff=0.18)
        rb[3].set_color(MUTED); rb[4].set_color(MUTED)
        self.add(label("FUSE").next_to(right, UP, buff=0.22).align_to(right, LEFT), right, rb)
        self.add(arrow(left[0].get_right(), mid[0].get_left(), color=ACCENT))
        self.add(arrow(mid[0].get_right(), right[0].get_left(), color=ACCENT))
        foot(self, "規則是普通的 Python 函式、表是可以換的字典；pointwise 儘管拆，matmul、conv 留給專屬實作。")


class D15(Scene):
    def construct(self):
        head(self, "min-cut partitioner", "切線落在哪，決定 forward 要保存什麼", (0, 7))
        W, H = 12.3, 3.3
        joint = titled(W, H, "JOINT GRAPH", "forward 和 backward 先畫成一張").move_to([0, 0.55, 0])
        self.add(label("JOINT  ·  aot_joint_graph").next_to(joint, UP, buff=0.22).align_to(joint, LEFT), joint)

        def node(txt, w=1.5, accent=False):
            r = panel(w, 0.5, fill=ACTIVE_FILL if accent else CARD_DIM, edge=ACCENT if accent else EDGE, r=0.08, sw=2 if accent else 1.5)
            t = T(txt, font=MONO, font_size=12, color=TXT).move_to(r)
            return VGroup(r, t)

        y = 0.25
        n1 = node("primals").move_to([-5.1, y, 0])
        n2 = node("mm", w=1.1, accent=True).move_to([-3.2, y, 0])
        n3 = node("tanh", w=1.2).move_to([-1.4, y, 0])
        n4 = node("sum", w=1.1).move_to([0.4, y, 0])
        y2 = -0.75
        n5 = node("1 - tanh^2", w=1.9).move_to([1.3, y2, 0])
        n6 = node("mul", w=1.1).move_to([3.2, y2, 0])
        n7 = node("mm_1 = grad_w", w=2.2, accent=False).move_to([5.05, y2, 0])
        for a, b in [(n1, n2), (n2, n3), (n3, n4)]:
            self.add(arrow(a[0].get_right(), b[0].get_left()))
        self.add(arrow(n3[0].get_bottom(), n5[0].get_left(), color=MUTED))
        self.add(arrow(n5[0].get_right(), n6[0].get_left()))
        self.add(arrow(n6[0].get_right(), n7[0].get_left()))
        self.add(n1, n2, n3, n4, n5, n6, n7)
        cut = DashedLine([0.25, 1.1, 0], [0.95, -1.1, 0], color=ACCENT, stroke_width=3, dash_length=0.12)
        self.add(cut, T("cut", font=MONO, font_size=13, color=ACCENT, weight=BOLD).next_to(cut.get_start(), UR, buff=0.08))
        self.add(T("forward", font=MONO, font_size=12, color=MUTED).move_to([-4.9, 1.0, 0]))
        self.add(T("backward", font=MONO, font_size=12, color=MUTED).move_to([5.15, 1.0, 0]))

        W2, H2 = 6.05, 2.0
        lo = titled(W2, H2, "存 mm，重算 tanh", "min-cut 預設", edge=ACCENT, sw=2, fill=ACTIVE_FILL).move_to([-3.35, -2.35, 0])
        lob = body(lo, ["跨線的量最小：saved = (mm, permute)", "tanh 是 pointwise，重算幾乎免費"], size=12, buff=0.12)
        ro = titled(W2, H2, "checkpoint", "旋鈕轉到底").move_to([3.35, -2.35, 0])
        rob = body(ro, ["只存 primals，backward 重算整段", "記憶體最省，多付一次 forward"], size=12, buff=0.12)
        self.add(lo, lob, ro, rob)
        foot(self, "訓練記憶體的大頭是 activation：切線往 forward 靠是多存、往 backward 靠是多算；把保存成本當邊權重求最小割。")
