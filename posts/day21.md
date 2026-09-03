# Day 21 | 一人一塊磚？讀懂 Inductor 生出來的 Triton Kernel

## 前言

昨天把 fusion 的邊界走完了，Scheduler 手上是一串融合好的 node。接下來是產線的最後一站 codegen，把每一組 node 寫成 GPU 上跑得動的程式碼。畫 pipeline 地圖那次結尾偷看過一眼產物，那個叫 `triton_poi_fused_add_cos_mul_sin_tanh_0` 的神秘函式當時只能當黑盒子欣賞，今天就把這種 Triton kernel 攤開來逐行讀，搞懂 xindex、XBLOCK、mask 這些角色，以及 kernel 是怎麼被 launch 的。前面幾天鋪的路今天也正好驗收，lowering 攤開的 loop、fusion 敲定的邊界，全部會印在 kernel 的長相上。本篇實驗跑在 Modal 租的 L40S GPU 上（torch 2.8.0）。

正文開始！

## Triton 站在哪個高度寫 GPU

先交代 Triton 是什麼。它是 OpenAI 開源的專案，後來成了 PyTorch 預設 GPU 後端的地基。表面上它是用 Python 語法寫 GPU kernel 的語言，但真正的差別不在語法，在抽象的程度。

- **CUDA 的單位是 thread**。寫的人要自己安排每條 thread 摸哪個元素、shared memory 怎麼擺、存取怎麼對齊，任何一處寫錯效能就掉一截。
- **Triton 的單位是 block**。一份 kernel 描述的是「一個 program instance 一次處理一塊資料」，塊內怎麼分工給 thread、存取怎麼 coalesce，全交給 Triton 編譯器處理。

換個說法，CUDA 寫的是每個工人的動作，Triton 寫的是每塊磚的流程，工人怎麼分配是工頭的事。

這個高度剛好接得上 Inductor 的 IR。lowering 攤出來的 loop-level IR 描述「迴圈的每一輪做什麼」，把一輪放大成一塊，load 翻成 `tl.load`、store 翻成 `tl.store`，中間的算術逐條照抄，一顆 kernel 就成形了。翻譯的人不必懂 GPU 微架構，難寫對的部分讓 Triton 兜底。這正是 PyTorch 2 論文選 Triton 當 GPU 後端的理由，生成器寫得簡單，產物又常能追上手寫 CUDA。

## 逐行讀一顆 pointwise kernel

沿用之前那條 op 鏈當實驗品，`relu(x + y) * 2`。開 `TORCH_LOGS="output_code"` 在 GPU 上編一次，輸入是一百萬個元素的 tensor，kernel 完整長這樣。

```python
@triton.jit
def triton_poi_fused_add_mul_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
```

名字先講。`poi` 代表 pointwise，後面接被融進來的 op 清單。一個名字一顆 kernel，光看名字就知道 fusion 的戰果。

開頭四行是所有 pointwise kernel 共用的模板。而 program instance 就是 GPU 上同時開跑的一份 kernel 副本，一份副本領一塊磚。

Inductor 先把輸出攤平成長度 xnumel 的線性索引，切成 XBLOCK 大小的磚。GPU 同時 launch 一大批 program instance，每個用 `tl.program_id(0)` 拿到編號，乘上 XBLOCK 就是自己那塊磚的起點，再加上 `tl.arange` 展開，xindex 就是磚裡每一格（也就是每條 lane）負責的元素編號。一個 instance 一次操作整塊 XBLOCK 長的向量，而不是單一元素，這是讀 Triton 最需要切換的視角。

再來是 mask。一百萬除以 1024 不能整除，最後一塊磚只有一部分是真的。`xindex < xnumel` 把越界的 lane 關掉，load 和 store 都帶著它，被遮住的 lane 不讀也不寫，尾巴就安全處理掉了。

中段就是 IR 那個 body 的直譯。兩筆 `tl.load` 把磚吸進來，接著加法、`maximum`（relu 在 lowering 之後的長相）、乘 2，中間值全活在暫存器，最後一筆 `tl.store` 把磚放回去。三個 op 一顆 kernel，讀兩次寫一次，fusion 承諾的事在這十幾行裡兌現。

還有兩個容易忽略的細節。

- `**xnumel = 1000000**`。參數明明傳進來了，卻直接用常數蓋掉。這是講 automatic dynamic 時說過的 static shape 特化，shape 押死之後編譯器能做更多假設，走 dynamic shape 這裡就會留著變數。
- **XBLOCK 宣告成 `tl.constexpr`**，整份原始碼卻沒有它的數值。這是故意留白，同一份程式配不同的 XBLOCK 就是不同的效能，這個決定被推遲到 launch 時刻。

![tensor 切成 XBLOCK 大小的磚，program instance 領磚，mask 遮尾，load 算完 store 回去](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day21/triton_codegen.gif)

*圖一：一顆 pointwise kernel 的一生。輸出攤平成 xnumel 格，切成 XBLOCK 大小的磚，每個 program instance 憑 pid 算出自己的 xindex，mask 把尾端越界的 lane 遮掉，tl.load 把磚吸進暫存器，算完 tl.store 放回去，wrapper 算好 grid 把所有 instance 一次發射。*

## XBLOCK 是誰決定的

同一份輸出的下半部是 wrapper，負責檢查輸入、配置 buffer、按順序呼叫 kernel（節錄）。

```python
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1000000, ), (1, ))
    assert_size_stride(arg1_1, (1000000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1000000, ), (1, ), torch.float32)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_relu_0.run(arg0_1, arg1_1, buf0, 1000000, stream=stream0)
    return (buf0, )
```

注意 `.run` 只傳了指標和 xnumel，沒有 XBLOCK 也沒有 grid。答案在 kernel 定義上方那圈裝飾器，它按 shape 量級從幾組候選 config 裡挑，必要時逐一實測比快慢，這就是 autotune，怎麼挑留給之後專門談它的那篇。實驗最後把選中的參數印了出來。

```
triton_poi_fused_add_mul_relu_0: picked {'XBLOCK': 1024} num_warps=4 num_stages=1
triton_poi_fused_add_mul_relu_0: grid = (ceil(1000000 / 1024), 1, 1) = (977, 1, 1)
```

XBLOCK 選了 1024，一塊磚 1024 個元素。num_warps=4 表示塊內由 128 條 thread 分工，正是 Triton 幫忙藏掉的那一層。grid 就是磚的數量，一百萬個元素切成 977 塊，最後一塊裝不滿，xmask 就是為它準備的。977 個 instance 撒到各個 SM 上同時開工，一次 launch 完成。

## reduction kernel 的兩種長相

pointwise 的磚彼此獨立，reduction 卻要把一整段收成一個值，模板自然不同。而選哪個模板，只看一個問題，要收的那一段塞不塞得進一塊磚。

- **`per`**：塞得進。一塊磚裝下整段，load 進來一口氣收掉，一個 instance 負責一段。
- **`red`**：塞不進。磚裡擺一個累加器，用迴圈一輪一輪吃完。

先看塞得進的。`relu(x + 1).sum(dim=1)` 配 1024x1024 的輸入，每個 row 剛好 1024 個元素，kernel 名字是 `triton_per_fused_add_relu_sum_0`（節錄）。

```python
def triton_per_fused_add_relu_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 + tmp1
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp7, None)
```

`per` 是 persistent reduction，一塊磚裝下一整個 row，所以 R0_BLOCK 直接開到 1024。load 進來、pointwise 順路做完、`tl.sum` 一口氣收掉，一個 instance 負責一個 row。講 fusion 時說的「pointwise 融進 reduction 的迴圈」，就是這個模樣。索引也出現新面孔，x 開頭的變數管平行維度，r 開頭的管收縮維度。

換成全域的 `x.sum()` 就沒這麼舒服。4096x4096 一千六百多萬個元素，一塊磚吞不下，可是只派一個 instance 慢慢收，整張 GPU 又閒著。

Inductor 的解法是拆兩段。第一段派 512 個 instance 各認領一小段，前綴 `red` 代表帶迴圈，核心是一個 for 迴圈抱著累加器一輪一輪吃（節錄）。

```python
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        tmp0 = tl.load(in_ptr0 + (r0_1 + 32768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
```

跑完迴圈才把累加器收成一個部分和，512 個 instance 就吐出 512 個部分和。第二段再用一顆 `per` 把它們收成最後的答案，wrapper 裡就寫著兩次 launch。

```
triton_red_fused_sum_0.run(arg0_1, buf0, 512, 32768, stream=stream0)
triton_per_fused_sum_1.run(buf0, buf1, 1, 512, stream=stream0)
```

多付一次 kernel launch，換到 512 倍的平行度，划算。講 fusion 時說過全域 reduction 是一面牆，這面牆的另一個面貌在這裡，連 reduction 自己都得拆成兩顆 kernel 才餵得飽整張 GPU。

分工的參數也跟著體質走，實驗把這兩顆 kernel 選中的 config 一併印了出來。

```
triton_per_fused_add_relu_sum_0: picked {} num_warps=8 num_stages=1
triton_red_fused_sum_0: picked {'XBLOCK': 1, 'R0_BLOCK': 2048} num_warps=16 num_stages=1
```

逐 row 那顆沒什麼好挑，塊的大小被 row 定死了。兩段式的第一段則選了比較大的 R0_BLOCK，迴圈一輪多吃一點，早點吃完自己那一段。同一套模板，不同的 shape，分工的帳一顆一顆算。

## 自己讀 kernel 的小抄

拿到一份陌生的 inductor-generated kernel，可以照這四個地方看：

- **名字的前綴**。`poi` 是 pointwise，`per` 是一塊磚裝下整段收縮的 persistent reduction，`red` 是帶迴圈的長跑型。
- **變數的字首**。x 家族管平行維度，r 家族管收縮維度。
- **mask 的位置**。mask 出現在哪個維度，那個維度就有除不盡的尾巴，全是 None 代表 shape 剛好對齊。
- **wrapper 的 `.run`**。`call` 裡每一行 `.run` 就是一次 launch，兩段式 reduction 就是兩行。

想親眼看這一切，`TORCH_LOGS="output_code"` 一行就夠。

## 結語

今天把 Inductor 的 GPU 產物逐行讀完了。Triton 把寫 kernel 的單位從 thread 抬高到 block，codegen 只需要把 loop-level IR 逐行翻成 tl.load、算術、tl.store。pointwise 是一磚配一個 instance 的模板，mask 照顧除不盡的尾端，reduction 按大小選 persistent 或兩段式，XBLOCK 和 grid 留到 launch 時刻由 heuristics 拍板。pipeline 地圖裡那個只能遠觀的黑盒子，現在應該已經是可以逐行指認的老朋友了。

不過 GPU 只是 codegen 的其中一條分流。同一層 IR 落在 CPU 上走的是另一條路，生出來的是 C 再加上 OpenMP，決策長得完全不同。明天就來把 C Codegen 這條路走完。那我們明天見！

## 參考資料

- [torch/_inductor/codegen/triton.py：TritonKernel（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/triton.py)
- [torch/_inductor/codegen/simd.py：SIMDScheduling（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/simd.py)
- [torch/_inductor/runtime/triton_heuristics.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/runtime/triton_heuristics.py)
- [Triton 官方文件](https://triton-lang.org/main/index.html)
- Tillet et al., [*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*](https://dl.acm.org/doi/10.1145/3315508.3329973), MAPL 2019
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）

