# Day 21 | 一人一塊磚？讀懂 Inductor 生出來的 Triton Kernel

## 前言

昨天把 fusion 的邊界走完了，誰跟誰同組已經定案，Scheduler 手上是一串融合好的 node。接下來就是產線的最後一站 codegen，把每一組 node 真的寫成 GPU 上跑得動的程式碼。Day 2 結尾其實偷看過一眼產物，那個叫 `triton_poi_fused_add_cos_mul_sin_tanh_0` 的神秘函式，當時只能當黑盒子欣賞，今天就把這種 Triton kernel 攤開來逐行讀，搞懂 xindex、XBLOCK、mask 這些反覆出現的角色，以及 kernel 最後是怎麼被 launch 的。讀完之後，任何一顆 Inductor 生的 kernel 應該都能看出骨架。前面幾天鋪的路今天也正好驗收，lowering 攤開的 loop、Scheduler 排好的組、fusion 敲定的邊界，全部會印在 kernel 的長相上。本篇實驗跑在 Modal 租的 L40S GPU 上（torch 2.8.0），生成這些 kernel 的原始碼在 v2.8.0 的 [`codegen/triton.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/triton.py)。

正文開始！

## Triton 站在哪個高度寫 GPU

先交代 Triton 是什麼。它是 OpenAI 開源的專案，後來成了 PyTorch 預設 GPU 後端的地基。表面上它是一個用 Python 語法寫 GPU kernel 的語言，但真正的差別不在語法，在抽象的高度。CUDA 的心智模型是 thread，寫的人要自己安排每條 thread 摸哪個元素、幾條 thread 湊成一組、shared memory 怎麼擺、記憶體存取怎麼對齊，這些細節寫錯任何一個，效能就直接掉一截。Triton 的心智模型是 block，一份 kernel 描述的是「一個 program instance 一次處理一塊資料」，塊內怎麼分工給 thread、存取怎麼 coalesce，全部交給 Triton 的編譯器處理。用昨天的話說，CUDA 寫的是每個工人的動作，Triton 寫的是每塊磚的處理流程，工人怎麼分配是工頭的事。

這個高度剛好接得上 Inductor 的 IR。Day 18 的 loop-level IR 描述「迴圈的每一輪做什麼」，把一輪放大成一塊，`ops.load` 翻成 `tl.load`、`ops.store` 翻成 `tl.store`，中間的算術逐條照抄，一顆 kernel 就成形了。負責這段翻譯的是 `codegen/triton.py` 裡的 `TritonKernel`，生成器不需要懂任何 GPU 微架構，難寫對的部分讓 Triton 兜底，這正是 PyTorch 2 論文裡選 Triton 當 GPU 後端的理由，生成器可以寫得簡單，產物又常能追上手寫 CUDA 的效能。

## 逐行讀一顆 pointwise kernel

拿跟 Day 17 同一條 op 鏈當實驗品，`relu(x + y) * 2`。那天它在 CPU 上生出一個 C++ 迴圈，今天開 `TORCH_LOGS="output_code"` 在 GPU 上重編一次，輸入換成一百萬個元素的 tensor，生出來的 kernel 完整長這樣。

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

名字先講。`poi` 代表 pointwise，後面接被融進來的 op 清單，跟 Day 17 的 `cpp_fused_add_mul_relu_0` 是同一套命名邏輯，一個名字一顆 kernel，光看名字就知道 fusion 的戰果。

再來逐行讀開頭四行，這是所有 pointwise kernel 共用的模板。Inductor 先把輸出攤平成一條長度 xnumel 的線性索引，切成 XBLOCK 大小的磚。GPU 會同時發射一大批 program instance，每個 instance 用 `tl.program_id(0)` 拿到自己的編號，乘上 XBLOCK 就是自己那塊磚的起點，再加上 `tl.arange` 展開，xindex 就是這塊磚裡每條 lane 負責的元素編號。一個 instance 一次操作的是整塊 XBLOCK 長的向量，而不是單一元素，這是讀 Triton 程式最需要切換的視角。然後是 mask。一百萬除不盡 1024，最後一塊磚只有一部分是真的，`xindex < xnumel` 把越界的 lane 關掉，load 和 store 都帶著它，被遮住的 lane 不讀也不寫，尾巴就這麼安全地處理掉了。

中段就是 Day 18 的 loop body 直譯。兩筆 `tl.load` 把磚吸進來，接著加法、`maximum`（relu 在 lowering 之後的長相）、乘 2，中間值 tmp2 到 tmp6 全部活在暫存器，最後一筆 `tl.store` 把磚放回去。三個 op 一顆 kernel，讀兩次寫一次，fusion 昨天承諾的事在這十幾行裡兌現。對照 Day 17 的 C++ 版本會發現結構一模一樣，CPU 用 `at::vec` 一次搬 4 個 float，GPU 用一塊磚一次搬 XBLOCK 個，同一層 IR，兩種方言。

還有兩個容易忽略的細節。一是 `xnumel = 1000000` 這行，參數明明傳進來了，第一行卻直接用常數蓋掉，這是 Day 11 說過的 static shape 特化，shape 押死之後編譯器能做更多假設，如果當初走了 dynamic shape，這裡就會留著變數。二是 XBLOCK 宣告成 `tl.constexpr`，整份原始碼裡卻沒有它的具體數值。這是故意留白的，同一份程式配上不同的 XBLOCK 就是不同的效能，這個決定被推遲到了 launch 的時刻。

![tensor 切成 XBLOCK 大小的磚，program instance 領磚，mask 遮尾，load 算完 store 回去](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day21/triton_codegen.gif)

*圖一：一顆 pointwise kernel 的一生。輸出攤平成 xnumel 格，切成 XBLOCK 大小的磚，每個 program instance 憑 pid 算出自己的 xindex，mask 把尾端越界的 lane 遮掉，tl.load 把磚吸進暫存器，算完 tl.store 放回去，wrapper 算好 grid 把所有 instance 一次發射。*

## XBLOCK 是誰決定的

同一份輸出的下半部是 wrapper，跟 Day 17 看過的 CPU 版一樣，檢查輸入、配置 buffer、按順序呼叫 kernel（節錄）。

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

注意 `.run` 只傳了指標和 xnumel，沒有 XBLOCK 也沒有 grid。答案在 kernel 定義上方那圈 `@triton_heuristics.pointwise` 裝飾器，它拿著 size_hints 從幾組候選 config 裡挑，必要時逐一實測比快慢，這就是 autotune，原始碼在 [`runtime/triton_heuristics.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/runtime/triton_heuristics.py)。實驗最後把它實際選中的參數印了出來。

    triton_poi_fused_add_mul_relu_0: picked {'XBLOCK': 1024} num_warps=4 num_stages=1
    triton_poi_fused_add_mul_relu_0: grid = (ceil(1000000 / 1024), 1, 1) = (977, 1, 1)

XBLOCK 選了 1024，一塊磚 1024 個元素，num_warps=4 表示塊內由 128 條 thread 分工，這正是 Triton 幫忙藏掉的那一層。grid 就是磚的數量，一百萬個元素切成 977 塊，前 976 塊填滿，第 977 塊只裝 576 個，xmask 就是為它準備的。977 個 instance 撒到 GPU 的各個 SM 上同時開工，一次 launch 就此完成。這也解釋了第一次編譯慢的另一個原因，除了生程式碼，autotune 的實測也要花時間，好在挑出來的結果會跟著進快取，同樣的 kernel 之後直接沿用。

## reduction kernel 的兩種長相

pointwise 的磚彼此獨立，reduction 得把一整段收成一個值，模板自然不同。先看逐 row 的，`relu(x + 1).sum(dim=1)` 配一個 1024x1024 的輸入，編出來的 kernel 名字變成 `triton_per_fused_add_relu_sum_0`（節錄）。

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

`per` 是 persistent reduction。一個 row 有 1024 個元素，R0_BLOCK 直接開到 1024，一塊磚裝下一整個 row，load 進來、pointwise 順路做完、`tl.sum` 一口氣收掉，一個 instance 負責一個 row，昨天說的「pointwise 融進 reduction 的迴圈」生出來就是這副模樣。模板怎麼選看的是收縮段的長度，塞得進一塊磚就走 persistent，塞不進才走待會的迴圈版。索引這邊出現了新面孔，x 開頭的變數管平行維度，r 開頭的管收縮維度，`r0_1 + 1024*x0` 讀作第 x0 個 row 的第 r0_1 格，Day 19 的 MemoryDep 裡那些 index 式指的就是這個東西。另外兩筆 load 和 store 的 mask 位置都是 None，因為 1024 恰好塞滿 R0_BLOCK，沒有尾巴要遮，連檢查都省了。

換成全域的 `x.sum()` 就沒這麼舒服了，4096x4096 一千六百多萬個元素，一顆磚吞不下，只派一個 instance 收又會讓整張 GPU 閒著。Inductor 的解法是拆成兩段，第一段 `triton_red_fused_sum_0` 派 512 個 instance 各自負責 32768 個元素，`red` 代表這種帶迴圈的 reduction，核心是一個 for 迴圈抱著累加器一輪一輪吃（節錄）。

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

跑完迴圈才用 `tl.sum` 把累加器收成一個部分和。第一段吐出 512 個部分和，第二段 `triton_per_fused_sum_1` 再用一顆 persistent reduction 把這 512 個數收成最後的 scalar，wrapper 裡清楚寫著兩次 launch。

    triton_red_fused_sum_0.run(arg0_1, buf0, 512, 32768, stream=stream0)
    triton_per_fused_sum_1.run(buf0, buf1, 1, 512, stream=stream0)

多付一次 kernel launch，換到 512 倍的平行度，划算。Day 20 說全域 reduction 是一面牆，這面牆的另一個面貌在這裡，連 reduction 自己都得拆成兩顆 kernel，才餵得飽整張 GPU。

分工的參數也跟著體質走，實驗把這兩顆 kernel 被選中的 config 一併印了出來。

    triton_per_fused_add_relu_sum_0: picked {} num_warps=8 num_stages=1
    triton_red_fused_sum_0: picked {'XBLOCK': 1, 'R0_BLOCK': 2048} num_warps=16 num_stages=1

逐 row 那顆 persistent reduction 沒什麼好挑的，塊的大小被 row 定死，只挑了 num_warps=8，讓 256 條 thread 合力收一個 row。兩段式的第一段則選了 R0_BLOCK=2048 配 num_warps=16，迴圈一輪吃兩千多個元素，十六輪吃完自己負責的那一段。同一套模板，不同的 shape，分工的帳都由 heuristics 一顆一顆算。

## 自己讀 kernel 的小抄

最後整理一份判讀小抄。kernel 名字的前綴就是它的體質，`poi` 是 pointwise，`per` 是一塊磚裝下整段收縮的 persistent reduction，`red` 是帶迴圈的長跑型，這些模板的分派邏輯都在 [`codegen/simd.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/simd.py) 和 `codegen/triton.py` 裡。變數的字首對回 Day 19 的 (numel, rnumel)，x 家族管平行、r 家族管收縮。mask 出現在哪，就代表那個維度有除不盡的尾巴，mask 全變 None 則代表 shape 剛好對齊，檢查被整個省掉。launch 的參數不在 kernel 原始碼裡，想知道 XBLOCK 和 grid 最後選了什麼，得去問 heuristics。kernel 的數量則看 wrapper，call 裡每一行 `.run` 就是一次 launch，兩段式 reduction 在這裡就是兩行，跟 Day 20 拿 call 順序數 kernel 是同一招。想親眼看這一切，`TORCH_LOGS="output_code"` 一行就夠，再搭配 Day 17 的 TORCH_COMPILE_DEBUG 目錄，從 FX Graph、IR 到 kernel 的每一步變形都有存檔。

## 結語

今天把 Inductor 的 GPU 產物逐行讀完了。Triton 把寫 kernel 的單位從 thread 抬高到 block，codegen 只需要把 loop-level IR 逐行翻成 tl.load、算術、tl.store。pointwise 是一磚配一個 instance 的模板，mask 負責照顧除不盡的尾端，reduction 按大小選擇 persistent 或兩段式，XBLOCK 和 grid 則留到 launch 時刻由 heuristics 拍板。Day 2 那個只能遠觀的黑盒子，現在應該已經是可以逐行指認的老朋友了，之後想追效能問題，也知道該從哪一行看起。

不過 GPU 只是 codegen 的其中一條分流。同一層 IR 落在 CPU 上走的是另一條路，生出來的是 C++ 加 OpenMP，向量化、平行化的決策長得完全不同，Day 17 只匆匆看了一眼。明天就來把 C++ Codegen 這條路走完。那我們明天見！

## 參考資料

- [torch/_inductor/codegen/triton.py：TritonKernel（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/triton.py)
- [torch/_inductor/codegen/simd.py：SIMDScheduling（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/simd.py)
- [torch/_inductor/runtime/triton_heuristics.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/runtime/triton_heuristics.py)
- [Triton 官方文件](https://triton-lang.org/main/index.html)
- Tillet et al., [*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*](https://dl.acm.org/doi/10.1145/3315508.3329973), MAPL 2019
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）
