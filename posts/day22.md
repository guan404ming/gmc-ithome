# Day 22 | 同一條迴圈，C++ 後端有三段變速

## 前言

昨天看完 GPU 這條線，Scheduler 手上融好的 node 被寫成一份份 Triton kernel。但介紹 Inductor 總覽時就說過，codegen 是整條產線唯一分流的地方，lowering、scheduler、fusion 全部共用，最後一步才按裝置各走各的。今天換到 CPU 這條線，看 [`torch/_inductor/codegen/cpp.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/cpp.py) 怎麼把同一層 loop-level IR 寫成 C++，再交給系統編譯器變成 `.so`。先把答案放前面，cpp 後端拿到一條迴圈之後不是只有一種寫法，而是像變速箱一樣準備了三段，純量、SIMD、OpenMP，換不換檔看的是 shape。

正文開始！

## 分流之後的另一條產線

CPU 後端收到的原料跟 Triton 後端一模一樣，就是 scheduler 融好的 node，裡面還是那個用 `ops.load`、`ops.store` 描述「迴圈每一輪做什麼」的 body 函式。差別在輸出的形狀。Triton kernel 天生是「每個 program 抓一塊 tile」的平行寫法，誰跑哪一塊由 grid 決定。C++ kernel 就是一條普通的 for 迴圈，誰來跑、一步走多寬、要不要開多執行緒，全部得由 codegen 自己寫明在程式碼裡。

負責這件事的角色就在 `cpp.py` 裡，它從 scheduler 手上接過一組組 node，逐組生出 `extern "C" void kernel(...)` 這樣的 C 函式，指標進、指標出，函式本體就是迴圈。講 fusion 時提過的那個判讀陷阱在這裡也補一句，cpp 後端會把沒融合的多個 node 打包進同一個 C++ 函式，一個函式裡可能有好幾個獨立的 loop nest，數融合結果要看 loop 而不是函式。

本篇實驗都在本機 CPU 上跑（Apple M1 Max，arm64，torch 2.8.0），完整程式與 log 在 `code/day22/`。實驗品沿用介紹 Inductor 總覽時用過的函式。

```python
def f(x, y):
    return torch.relu(x + y) * 2
```

用 `TORCH_LOGS="output_code"` 把生成的 kernel 抓出來，接下來三段變速一段一段看。

## 第一段，純量迴圈

先看最素的版本。把 config 裡的向量寬度設成 1 關掉向量化，1024 個元素的輸入生出來的 kernel 長這樣（節錄）。

```cpp
    for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1024LL); x0+=static_cast<int64_t>(1LL))
    {
        auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
        auto tmp1 = in_ptr1[static_cast<int64_t>(x0)];
        auto tmp2 = float(tmp0 + tmp1);
        auto tmp3 = std::max(tmp2, decltype(tmp2)(0));
        auto tmp4 = static_cast<float>(2.0);
        auto tmp5 = float(tmp3 * tmp4);
        out_ptr0[static_cast<int64_t>(x0)] = tmp5;
    }
```

這就是 lowering 攤出來那個 body 函式的直譯，`ops.load` 變成 `in_ptr0[x0]`，`ops.relu` 變成 `std::max`，`ops.store` 變成最後那行賦值，一步走一格，一次算一個 float。中間那串 `tmp0` 到 `tmp5` 也跟 IR 裡的中間值一一對應，全是區域變數，編譯器會把它們放進暫存器，三個 op 融成一個迴圈的效果在這裡看得最清楚，讀兩筆、寫一筆，中間值不落地。

還有一件值得注意的事，這段程式碼裡沒有任何 PyTorch 的影子。沒有 dispatcher、沒有 TensorImpl，連 shape 都直接寫死成 `1024LL`，因為編譯期已經知道所有 metadata，生出來的就是一段裸的 C++。拿它當基準，後面兩段變速省的都是這個版本的時間。

## 第二段，SIMD 一步四格

把向量寬度還原成預設再編一次，同一條迴圈換了一副身體（節錄）。

```cpp
    for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1024LL); x0+=static_cast<int64_t>(4LL))
    {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
        auto tmp4 = static_cast<float>(2.0);
        auto tmp5 = at::vec::Vectorized<float>(tmp4);
        auto tmp6 = tmp3 * tmp5;
        tmp6.store(out_ptr0 + static_cast<int64_t>(x0));
    }
```

`x0` 一次前進 4 格，每個 `tmp` 都從單一 float 變成 `at::vec::Vectorized<float>`，加法還是寫 `+`、relu 換成 `clamp_min`，但每個運算元都是一整包 SIMD 暫存器。`Vectorized` 是 PyTorch 自己的跨 ISA 向量抽象，好處是 codegen 不必認識每一種指令集，永遠生同一種樣子的程式碼，`Vectorized<float>` 在編譯 `.cpp` 的時候才按目標機器展開成對應的 intrinsic。同一份原始碼，在 AVX2 的機器上一包是 8 個 float，AVX512 是 16 個，這台 M1 Max 用的 NEON 是 128-bit，所以一包 4 個。log 第一行印出的 `isa: asimd | bit_width: 128` 就是編譯期探測指令集的結果，codegen 按它決定步幅，128 除以 float 的 32 位元，正好就是迴圈裡那個 4。

原本以為小張量會退回純量版，實測發現不是。把輸入縮到只剩 3 個元素，生出來的迴圈照樣是向量版，只是 load 和 store 都多帶了一個長度參數，`loadu(in_ptr0 + x0, 3LL)` 用 masked load 只搬 3 格。也就是說在 v2.8.0 的 cpp 後端裡，向量化幾乎總是開著，零頭用遮罩處理掉，真正跟張量大小掛鉤的是下一段。

## 第三段，OpenMP 上多執行緒

把輸入放大到 `1 << 20` 個元素重編，迴圈外面多了兩行 pragma（節錄）。

```cpp
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1048576LL); x0+=static_cast<int64_t>(4LL))
```

`#pragma omp parallel` 起了 8 條 thread，數字來自 log 開頭的 `threads: 8`，也就是 `torch.get_num_threads()` 回報的值。`#pragma omp for` 再把迴圈的迭代範圍切給這 8 條 thread 分工，每條 thread 分到的那段裡面照樣是一步 4 格的 SIMD，兩層平行疊在一起，8 條 thread 乘上 4 條 lane，同一個時間點最多有 32 格在前進。值得留意的是 kernel 函式本身對此渾然不覺，它還是那個 `extern "C"` 的普通函式，平行完全由編譯進去的 OpenMP runtime 在函式內部發生，呼叫端一無所知。

那門檻在哪裡。既然先前看過 1024 個元素不開、一百萬個會開，中間必有一條線。實測掃了一輪 shape，log 是這麼說的。

```
n=16384: single thread
n=32768: omp parallel
```

這條線劃在 `cpp.py` 的 `decide_parallel_depth`，規則是總工作量除以 thread 數小於一條最小工作量門檻（預設 4096）就不開，原始碼裡的註解只有一句 not enough work。8 條 thread 乘 4096，門檻正好是 32768，跟實測的分界完全對上。thread 不是免費的，起手就要付建立與同步的成本，每條 thread 分不到幾千個元素的活，省下的時間還不夠付開工錢，攤不回來就乖乖單執行緒。這種按 shape 換檔的決策全部發生在編譯期，靠的又是 FakeTensor 一路推下來的 shape metadata，今天算是把那個現象的出處找到了。

三段變速用動畫走一遍。

![同一條 loop IR 分成 GPU 與 CPU 兩條路，CPU 這條從純量換檔到 SIMD 再換到 OpenMP](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day22/cpp_codegen.gif)

*圖一：同一條 loop-level IR 分流成兩條路，GPU 把格子直接攤給上千個 thread，CPU 這條先是一格一格走的純量迴圈，SIMD 把 4 格併成一步，OpenMP 再把整條迴圈切給 8 個 worker，換檔與否由 shape 在編譯期決定。*

## reduction 要自己收尾

pointwise 切一切就能分工，reduction 不行，8 條 thread 各自加各自的，最後總得有人把帳合起來。拿 `(x * x).sum()` 編一次，生成的 kernel 把這件事的全貌攤開了（節錄）。

```cpp
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        at::vec::Vectorized<float> tmp_acc0_vec_local = at::vec::Vectorized<float>(0);
        #pragma omp for
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1048576LL); x0+=static_cast<int64_t>(4LL))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
            auto tmp1 = tmp0 * tmp0;
            tmp_acc0_vec_local = tmp_acc0_vec_local + tmp1;
        }
        tmp_acc0_vec_arr[tid] = tmp_acc0_vec_local;
    }
    for (int tid = 0; tid < 8; tid++)
    {
        tmp_acc0_vec = tmp_acc0_vec + tmp_acc0_vec_arr[tid];
    }
    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
```

有趣的是 `#pragma omp for` 後面並沒有掛教科書式的 `reduction(+:...)` 子句，Inductor 選擇自己把這件事攤開來寫。每條 thread 抱著一個私有的向量累加器 `tmp_acc0_vec_local`，掃完自己的地盤後存進以 `tid` 為索引的陣列，離開平行區之後由主執行緒把 8 份加總，最後再用 `vec_reduce_all` 把向量裡的 4 條 lane 收成一個數。私有累加器是平行 reduction 的標準解法，8 條 thread 要是直接往同一個變數上加，不是搶成一團就是每次都要上鎖，各記各的帳、最後合併一次，貴的同步只發生在收尾那一下。兩層平行怎麼開的，就得兩層各收一次尾，這正是前面說的「全部寫明在程式碼裡」的具體長相。

## 跟 Triton 那條線對照

把兩個後端的產物擺在一起，差的其實是對硬體的想像。Triton kernel 假設 thread 要多少有多少，codegen 只描述一個 program 算哪塊 tile，launch 幾千個 program 是常態，甚至鼓勵超額，讓硬體用排程把記憶體延遲藏起來，向量化、warp 怎麼跑這些事全部下放給 Triton 編譯器和 GPU。C++ kernel 面對的是一台 thread 個位數、起 thread 要付真金白銀成本的機器，所以平行不平行要算過才開，SIMD 寬度要明碼寫進指令，效能靠的不是人海而是每條 thread 順著 cache 走。

兩邊處理零頭的手法倒是異曲同工。Triton kernel 靠 mask 讓越界的 lane 不作數，cpp 後端靠帶長度參數的 masked `loadu`，都是同一個思路，向量寬度是硬體定的，資料長度是使用者給的，對不齊的部分用遮罩補平。同一層 IR，GPU 那條線把「怎麼平行」交出去，CPU 這條線把「怎麼平行」一筆一筆自己寫完，這就是 thread 模型的差異在 codegen 上留下的痕跡。

## 從 .cpp 到 .so

最後看產物怎麼落地。Triton 有自己的編譯 pipeline，cpp 後端借的則是系統編譯器。把編譯命令攔下來看，實測長這樣（節錄）。

```
compile cmd: clang++ .../ck3n4ozn...main.cpp ... -shared -fPIC ... -O3 -DNDEBUG ... -Xclang -fopenmp ... -o .../ck3n4ozn...main.so -lomp ...
```

組裝命令的人是 [`cpp_builder.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cpp_builder.py)，它按平台挑編譯器、湊齊 include 路徑和連結參數，在這台 Mac 上用的是 clang++ 加 Homebrew 裝的 libomp。生成的 kernel 以內容 hash 命名存成 `.main.cpp`，開著 `-O3` 和 `-fopenmp` 編成同名的 `.main.so`，翻 cache 目錄就能看到成對躺著的兩個檔案，旁邊還有一份 wrapper 的 `.py`。wrapper 再把這個動態庫載回 Python，之後每次呼叫都直接進 C++，那條改寫過的 bytecode 呼叫下來，最後落點就是這個 `.so` 裡的函式。編譯一次要花上一兩秒，所以先前講過的快取在 CPU 後端格外有感，hash 沒變就不再叫醒 clang++。

## 結語

CPU 後端的 codegen 今天走完了。同一份 loop-level IR，cpp 後端用三段變速把它寫成 C++，純量版是語意的直譯，SIMD 版靠 `at::vec::Vectorized` 一步多格而且幾乎總是開著，OpenMP 版在工作量攤得回 thread 成本時才掛檔，reduction 則要每層平行各收一次尾。最後由系統編譯器把 `.cpp` 鑄成 `.so`，快取記住一切。

不過到目前為止，不管哪個後端，一個 kernel 都只有「一種生法」。但同一個運算其實常有好幾種寫法可選，tile 怎麼切、迴圈怎麼排，效能可以差好幾倍，矩陣乘尤其明顯。Inductor 的辦法很實在，把候選寫法都生出來、真的跑一遍、用碼表挑冠軍。明天就來看 Autotune 這場比賽怎麼辦。那我們明天見！

## 參考資料

- [torch/_inductor/codegen/cpp.py：CppVecKernel 與 decide_parallel_depth（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/cpp.py)
- [torch/_inductor/cpu_vec_isa.py：pick_vec_isa（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cpu_vec_isa.py)
- [torch/_inductor/cpp_builder.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cpp_builder.py)
- [torch/_inductor/config.py：cpp.min_chunk_size（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/config.py)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）
