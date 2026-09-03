# Day 22 | Inductor Codegen 的 C++ 後端有三段變速？

## 前言

昨天看完 GPU 這條線，融好的 node 被寫成一份份 Triton kernel。codegen 是整條產線唯一分流的地方，前面的步驟全部共用，最後一步才按裝置各走各的。今天換到 CPU 這條線，看 Inductor 怎麼把同一層 loop-level IR 寫成 C++，再交給系統編譯器變成 `.so`。這裡先小小透漏一下其實 cpp 後端 codegen 完拿到的一條迴圈不是只有一種寫法，而是像變速箱一樣準備了三段。

- **純量**：一次只算一個數字，最樸素的迴圈。
- **SIMD**：一道 instruction 同時算好幾個數字。
- **OpenMP**：把整條迴圈切開，交給好幾條 thread 一起跑。

決定換不換檔，取決於當下的張量有多大。

正文開始！

## 分流之後的另一條產線

CPU 後端收到的原料跟 Triton 後端一樣，就是 scheduler 融好的 node，裡面帶著一份「迴圈每一輪做什麼」的說明，差別在輸出的形狀。Triton kernel 天生是平行寫法，誰跑哪一塊由硬體排程決定。C++ kernel 就是一條普通的 for 迴圈，誰來跑、一步走多寬、要不要開多執行緒，全得由 codegen 自己寫明在程式碼裡。

cpp 後端接過一組組 node，逐組生出 `extern "C" void kernel(...)` 這樣的 C 函式，指標進、指標出，本體就是迴圈。這裡補一個判讀陷阱，沒融合的多個 node 也會被打包進同一個 C++ 函式，一個函式裡可能有好幾條各自獨立的迴圈，數融合結果要看迴圈不是函式。

本篇實驗都在本機 CPU 上跑（Apple M1 Max，arm64，torch 2.8.0），完整程式與 log 在 `code/day22/`。實驗品是這個函式。

```python
def f(x, y):
    return torch.relu(x + y) * 2
```

用 `TORCH_LOGS="output_code"` 把生成的 kernel 抓出來，三段變速一段一段看。

## 第一段，純量迴圈

先看最素的版本。把向量寬度設成 1 關掉向量化，1024 個元素的輸入生出的 kernel 長這樣（節錄）。

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

這就是 loop-level IR 的直譯，讀值變成 `in_ptr0[x0]`，relu 變成 `std::max`，寫值變成最後那行賦值，一次算一個 float。`tmp0` 到 `tmp5` 全是區域變數，編譯器會把它們放進暫存器。三個 op 融成一條迴圈的效果這裡最清楚，讀兩筆、寫一筆，中間值不落地。

另外這段程式碼裡沒有任何 PyTorch 的影子，連 shape 都寫死成 `1024LL`。編譯期已經知道所有 metadata，生出來的就是一段裸的 C++。拿它當基準，後面兩段變速省的都是這個版本的時間。

## 第二段，SIMD 一步四格

把向量寬度還原成預設再編一次，同一條迴圈換了副身體（節錄）。

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

`x0` 一次前進 4 格，每個 `tmp` 從單一 float 變成 `at::vec::Vectorized<float>`。加法還是寫 `+`，relu 換成 `clamp_min`，但每個 operand 都是一整包數字。`Vectorized` 是 PyTorch 自己的向量抽象，codegen 因此不必認識每種指令集，永遠生同一種程式碼，編譯 `.cpp` 時才按目標機器展開成真正的 instruction。同一份原始碼，換台機器一包可能是 8 個或 16 個 float，這台 M1 Max 一包 4 個。log 第一行的 `isa: asimd | bit_width: 128` 就是編譯期探測到的結果，128 除以 float 的 32 位元，正好是迴圈裡那個 4。

原本以為小張量會退回純量版，實測不是。把輸入縮到只剩 3 個元素，迴圈照樣是向量版，只是 load 和 store 多帶了長度參數，用遮罩擋掉多出來的那一格。也就是說向量化幾乎總是開著，零頭用遮罩處理掉，真正跟張量大小掛鉤的是下一段。

## 第三段，OpenMP 上多執行緒

把輸入放大到 `1 << 20` 個元素 recompile，迴圈外多了兩行 pragma（節錄）。

```cpp
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1048576LL); x0+=static_cast<int64_t>(4LL))
```

`#pragma omp parallel` 起了 8 條 thread，數字來自 log 開頭的 `threads: 8`。`#pragma omp for` 再把迴圈的範圍切給這 8 條 thread 分工。每條 thread 分到的那段照樣是一步 4 格的 SIMD，兩層平行疊在一起，8 條 thread 乘 4 格，同一時間最多有 32 格在前進。值得留意的是 kernel 函式本身渾然不覺，它還是那個普通的 C 函式，平行完全發生在函式內部，呼叫端一無所知。

那門檻在哪裡。1024 個元素不開、一百萬個會開，中間必有一條線。實測掃了一輪 shape，log 是這麼說的。

```
n=16384: single thread
n=32768: omp parallel
```

規則其實很土法煉鋼，總工作量除以 thread 數，小於門檻（預設 4096）就不開。8 條 thread 乘 4096 正好是 32768，跟實測的分界完全對上。thread 不是免費的，起手就要付建立與同步的成本，每條 thread 分不到幾千個元素的活，省下的時間不夠付開工錢，攤不回來就乖乖單執行緒。這種按 shape 換檔的決策全發生在編譯期，靠的還是編譯前就推好的 shape metadata。

三段變速用動畫走一遍。

![同一條 loop IR 分成 GPU 與 CPU 兩條路，CPU 這條從純量換檔到 SIMD 再換到 OpenMP](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day22/cpp_codegen.gif)

*圖一：同一條 loop-level IR 分流成兩條路，GPU 把格子直接攤給上千個 thread，CPU 這條先是一格一格走的純量迴圈，SIMD 把 4 格併成一步，OpenMP 再把整條迴圈切給 8 個 worker，換檔與否由 shape 在編譯期決定。*

## reduction 要自己收尾

pointwise 切一切就能分工，reduction 不行，8 條 thread 各加各的，最後總得有人把帳合起來。拿 `(x * x).sum()` 編一次，生成的 kernel 把全貌攤開了（節錄）。

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

有趣的是 Inductor 沒用 OpenMP 內建的 reduction 子句，而是自己攤開來寫。每條 thread 抱著私有的累加器，掃完自己的地盤存進陣列，離開平行區後由主執行緒把 8 份加總，最後再把一包裡的 4 個數字收成一個。私有累加器是標準解法，8 條 thread 要是直接往同一個變數加，不是搶成一團就是每次都要上鎖。各記各的帳、最後合併一次，貴的同步只發生在收尾那一下。兩層平行怎麼開的，就得兩層各收一次尾，這正是前面說的「全部寫明在程式碼裡」。

## 跟 Triton 那條線對照

把兩個後端的產物擺在一起，差的其實是對硬體的想像。昨天的 Triton kernel 假設 thread 要多少有多少，一次 launch 幾千個是常態，怎麼平行全下放給 Triton 編譯器和 GPU。C++ kernel 面對的是 thread 只有個位數、起 thread 要付真金白銀的機器，平行不平行要算過才開，一步走幾格要明碼寫進程式碼，效能靠的不是人海而是每條 thread 順著 cache 走。同一層 IR，GPU 那條線把「怎麼平行」交出去，CPU 這條線一筆一筆自己寫完。

## 從 .cpp 到 .so

最後看產物怎麼落地。Triton 有自己的編譯 pipeline，cpp 後端借的是系統編譯器。把編譯命令攔下來，實測長這樣（節錄）。

```
compile cmd: clang++ .../ck3n4ozn...main.cpp ... -shared -fPIC ... -O3 -DNDEBUG ... -Xclang -fopenmp ... -o .../ck3n4ozn...main.so -lomp ...
```

Inductor 按平台挑編譯器、湊齊 include 路徑和連結參數，這台 Mac 用的是 clang++。生成的 `.cpp` 開著 `-O3` 和 `-fopenmp` 編成 `.so`，再由一段 wrapper 把它載回 Python。改寫過的 bytecode 呼叫下來，落點就是這個 `.so` 裡的函式。編譯一次要一兩秒，所以快取在 CPU 後端格外有感，程式碼沒變就不再叫醒編譯器。

## 結語

CPU 後端的 codegen 今天走完了。同一份 loop-level IR，cpp 後端用三段變速寫成 C++。純量版是語意的直譯，SIMD 版一步多格而且幾乎總是開著，OpenMP 版要工作量攤得回 thread 成本才掛檔，reduction 則是每層平行各收一次尾。最後由系統編譯器把 `.cpp` 鑄成 `.so`。

不過到目前為止，不管哪個後端，一個 kernel 都只有「一種生法」。但同一個運算常有好幾種寫法可選，效能可以差好幾倍，矩陣乘尤其明顯。Inductor 的辦法很實在，把候選都生出來、真的跑一遍、用碼表挑冠軍。明天就來看 Autotune 這場比賽怎麼辦。那我們明天見！

## 參考資料

- [torch/_inductor/codegen/cpp.py：CppVecKernel 與 decide_parallel_depth（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/cpp.py)
- [torch/_inductor/cpu_vec_isa.py：pick_vec_isa（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cpu_vec_isa.py)
- [torch/_inductor/cpp_builder.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/cpp_builder.py)
- [torch/_inductor/config.py：cpp.min_chunk_size（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/config.py)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）

