# Day 17 | torch.compile 的程式碼鑄造廠：TorchInductor

## 前言

昨天用 FakeTensor 把「不碰真資料也能知道每個 node 的 shape 與 dtype」這件事講完，Part 2 的 AOTAutograd 篇也就收尾了。回頭看 Day 2 畫的 pipeline 地圖，前兩站已經走完。Dynamo 把 Python 攔下來抓成 FX Graph，AOTAutograd 把它展開成 forward 與 backward 兩張乾淨的 ATen 圖。從今天起進入 Part 3，主角是預設後端 TorchInductor，負責把圖真的變成機器上跑得動的程式碼。今天先不鑽細節，把整條產線從頭到尾走一遍、把地圖畫好，接下來幾天再一站一站放大。

正文開始！

## Inductor 接手的是一張什麼樣的圖

先講清楚原料。Inductor 拿到的不是 Dynamo 那張還留著 `add_`、`view` 的圖，而是 AOTAutograd 加工完的版本。Functionalization 把 mutation 和 aliasing 洗掉了，Decomposition 把兩千多個 ATen op 收斂成幾百個核心 op，min-cut partitioner 把 forward 與 backward 分成兩張圖，而每個 node 上都掛著 FakeTensor 推好的 shape、dtype、stride。純函數式、詞彙表小、metadata 齊全，這是一張對編譯器最友善的圖。

也因此 Inductor 的職責可以收得很窄。它不需要懂 Python 的動態，也不需要懂 autograd，只需要回答一個問題，這張 ATen 圖怎麼變成最少、最快的 kernel。前面站點的每一層正規化，都是在幫這一站減負，Dynamo 擋掉了 Python 的不可預測，AOTAutograd 擋掉了 mutation 和高階 op 的組合爆炸，輪到 Inductor 時，問題已經被削成純粹的編譯問題。

另外值得再提醒一次 Day 2 講過的事，這一站是可以換掉的。前兩站交出來的是標準的 FX Graph，任何吃 FX Graph 的東西都能接在這裡當 backend，Inductor 只是 PyTorch 自帶、預設、也最成熟的那一個。今天之後講的所有機制，都是這個預設選項的內部。

## 為什麼需要自己的 IR

第一個念頭可能是，圖都這麼乾淨了，一個 node 生一個 kernel 不就好了。這其實就是 eager mode 的做法，而 Day 2 量過，elementwise 運算的瓶頸是記憶體頻寬，逐 op 執行讓中間結果一直在記憶體之間來回，加速的最大來源正是把好幾個 op 融進同一個 kernel，資料讀一次、一路算完、寫一次。

但要決定哪些 op 能融在一起，op 等級的 node 不夠用。`aten.add` 對排程器來說是一個黑盒子，看不出它讀哪些記憶體、寫哪些記憶體、迴圈長什麼樣。所以 Inductor 定義了自己的中間表示，把每個 op 攤開成一個描述「迴圈的每一輪在做什麼」的 Python 函式，每一筆讀取和寫入都變成顯式的 `ops.load`、`ops.store`。PyTorch 把這個設計叫 define-by-run IR，IR 本身就是可執行的 Python。今天只需要知道有這一層以及它存在的理由，後面 scheduler 做 fusion、後端生程式碼，依據的都是這層 IR 而不是 FX Graph，細節留到明天。

## 從 compile_fx 走一遍管線

Inductor 的正門在 [`torch/_inductor/compile_fx.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/compile_fx.py) 的 `compile_fx`。翻原始碼會發現一件有趣的事，AOTAutograd 其實是被它呼叫的。`compile_fx` 拿到 Dynamo 的圖後先跑一輪 pre-grad passes，接著把 `fw_compiler` 和 `bw_compiler` 交給 `aot_autograd`，讓它展開出兩張 ATen 圖，再各自送回 `compile_fx_inner` 手上。所以 Day 2 說 Inductor 是第三站，是使用者視角的說法，以呼叫關係來說是 Inductor 把第二站包了起來。

從 `compile_fx_inner` 開始，一張圖固定走四步。

1. **graph passes**：還在 FX 層，joint graph passes 與 post-grad passes 做 pattern matching 式的最後整理。
2. **lowering**：`GraphLowering.run()` 逐 node 走訪 FX Graph，查一張對照表把每個 ATen op 翻成上面說的 loop-level IR。這張表在 [`lowering.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/lowering.py)，實測有 1713 條 entry。
3. **scheduling**：[`scheduler.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/scheduler.py) 的 `Scheduler` 接手所有 IR node，決定執行順序、哪些 node 融成同一個 kernel、哪些 buffer 可以重用。
4. **codegen**：每一組融合好的 node 交給後端生出 kernel 原始碼，GPU 上生 Triton，CPU 上生 C++ 加 OpenMP，最後編譯、載入、寫進快取。

四步走完，`graph.compile_to_module()` 把生成的原始碼寫進快取目錄、丟給編譯器，組裝成一個可以直接呼叫的 Python module，交還給 Dynamo 改寫後的 bytecode 使用。快取這件事做得很徹底，實驗時 log 印出 `Output code written to: .../torchinductor_wchiu/...` 這樣的路徑，生成的程式碼以內容的 hash 命名存在磁碟上。第二次跑同一個實驗時，整張圖直接命中 FX graph cache，連中間產物都不再產生，這也是為什麼同一個模型第二次編譯會快很多。

整個流程用動畫走一遍。

![ATen 圖沿著 LOWER、SCHEDULE、CODEGEN 三道閘一路變形，最後吐出 kernel](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day17/inductor_overview.gif)

*圖一：一張 ATen 圖掉進 Inductor 的流水線。LOWER 把每個 op 攤開成 loop-level IR，SCHEDULE 把能融的 node 合併成一組，CODEGEN 按裝置分流，GPU 生 Triton、CPU 生 C++，最後由 wrapper 把 kernel 組裝成可呼叫的 module。*

## 實際看 CPU 後端吐出什麼

實驗跟昨天一樣在本機 CPU 上跑，`torch 2.8.0`，正好可以驗證雙後端這件事。同一條管線，codegen 在 CPU 上生的是 C++，在 GPU 上生的則是 Triton kernel，Day 2 結尾節錄過後者的長相，可以翻回去對照。拿一條三個 pointwise op 的鏈當實驗品。

```python
def f(x, y):
    return torch.relu(x + y) * 2

torch._logging.set_logs(output_code=True)
torch.compile(f)(torch.randn(1024), torch.randn(1024))
```

`output_code` 印出 Inductor 最終的產物，kernel 的部分長這樣（節錄）。

```cpp
cpp_fused_add_mul_relu_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
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
}
''')
```

名字 `cpp_fused_add_mul_relu_0` 已經把重點講完了，`add`、`relu`、`mul` 三個 op 融成了一個 kernel。在 eager mode 下這是三次獨立的 kernel 呼叫，中間結果要寫回記憶體再讀出來兩次，這裡整段程式只有一個迴圈，兩筆輸入各讀一次，中間結果 `tmp2`、`tmp3` 只活在暫存器裡，最後寫一次。`at::vec` 則是 CPU 後端自動做的 SIMD 向量化，`x0` 一次前進 4 個 float。

輸入的大小也會改變生成的程式碼。把兩個輸入從 1024 換成 `1 << 20` 個元素重編一次，同一條 op 鏈生出的 kernel 外面多了一圈，實測節錄長這樣。

```cpp
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1048576LL); x0+=static_cast<int64_t>(4LL))
```

小張量單執行緒跑完就好，大張量才值得付出 thread 啟動的開銷，Inductor 靠 OpenMP 把迴圈切給 8 條 thread。這種按 shape 量身訂做的決策，正是編譯期知道 shape 的好處，也就是 FakeTensor 一路推下來的 metadata 在這裡兌現。

同一份產物裡還有一段 Python，這就是 wrapper code。

```python
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, ), (1, ))
    assert_size_stride(arg1_1, (1024, ), (1, ))
    buf0 = empty_strided_cpu((1024, ), (1, ), torch.float32)
    cpp_fused_add_mul_relu_0(arg0_1, arg1_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )
```

kernel 只是零件，wrapper 是組裝說明書。它檢查輸入的 size 和 stride、配置輸出 buffer、按正確順序呼叫每一個 kernel、及時 `del` 釋放引用。Day 9 那條 bytecode 裡的 `__compiled_fn` 被呼叫之後，最後真正執行的就是這個 `call`。真實模型會生出幾十個 kernel，wrapper 就是串起它們的那條主線。另外 wrapper 預設生成 Python，好讀好改，但也有 `torch._inductor.config.cpp_wrapper` 這個開關可以改生 C++ 版本，把 Python 呼叫的 overhead 也省掉。

中間產物也留得下痕跡。設環境變數 `TORCH_COMPILE_DEBUG=1` 重跑一次，Inductor 會把每一站的產物寫進 `torch_compile_debug/` 目錄，實測列出來有 `fx_graph_readable.py`、`ir_pre_fusion.txt`、`ir_post_fusion.txt`、`output_code.py` 這些檔案，正好對應進廠的 FX Graph、scheduler 前後的 IR、最終程式碼。打開 `ir_pre_fusion.txt`，loop-level IR 就在裡面（節錄）。

```python
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        relu = ops.relu(add)
        constant = ops.constant(2.0, torch.float32)
        mul = ops.mul(relu, constant)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, mul, None)
        return store
```

對照上面的 C++ kernel，`ops.load` 對到 `loadu`、`ops.add` 對到 `tmp0 + tmp1`，一行一行都找得到影子。這個 `body` 就是那個「描述迴圈每一輪在做什麼」的 Python 函式，後端只是把它用另一種語言重新念了一遍。同一份檔案的上半部還躺著 scheduler 關心的資訊，例如 `op0.writes` 與 `op0.met_dependencies` 記著這個 node 讀了 `arg0_1`、`arg1_1`、寫了 `buf0`，`op0.group.iteration` 記著迭代空間是 1024。哪些 node 讀寫同一塊 buffer、迭代空間合不合得起來，fusion 的判斷靠的全是這些欄位，這也回應了前面的問題，為什麼 op 等級的 node 不夠用。

這套目錄在除錯時非常好用。編譯結果不對的時候，沿著 `fx_graph_readable.py`、`ir_pre_fusion.txt`、`output_code.py` 一路往下比對，就能定位問題出在哪一站，算是 Day 2 用 backend 參數切 pipeline 的細粒度版本。

## 兩個後端，一套管線

值得強調的是分家的位置。lowering、IR、scheduler 全部共用，只有最後 codegen 一步按裝置分流，CPU 走 [`codegen/cpp.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/cpp.py)，GPU 走 [`codegen/triton.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/triton.py)，wrapper 的生成則統一在 [`codegen/wrapper.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/wrapper.py)。這個切法的好處很直接，fusion 決策、記憶體規劃這些難的部分不必每個後端重寫一遍，支援一種新硬體，理論上只要補上最後一段 codegen，這正是自己養一層 IR 的回報。

GPU 那一側還有一個值得記下的選擇。Inductor 生的不是 CUDA C 而是 Triton，一種用 Python 寫 GPU kernel 的語言。原因是 Triton 把 thread 排布、記憶體 coalescing 這些難寫對的細節接手掉，codegen 只需要描述 tile 等級的計算，生成器簡單得多，生出來的 kernel 又常能追上手寫 CUDA 的效能，這是 PyTorch 2 論文特別強調的取捨。Day 2 結尾那個 `triton_poi_fused_add_cos_mul_sin_tanh_0` 的 kernel，就是這條路徑的產物，跟今天的 C++ kernel 對照著看，同一層 IR、兩種長相。

## 之後三站的地圖

今天走馬看花的三道閘，接下來各用一天放大。

- **Day 18 Lowering 與 Loop-level IR**：`aten.add` 是怎麼被攤開成那個 `body` 函式的，define-by-run 這個設計到底在解什麼問題，以及 Pointwise、Reduction 這些 IR node 的長相。
- **Day 19 Scheduler**：IR node 之間的依賴怎麼建、順序怎麼排，`ir_pre_fusion.txt` 和 `ir_post_fusion.txt` 之間發生了什麼。
- **Day 20 Fusion**：加速的最大來源單獨講一天，什麼能融、什麼不能融、Inductor 怎麼給每個融合機會打分。

## 結語

把今天的地圖收攏一下。Inductor 從 AOTAutograd 手上接過純函數式、metadata 齊全的 ATen 圖，先 lower 成 define-by-run 的 loop-level IR,scheduler 在 IR 上決定融合與順序，codegen 按裝置生出 Triton 或 C++ kernel,wrapper 負責把零件組裝成一個可呼叫的 module，成品進快取，下次直接取用。三個 pointwise op 進來，一個 kernel 出去，這就是這座鑄造廠的日常。

明天先從第一道閘開始，把 lowering 和 loop-level IR 打開來看，搞懂這層 IR，後面的 scheduler 和 fusion 才讀得懂。那我們明天見！

## 參考資料

- [torch/_inductor/compile_fx.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/compile_fx.py)
- [torch/_inductor/graph.py：GraphLowering（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/graph.py)
- [torch/_inductor/lowering.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/lowering.py)
- [torch/_inductor/scheduler.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/scheduler.py)
- [torch/_inductor/codegen/wrapper.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codegen/wrapper.py)
- Ansel et al., [*PyTorch 2*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024（第 5 節）
