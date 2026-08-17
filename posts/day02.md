# Day 2 | torch.compile 之後，你的 Python 去哪了？

## 前言

昨天我們簡單的把 PyTorch 的發展史介紹了一下，看到 `torch.compile` 是 PyTorch 在「保住 Eager 語意」的前提下，第三次嘗試把計算圖拿回來的成果。那今天換成使用者的視角來看這裡發生了什麼事：這一行程式碼被呼叫之後，從你的 Python 函式到 GPU 上真正執行的程式，中間到底經過了哪幾站？

今天的目標很單純：把這條 pipeline 的四個階段講清楚，實際跑幾段程式去感受一下加速，然後用一個 `backend` 參數，去把四個階段一層一層剝開。之後將近一個月的內容，就是沿著這條 pipeline 一段一段往下鑽，所以今天就一起來先把地圖畫好吧！正文開始！

## 一行程式碼背後的四站

先看最常見的用法：

```python
import torch

model = torch.compile(model)
```

這一行執行完的當下幾乎什麼都沒發生，它只是把 `model` 包了一層。真正的工作要等到你第一次帶著真實輸入呼叫它，那一刻才會一口氣走完下面四個階段：

![torch.compile 的四段pipeline](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day02/pipeline.gif)

*圖一：torch.compile 的四段 pipeline：Dynamo 擷取、AOTAutograd 展開、Inductor 生成程式碼、Runtime 執行。*

**第一站是 TorchDynamo，負責擷取。** 昨天提過，它利用 CPython 的 Frame Evaluation Hook，在你的 Python Bytecode 執行的當下把它攔下來，一條一條指令模擬執行，把能編的 Tensor 運算「錄」成一張 FX Graph。碰到它看不懂的 Python，例如依賴 Tensor 數值的 `if`、`print`、第三方函式庫的呼叫，就在那裡斷開，這叫 Graph Break：前半段圖交給後面編譯，斷點處退回一般 Python 執行，之後再接回來。同時它會裝一組 Guard，記住這張圖成立的前提，例如輸入的 shape、dtype、某個 Python 常數的值。下次再呼叫時先檢查 Guard，全部通過就直接用編好的結果，否則就重新編譯。

**第二站是 AOTAutograd，負責展開與正規化。** Dynamo 抓到的只是 forward 圖。AOTAutograd 拿到之後，會用 Autograd 引擎把 backpropagation 也 Trace 出來，變成一張 forward 圖加一張 backward 圖，這樣訓練也能整張圖被編譯，這也是它名字裡 Ahead-of-Time 的意思。同時它會做 Functionalization，把 In-place 修改、View 這些會讓圖變得難以最佳化的東西改寫成純函數式，再透過 Decomposition 把高階 Operator 拆解成一組更小的基本運算，讓後端只需要處理少量的 Operator。

**第三站是 TorchInductor，負責生成程式碼。** 這是預設的後端。它把圖 Lower 成一種以 Loop 為單位的中間表示，決定哪些運算可以融合成同一個 Kernel、怎麼排程，然後真的生出程式碼：GPU 上是 Triton，CPU 上是 C++ 加 OpenMP。生出來的程式碼會被寫到快取目錄、編譯、載入。這一站是可以換掉的：前兩站交出來的是一張標準的 FX Graph，任何吃 FX Graph 的東西都能接在這裡當後端，例如 TVM、TensorRT、ONNX Runtime，或是你自己寫的一個 Python 函式。Inductor 只是 PyTorch 自帶、也最成熟的那一個。

**第四站是 Runtime，負責執行。** 載入生成好的 Kernel、管理快取、必要時再掛上 CUDA Graph 之類的機制把 Kernel 啟動的開銷壓掉。

昨天的歷史課有說「ML Compiler 本質上是擷取、最佳化、生成程式碼三步」，這四段就是那個骨架的具體填法：Dynamo 是擷取，AOTAutograd 加 Inductor 前半是最佳化，Inductor 後半是生成程式碼。差別在於，這次每一段我們都會打開來讀它真正吐出的東西。

## 加速從哪來？

在拆解之前，先確認這條 pipeline 真的有用。這個系列的程式都丟到 [Modal](https://modal.com/) 上跑，GPU 是 L40S，PyTorch 2.8.0 加 CUDA 12.8。

拿一段會被記憶體頻寬拖住的 Elementwise 運算來比：

```python
import statistics
import torch

def f(x):
    return torch.sin(x) * torch.cos(x) + torch.tanh(x)

x = torch.randn(4096, 4096, device="cuda")
f(x)  # eager 也暖身一次，把第一次載入 CUDA kernel 的時間排除

compiled = torch.compile(f)
compiled(x)  # 第一次呼叫才會真正編譯，先暖身

def bench(fn, iters=100):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def bench10(fn):  # 每輪 100 次取平均，跑 10 輪再取平均與標準差
    r = [bench(fn) for _ in range(10)]
    return statistics.mean(r), statistics.stdev(r)

print("eager   ", bench10(f))
print("compiled", bench10(compiled))
```

在 L40S 上跑出來的結果：

```
torch 2.8.0+cu128 | NVIDIA L40S
first call (compile + run): 1592.9 ms
eager    1.171 ms (+/- 0.001)
compiled 0.191 ms (+/- 0.001)
speedup  6.12x
```

同一個函式、同一張卡，編譯後快了六倍，十輪的標準差只有 0.001 ms，數字的表現很穩定。但更值得注意的是第一行：第一次呼叫花了 1.5 秒，是之後每次呼叫的八千倍。這段程式裡有兩個細節，而正是之後每一篇都有可能會踩到的坑，我們先筆記起來：

第一，**編譯是延遲的**。`torch.compile(f)` 只是包一層，真正的編譯發生在第一次用真實輸入呼叫的時候，而且是一口氣把 Dynamo、AOTAutograd、Inductor 三段全部跑完。所以第一次呼叫特別慢，慢的是編譯本身而不是執行。量測前一定要先暖身一次，把編譯時間排除在計時之外。

第二，**加速從哪裡來**。`sin`、`cos`、`tanh`、乘、加這五個運算，在 Eager Mode 下是五個獨立的 Kernel，每一個都把 `x` 從 GPU 的全域記憶體讀進來、算完再寫回去，中間結果明明下一步馬上要用，卻先乖乖寫回去再重新讀出來。Inductor 會把這五個運算融合成一個 Triton Kernel，資料讀一次、一路算完、寫一次。這種 Elementwise 運算的瓶頸從來不是算得多快，而是資料搬得多快，所以少掉四趟來回，速度就差很多。

可以粗略算一下這個六倍是哪來的。`x` 是 4096 x 4096 個 float32，64 MB。Eager 下 `sin`、`cos`、`tanh` 各讀 64 MB 寫 64 MB，`mul` 和 `add` 各讀 128 MB 寫 64 MB，加起來大約 768 MB 的記憶體流量；fuse 後只剩讀 64 MB 寫 64 MB，128 MB。768 除以 128 剛好是 6，跟量到的 6.12 倍幾乎一樣。這不是巧合，這個運算就是純粹被記憶體頻寬綁住的，省掉多少流量就快多少。

## backend：pipeline 的斷點

現在是今天的重點。`torch.compile` 有一個 `backend` 參數，它剛好對應到 pipeline 的斷點，讓你只跑前面幾段、把後面關掉：

```python
f_dynamo = torch.compile(f, backend="eager")      # 只跑 Dynamo
f_aot    = torch.compile(f, backend="aot_eager")  # Dynamo + AOTAutograd
f_full   = torch.compile(f, backend="inductor")   # 完整 pipeline，這是預設值
```

| backend | Dynamo 擷取 | AOTAutograd 展開 | Inductor 生成程式碼 |
| --- | --- | --- | --- |
| `"eager"` | 有 | 無 | 無 |
| `"aot_eager"` | 有 | 有 | 無 |
| `"inductor"` | 有 | 有 | 有 |

三個名字很容易誤導，這邊先給個簡單的解釋。`backend="eager"` 不是「不編譯」。Dynamo 照樣攔截你的 Bytecode、建圖、裝 Guard，只是最後不生成新的 Kernel，而是把圖裡的 Operator 照原樣用 Eager 跑。所以它跑起來跟原生差不多快，但該 Graph Break 的地方一樣會斷、該 Recompile 的地方一樣會重編。`backend="aot_eager"` 則是多接一段 AOTAutograd：圖被展開、正規化、拆解，但還是用 Eager 執行每個 Operator。

實際跑一次三個 backend：

```
backend=eager      1.171 ms (+/- 0.001)
backend=aot_eager  1.172 ms (+/- 0.001)
backend=inductor   0.191 ms (+/- 0.000)
```

前兩個跟原生 eager 的 1.171 ms 幾乎一樣，因為它們最後都還是一個一個 Operator 用 eager 跑，Dynamo 和 AOTAutograd 做的事都在編譯期，執行期看不出差別。加速全部發生在最後一段，也就是 Inductor 真的生出融合 Kernel 的那一刻。

三個 backend 是層層包住的：`inductor` 包含 `aot_eager`，`aot_eager` 包含 `eager`，每往後一個就多接一段 pipeline。理解這層關係，它就變成一個非常好用的除錯工具。假設你的模型編譯後結果不對，先換成 `backend="eager"`：如果這樣就錯，問題在 Dynamo 的擷取；如果 `eager` 沒事、`aot_eager` 才錯，問題在 AOTAutograd；如果只有 `inductor` 才錯，那就是生成程式碼那段的問題。透過這個參數，我們就把能四段 pipeline 切成可以各自檢查的區段，筆者個人認為這是個在工程上以及實用程度上都相當巧妙的設計。

而如果想實際看 Dynamo 到底攔截到什麼，`torch._dynamo.explain` 也是個好東西，它會把擷取到的 bytecode dump 出來：

```python
import torch._dynamo as dynamo

explanation = dynamo.explain(f)(x)
print(explanation)
```

輸出很長，前面幾行是重點：

```
Graph Count: 1
Graph Break Count: 0
Op Count: 5
Break Reasons:
Ops per Graph:
  Ops 1:
    <built-in method sin of type object ...>
    <built-in method cos of type object ...>
    <built-in function mul>
    <built-in method tanh of type object ...>
    <built-in function add>
Out Guards:
  Guard 2:
    Name: "L['x']"
    Create Function: TENSOR_MATCH
  Guard 4:
    Name: "L['torch'].cos"
    Create Function: FUNCTION_MATCH
    Code List: ["___check_obj_id(L['torch'].cos, 22515628001936)"]
  ...
```

像 `f` 這種乾淨的純 Tensor 運算，一張圖、零個 Graph Break、五個 Operator 全部進圖，這是最理想的情況。後面的 `Out Guards` 就是這張圖成立的前提：`x` 要是 Tensor 且 shape、dtype 對得上（`TENSOR_MATCH`），`torch.sin`、`torch.cos`、`torch.tanh` 要還是同一個函式物件（`FUNCTION_MATCH`，用 id 比對），還有 grad mode、預設 device 之類的全域狀態。下次呼叫時這些檢查全過，才會直接用編好的 Kernel。接下來在 Dynamo 那幾篇講的，就是什麼樣的 operator 或是函數會導致 Graph Break、什麼會讓 Guard 檢查失敗而重編，以及到底該怎麼修。

## 三段 pipeline 的中間產物

光看執行時間還是有種沒有那麼赤裸地看到中間發生什麼事的感覺。那在這一小節呢！我們就準備把每一段都把它真正吐出的東西看過一遍，所以最後再用 `TORCH_LOGS` 把三段的產物一次印出來：

```python
torch._logging.set_logs(graph_code=True, aot_graphs=True, output_code=True)
torch.compile(f)(x)
```

也可以不改程式，直接用環境變數 `TORCH_LOGS="graph_code,aot_graphs,output_code"`。三個開關剛好對應三段 pipeline。

**Dynamo 吐出的 FX Graph**（`graph_code`）：

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4096, 4096][4096, 1]cuda:0"):
        l_x_ = L_x_
        # File: bench.py:15 in f, code: return torch.sin(x) * torch.cos(x) + torch.tanh(x)
        sin: "f32[4096, 4096][4096, 1]cuda:0" = torch.sin(l_x_)
        cos: "f32[4096, 4096][4096, 1]cuda:0" = torch.cos(l_x_)
        mul: "f32[4096, 4096][4096, 1]cuda:0" = sin * cos;  sin = cos = None
        tanh: "f32[4096, 4096][4096, 1]cuda:0" = torch.tanh(l_x_);  l_x_ = None
        add: "f32[4096, 4096][4096, 1]cuda:0" = mul + tanh;  mul = tanh = None
        return (add,)
```

這就是 Dynamo 從你的 Bytecode 錄下來的東西：一個 `nn.Module`，`forward` 裡是一行一行的 Tensor 運算，每個值都標了 dtype、shape、stride 和 device。注意這裡還是 `torch.sin`、`sin * cos` 這種使用者層級的寫法，還帶著原始碼行號，這是給人看的層級。

**AOTAutograd 吐出的圖**（`aot_graphs`）：

```python
 ===== Forward graph 2 =====
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4096, 4096][4096, 1]cuda:0"):
        sin: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.sin.default(arg0_1)
        cos: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.cos.default(arg0_1)
        mul: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sin, cos);  sin = cos = None
        tanh: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.tanh.default(arg0_1);  arg0_1 = None
        add: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, tanh);  mul = tanh = None
        return (add,)
```

長得很像，但每一個 op 都變成了 `torch.ops.aten.*` 這種 ATen 層級的正式名字，變數名也從 `L_x_` 變成 `arg0_1`，這是給編譯器看的層級。這裡只有 Forward graph，因為 `x` 沒有 `requires_grad`，AOTAutograd 判定是 inference，就不會 trace 出 backward 圖。等到 AOTAutograd 那幾篇拿一個真的訓練步驟來跑，你會看到多出一張 Backward graph，以及兩張圖之間的切分。

**Inductor 吐出的程式碼**（`output_code`）。這段最長，只節錄核心，也就是那一個融合出來的 Triton Kernel：

```python
@triton.jit
def triton_poi_fused_add_cos_mul_sin_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl_math.sin(tmp0)
    tmp2 = tl_math.cos(tmp0)
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.tanh(tmp0)
    tmp5 = tmp3 + tmp4
    tl.store(out_ptr0 + (x0), tmp5, None)

def call(args):
    arg0_1, = args
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
    triton_poi_fused_add_cos_mul_sin_tanh_0.run(arg0_1, buf0, 16777216, stream=stream0)
    return (buf0, )
```

Kernel 的名字 `triton_poi_fused_add_cos_mul_sin_tanh_0` 已經把事情說完了：`poi` 是 pointwise，後面五個 op 全被融合進同一個 Kernel。看它的內容，`tl.load` 一次、五個運算全在暫存器裡做完、`tl.store` 一次，前面算的「讀 64 MB 寫 64 MB」就是這兩行。`xnumel = 16777216` 是 4096 x 4096，`XBLOCK` 個 thread 一組把整個 Tensor 掃過去。下面的 `call` 是 wrapper：先檢查輸入的 shape 和 stride，配一塊輸出 buffer，啟動這一個 Kernel，回傳。整個 `f` 在 GPU 上就是這一次 launch。

從 Python 函式到這個 Kernel，就是這一行 `torch.compile` 走的路。這三份東西之後每一篇都會再回來讀，只是每次都會再讀得更深。

## 結語

今天透過從使用者的角度來畫我們 landscape，簡單的理解了這個神秘黑盒子的骨架。`torch.compile` 從你寫的 Python 到 GPU 上的 Kernel，中間是 Dynamo、AOTAutograd、Inductor、Runtime 四段 pipeline。另一個值得記得的是 PyTorch 編譯是延遲的，第一次帶真實輸入呼叫時才一口氣跑完，所以第一次慢是正常的，通常在生產環境都會需要先 warm up 才可以測得比較準確的數據。`backend` 參數對應 pipeline 的斷點，`eager`、`aot_eager`、`inductor` 是層層包住的關係，也因此也成為定位問題最直接的工具。

明天即將正式進入我們第一個重要的 component --> TorchDynamo！我們會來聊聊它到底是怎麼「在 Python 跑的當下」攔截你的程式的？這個答案基本上是藏在 CPython 的 Frame Evaluation Hook 裡。我們會用 `dis` 把 Bytecode 攤開，看 Dynamo 是在哪一層動的手腳，以及為什麼這個設計讓它能吃下幾乎任何 Python，卻又總在某些地方不得不斷開。那我們明天見！

## 參考資料

- [torch.compile 官方文件](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [torch.compiler 概觀](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchDynamo 深入介紹](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)
- [torch.compile 除錯與疑難排解](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- Ansel et al., [*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024
