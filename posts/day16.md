# Day 16 | 沒有數值要怎麼 trace？編譯管線的空殼替身 FakeTensor

## 前言

昨天 min-cut partitioner 把 joint graph 一刀切成 forward 和 backward，AOTAutograd 這一站算是圓滿落幕，照理說今天該進 Inductor 了。不過在換站之前，想先把一位跑了整個系列龍套的配角請到台前。Day 12 說 AOTAutograd「拿 FakeTensor 重跑 forward」、Day 13 的 metadata 收集「用 FakeTensor 跑一次」、Day 11 的 Symbolic Shapes 也是在它身上長出來的。它每次都被一句話帶過，但其實 Dynamo 追蹤的每一步、AOTAutograd 展開的每一張圖，背後全是它在跑。

它就是 `FakeTensor`，一顆被抽乾數值、只剩 shape、dtype、stride 和 device 的空殼 Tensor。今天來把它講清楚，為什麼編譯期不能用真值算、meta device 是什麼、FakeTensor 又是怎麼在 meta 之上補了一層謊，以及這個空殼的極限在哪。另外先說一聲，前幾天的實驗都跑在 Modal 的 GPU 上，今天的主角恰好證明了沒有卡也能編譯，所以實驗全部在本機 CPU 上跑，`torch 2.8.0`。

正文開始！

## 為什麼編譯期不能真的算

回想一下 Dynamo 追蹤時在做的事。它逐條翻譯 bytecode，碰到 Tensor 運算就往圖裡加 node，但下一條指令可能馬上要問「這個結果的 shape 是多少」，例如 `y.view(-1)` 要知道元素個數、`x @ w` 要檢查兩邊維度對不對得上。要回答這些問題，最直接的辦法就是真的把運算執行一遍。

但真的執行有三個問題。第一是貴，編譯期每個 op 都真算一次，等於整個 forward 多跑一遍，模型一大這筆帳受不了。第二是根本不一定算得了，`torch.compile` 常見的用法是在沒有 GPU 的機器上先編譯、匯出，或是模型大到一顆 device 放不下，數值運算無從發生。第三是危險，Day 7 講過真的執行會把 side effect 提前洩漏出去。

其實用真值跑一遍來抓圖是有前例的，第一代的 `torch.jit.trace` 走的就是這條路，把範例輸入真的餵進函式執行，錄下沿途發生的每個 op。上面三個問題它全中，追蹤一次就是完整跑一次 forward，而且執行過程中的 print、檔案寫入這些 side effect 都會真的發生。`torch.compile` 這一代顯然不想重蹈覆轍。

關鍵的觀察是，編譯器其實從頭到尾都不關心數值。它要的只有 metadata，也就是每個中間結果的 shape、dtype、stride、device。矩陣乘完是多大、記憶體要怎麼排、kernel 要生成什麼樣子，全部由這幾樣決定，至於格子裡裝的是 3.14 還是 -0.5，對編譯決策毫無影響。所以理想的做法是找一種「只算 metadata、不算數值」的執行方式，讓整個 forward 用趨近於零的成本走一遍。PyTorch 把這件事拆成了兩層，meta device 和 FakeTensor。

## Meta device 是什麼

PyTorch 的 device 除了 `cpu` 和 `cuda`，還有一個特殊的 `meta`。放在 meta device 上的 Tensor 不配置任何儲存空間，只保留 metadata。直接建一顆來看。

```python
m = torch.empty(4, 8, device="meta")
print(m)
print("shape:", m.shape, "| dtype:", m.dtype, "| stride:", m.stride(), "| device:", m.device)
y = m @ torch.empty(8, 16, device="meta")
```

    tensor(..., device='meta', size=(4, 8))
    shape: torch.Size([4, 8]) | dtype: torch.float32 | stride: (8, 1) | device: meta
    matmul -> torch.Size([4, 16]) meta

印出來的內容是 `...`，因為真的沒有東西可以印。但 `(4, 8)` 乘 `(8, 16)` 的 matmul 照常執行，輸出一顆 `(4, 16)` 的 meta tensor，shape 推導完全正確。想讀數值則會直接吃 exception。

    item() -> RuntimeError - Tensor.item() cannot be called on meta tensors

這能運作是因為每個 op 除了 CPU kernel、CUDA kernel，還註冊了一個 meta kernel，只負責根據輸入的 metadata 算出輸出的 metadata，一個位元組的資料都不碰。這些 meta kernel 大多集中在 [`torch/_meta_registrations.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_meta_registrations.py)，跟著 dispatcher 的機制走，`device="meta"` 的輸入自然分派過去。對編譯器來說這就是完美的「乾跑」，成本和 shape 大小無關，因為根本沒有資料在動。

順帶一提，meta device 並不是編譯器的專屬玩具。想在筆電上把一個 70B 模型的骨架建出來、算算每層參數要吃多少記憶體，`with torch.device("meta")` 包住模型建構就辦得到，一個參數都不會真的配置。這也是各家推理框架載入大模型時先建殼、再逐層灌權重的基礎。

## 光有 meta 還不夠，FakeTensor 補上 device 的謊

那 Dynamo 直接把使用者的 Tensor 轉成 meta 來追蹤不就好了？差一步。轉成 meta 之後，「這顆 Tensor 原本在哪個 device」這個資訊就丟了。而編譯器極度在乎 device，Inductor 要據此決定生 Triton kernel 還是 C++ kernel，`if x.is_cuda` 這種程式碼在追蹤時也要能得到正確答案。

所以 [`torch/_subclasses/fake_tensor.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_subclasses/fake_tensor.py) 定義了 `FakeTensor`，一個 `torch.Tensor` 的 subclass。它的本體是一顆 meta tensor，身上多帶一個 `fake_device` 欄位，記著它假裝自己所在的 device。任何人問 `.device`，它回答的是 `fake_device` 而不是 `meta`。原始碼的 docstring 把這個結構講得很直白，「FakeTensor extends MetaTensors to also carry an additional `fake_device`」。也就是說，資料層面它活在 meta device 上，對外卻聲稱自己是一顆 `cpu` 或 `cuda:0` 的 Tensor，整條編譯管線就被這個謊安穩地騙了過去。

搭配它的是 `FakeTensorMode`。啟用之後，所有 Tensor op 在 dispatch 層被攔截，交給它處理。它的處理流程概念上是三步，先把參與運算的 FakeTensor 們攤開成裡面的 meta tensor，把 op 丟給 meta kernel 推出輸出的 metadata，再把結果包回 FakeTensor、貼上正確的 `fake_device`。實際跑一段。

```python
from torch._subclasses.fake_tensor import FakeTensorMode

mode = FakeTensorMode()
with mode:
    a = torch.randn(32, 64)
    b = torch.randn(64, 128)
    c = torch.relu(a @ b)
```

    type: FakeTensor | shape: (32, 128) | dtype: torch.float32 | device: cpu

在 mode 裡面連 `torch.randn` 都被攔掉了，沒有任何亂數被生成，但 `a @ b` 過 relu 之後的 shape、dtype、device 全部推對，而且 `device` 顯示 `cpu` 而不是 `meta`，謊撒得很完整。已經存在的真 Tensor 則用 `mode.from_tensor()` 轉換，抽掉數值、留下 metadata。這種乾跑有多便宜，拿一個誇張的 shape 就看得出來。

    fake 65536x65536 matmul (16 GB per tensor): 0.48 ms -> (65536, 65536)

單顆 16 GB 的矩陣，真算一次 matmul 要幾百 TFLOPs，這台筆電的記憶體連放都放不下，fake 世界裡 0.48 毫秒走完。這就是「整個 forward 乾跑一遍」敢成立的底氣。

這套機制也解釋了一個常見的報錯。自訂 C++ 或 Triton op 接上 `torch.compile` 時，如果只提供了真正的 kernel，編譯期一到就會抱怨缺少 fake implementation，因為 `FakeTensorMode` 攔下這個 op 之後找不到能推 metadata 的實作。`torch.library.register_fake` 要你補的就是這個 op 的 meta kernel，告訴編譯器輸入長這樣時輸出該長什麼樣。

![一顆真 tensor 被抽乾數值只剩 shape 外殼，op 流過外殼推出新 shape，直到 item() 亮紅卡住](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day16/fake_tensor.gif)

*圖一：FakeTensor 的一生。左邊一顆裝滿數值的真 Tensor 被 `from_tensor()` 抽乾，數值流走、只剩下記著 shape、dtype、device 的空殼。中段兩個空殼流進 `matmul`，dispatch 攔下 op 丟給 meta kernel，只憑 metadata 就推出 `(8, 4)` 的新殼，數值欄位永遠是空的。結尾 `.item()` 來要一個具體數值，殼裡拿不出東西，整格亮紅、丟出 DataDependentOutputException。*

## 它在管線裡的位置

有了這個替身，回頭看整條管線就會發現它無所不在。先看 Dynamo。Day 5 說 Tensor 被包成 `TensorVariable`，其實每個 `TensorVariable` 的 FX node 上都掛著一個 `example_value`，記錄「這個 node 的輸出長什麼樣」，而它正是一顆 FakeTensor。寫一個什麼都不編譯的 backend，把收到的圖上每個 node 的 `example_value` 印出來就能驗證。

```python
def peek(gm, example_inputs):
    for n in gm.graph.nodes:
        ev = n.meta.get("example_value")
        if isinstance(ev, torch.Tensor):
            print(f"  {n.op:13s} {n.name:6s} example_value = {type(ev).__name__}{tuple(ev.shape)}")
    return gm.forward

def f(x, w):
    return torch.tanh(x @ w)

torch.compile(f, backend=peek)(torch.randn(8, 16), torch.randn(16, 4))
```

    placeholder   l_x_   example_value = FakeTensor(8, 16)
    placeholder   l_w_   example_value = FakeTensor(16, 4)
    call_function matmul example_value = FakeTensor(8, 4)
    call_function tanh   example_value = FakeTensor(8, 4)

從輸入到中間結果，全是 FakeTensor。流程上，輸入的 Tensor 在被包成 `TensorVariable` 的那一刻（就是 Day 5 講的包裝流程裡）就被轉成 fake，之後圖上每長一個 node，[`torch/_dynamo/utils.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/utils.py) 的 `get_fake_value` 就拿著輸入 node 們的 fake 值，在 `FakeTensorMode` 下把這個 op 乾跑一次，得到的輸出掛回 node 當 `example_value`。追蹤到 `y.view(-1)` 時 Dynamo 能回答 shape 問題、Day 12 圖上那些 `"f32[8, 4]"` 標註，靠的都是這一套。用 `TORCH_LOGS="+dynamo"` 也能看到 Dynamo 建圖輸入時的自白（節錄）。

    create_graph_input L_x_ L['x'] FakeTensor(..., size=(8, 16)) at debug_level 0 before=False
    create_graph_input L_w_ L['w'] FakeTensor(..., size=(16, 4)) at debug_level 0 before=False

到了 AOTAutograd，同一批 FakeTensor 直接接手。Day 12 說它「拿 FakeTensor 把 forward 重新執行一遍，讓 autograd 引擎在上面展開 backward」，現在可以讀懂這句話的全部了。joint graph 的 trace 是一次真的 Python 執行，只是每顆 Tensor 都是 fake 的，所以幾千個 op 的模型也能在編譯期便宜地「跑」完，Functionalization 的 metadata 收集、partitioner 算保存成本用的元素個數，吃的全是同一套 fake metadata。一路到 Inductor 拿到的圖，每個 node 身上的 shape 標註也還是這批 FakeTensor 留下的。

順帶一提 Day 11 的 Symbolic Shapes，它和 FakeTensor 是同一枚硬幣的兩面。FakeTensor 的 shape 欄位不一定是具體的 int，`FakeTensorMode` 身上掛著一套管理符號 shape 的機制，某個維度被判定為 dynamic 時，填進 shape 欄位的就是 `s0` 這種 SymInt。之後 meta kernel 推 shape 時是拿符號在做算術，輸出的 shape 是 `(s0, 4)`，Day 11 看到的那些符號運算，發生的舞台正是 FakeTensor 的 metadata 欄位。

## 假數值算不出來的事

空殼終究有極限。只要程式真的需要一個具體數值，fake 世界就答不上來。最典型的就是 `.item()`，在 `FakeTensorMode` 下對一顆 fake tensor 呼叫它。

    fake item() -> DataDependentOutputException
       aten._local_scalar_dense.default

`FakeTensorMode` 丟出 `DataDependentOutputException`，意思是這個 op 的輸出取決於資料本身，而資料不存在。同一堵牆在 `torch.compile` 裡的樣子，就是 data-dependent 的控制流。

```python
def g(x):
    if x.sum() > 0:
        return x + 1
    return x - 1

torch.compile(g, fullgraph=True)(torch.randn(4))
```

    compile data-dependent branch -> Unsupported
    Data-dependent branching
      Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

`x.sum() > 0` 的真假只有算了才知道，但追蹤期手上只有空殼，Dynamo 無從決定該走哪個分支，`fullgraph=True` 之下直接舉手投降，預設模式則是 Graph Break 退回 eager。這不是實作偷懶，是這套設計的本質邊界，用「不算數值」換到的所有便宜，在真正需要數值的那一刻都要還。PyTorch 的緩解方案也都是繞著這條線走，例如 `torch.cond` 把兩個分支都抓進圖裡、unbacked SymInt 給 `.item()` 的結果一個符號讓它繼續往下流，這些之後聊到 Dynamic Shapes 進階題再展開。

## 結語

把今天濃縮成一句話，編譯期需要跑但不能真的算，於是每顆 Tensor 都換成一個只剩 metadata 的替身。meta device 提供「只算 shape 不碰資料」的 kernel，FakeTensor 在上面補一個 `fake_device` 把 device 語意保住，`FakeTensorMode` 在 dispatch 層攔下每個 op 完成乾跑。Dynamo 的 `example_value`、AOTAutograd 的 joint trace、partitioner 的成本計算，整條管線共用這一套替身，而 Symbolic Shapes 就住在替身的 shape 欄位裡。代價是碰到 data-dependent 的地方，空殼就再也演不下去。

配角的債清完了，明天正式進入第三站 TorchInductor。它從 AOTAutograd 手上接過乾淨的 ATen 圖，要走過 lowering、fusion、codegen 三道工序，最後吐出真正跑在硬體上的 kernel。明天先看總覽，把這座工廠的每個車間走一遍。那我們明天見！

## 參考資料

- [torch/_subclasses/fake_tensor.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_subclasses/fake_tensor.py)
- [torch/_meta_registrations.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_meta_registrations.py)
- [torch/_dynamo/utils.py：get_fake_value（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/utils.py)
- [Fake tensor（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html)
- [Meta device（PyTorch 官方文件）](https://pytorch.org/docs/stable/meta.html)
- [Dynamo Deep-Dive（PyTorch 官方文件）](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
