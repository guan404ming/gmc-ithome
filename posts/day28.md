# Day 28 | 第三站掛上自己的招牌：動手寫一個 backend

## 前言

前幾天把診斷 torch.compile 的工具一件一件收進了工具箱，今天可以做一件更過癮的事，自己下場把 pipeline 的第三站接管過來。畫 pipeline 地圖時說過，前兩站交出來的是一張標準的 FX Graph，第三站可以換掉，任何吃 FX Graph 的東西都能接在這裡當後端，Inductor 只是 PyTorch 自帶、也最成熟的那一個。這句話今天要兌現。我們會由淺入深寫三個 backend，一個只旁觀，一個動手改圖，一個往下接到 ATen 層，把「後端」從一個參數變成自己寫的程式。對照的原始碼在 [`torch/_dynamo/backends/registry.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/backends/registry.py)。

正文開始！

## 契約只有一句話

寫 backend 之前先把契約講清楚。一個 backend 就是一個 Python callable，收 Dynamo 抓好的 GraphModule 和一串 example inputs，回傳一個 callable。改寫後的 bytecode 執行到圖的位置時，呼叫的就是你回傳的那個東西。契約就這麼一句話，沒有要繼承的類別，也沒有要實作的介面。

這份契約把時間切成兩段。backend 本體在編譯期執行，一張圖一生只來一次，回傳的 callable 才是執行期的常客。所以再貴的分析、改寫、程式碼生成都可以放心塞進 backend 本體，那是一次性的成本，執行期的快慢完全取決於你交出去的那個 callable。義務也只有一條，回傳的 callable 吃的參數和吐的結果要跟圖的輸入輸出對得上，語意等價是你自己的責任，Dynamo 不會幫你驗算。

這代表最小的合法 backend 短得驚人，直接把 GraphModule 自己的 forward 回傳出去就行，圖照原樣用 eager 執行。用來剝 pipeline 的 `backend="eager"` 差不多就是這個長度，它存在的意義本來就只是「跑到我這裡為止」。而 Inductor 從 lowering 到 codegen 忙了整整一個 Part 3，最後交出來的也不過是一個名叫 call 的 callable。兩者在契約面前完全平等，這就是介面設計得薄的好處。

## 第一個後端只旁觀

第一個 backend 只多做一件事，把收到的東西印出來再原樣放行。以下實驗都在本機 CPU 上跑（torch 2.8.0），完整程式在 `code/day28/`。

```python
def observer(gm, example_inputs):
    calls["observer"] += 1
    print(f"[observer] call #{calls['observer']}")
    print(gm.graph)
    for i, t in enumerate(example_inputs):
        if isinstance(t, torch.Tensor):
            print(f"  input[{i}]: shape={tuple(t.shape)} dtype={t.dtype}")
        else:
            print(f"  input[{i}]: {type(t).__name__} = {t}")
    return gm.forward
```

拿一個 relu 加一的小函式，用 `torch.compile(f, backend=observer)` 編譯後呼叫一次。

```
[observer] call #1
graph():
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %y : [num_users=1] = call_function[target=torch.relu](args = (%l_x_,), kwargs = {})
    %add : [num_users=1] = call_function[target=operator.add](args = (%y, 1), kwargs = {})
    return (add,)
  input[0]: shape=(4, 8) dtype=torch.float32
matches eager: True
```

這就是 backend 收到的原料。圖還停在 Dynamo 這一層，node 的目標是 `torch.relu` 和 `operator.add` 這種 Python 層函式，跟使用者寫的程式一一對得上。example inputs 給的是帶著 shape 和 dtype 的真實張量，想在編譯期先試跑量測或驗證，直接拿這串輸入餵圖就行，Inductor 的 autotune 拿來 benchmark 候選 kernel 的也是同一份材料。回傳 `gm.forward` 之後圖照跑，結果和 eager 一致。

## 上游的機制照常運轉

第二次用同樣 shape 呼叫，log 一片安靜，observer 的計數停在 1。Guard 檢查、編譯成品掛在 code object 上的快取，這些都發生在 backend 被呼叫之前，是 Dynamo 的地盤。你的 backend 再簡陋，也自動繼承整套上游機制，驗過放行，不會重編。

換一個 shape 再呼叫，observer 第二次被叫醒，收到的圖也變了樣。

```
[observer] call #2
graph():
    %s77 : torch.SymInt [num_users=0] = placeholder[target=s77]
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    ...
  input[0]: SymInt = s77
  input[1]: shape=(6, 8) dtype=torch.float32
```

shape 一變 Guard 失敗觸發重編，而重編出來的第二張圖多了一個 SymInt 的 placeholder，example inputs 裡也混進了一個不是張量的符號，這是 automatic dynamic 在自動把 shape 放寬。寫 backend 的人得知道輸入不保證都是張量，這一格符號就是提醒。

graph break 也照常上班。函式要是中途斷開，Dynamo 會切出好幾張子圖，每張子圖各自把 backend 叫起來編一次，斷點之間的程式碼照舊回直譯器跑。自訂 backend 接手的從來不是「整個函式」，而是 Dynamo 切好的每一段安全區，前面看過的所有上游行為在這裡全部原樣成立。

## 第二個後端動手改圖

旁觀證明了我們拿得到圖，下一步證明圖可以改。FX Graph 是普通的 Python 資料結構，走訪 node、換掉目標、重新生成程式碼，三步完成一次改寫。

```python
def relu_to_sigmoid(gm, example_inputs):
    n = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.relu:
            node.target = torch.sigmoid
            n += 1
    gm.recompile()
    print(f"[rewriter] replaced {n} node(s): relu -> sigmoid")
    return gm.forward
```

拿 `[-2, 0, 2]` 對答案。

```
eager    f(t): tensor([1., 1., 3.])
[rewriter] replaced 1 node(s): relu -> sigmoid
compiled f(t): tensor([1.1192, 1.5000, 1.8808])
sigmoid(t)+1 : tensor([1.1192, 1.5000, 1.8808])
```

eager 版是 relu 的結果，編譯版和 sigmoid 加一逐位一致，這顆 op 真的被換掉了。使用者的 Python 程式碼一個字都沒動，動的是它被抓下來的那張圖，這正是圖表示法的價值，程式一旦變成資料，改寫它就只是普通的資料處理。故意把語意改壞只是為了讓證據夠明顯，實務上同一套手法做的事都溫和得多，插一個計數 hook 統計每種 op 出現幾次、在特定 node 前後夾一段 profiling、把某個 op 換成數值上更穩的等價寫法。Inductor 的各種圖層最佳化，起點也不過就是這三步。

![pipeline 第三站的廠房被換成自己的工作坊，FX Graph 流進來被逐節點檢視改寫](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day28/custom_backend.gif)

*圖一：pipeline 第三站的交接。Inductor 的廠房被拔下，插上自己的小工作坊，一張 FX Graph 流進來，逐 node 檢視，把 relu 換成 sigmoid，交回一個 callable，水流繼續往下走，上游的 Guard 與快取渾然不覺。*

## 第三個後端往下接到 ATen 層

Dynamo 層的圖貼近使用者，但對編譯器來說太高階，in-place 和 view 還在，backward 也還沒展開，這些正是 AOTAutograd 負責的髒活。好消息是這段不用自己重寫，用 `aot_autograd` 把自己的編譯函式包起來，就能站到跟 Inductor 一樣的位置。

```python
def fw(gm, example_inputs):
    print("[wrapper] fw_compiler got:")
    print(gm.graph)
    return gm.forward

aten_backend = aot_autograd(fw_compiler=fw)
```

同一個函式再編一次，`fw` 收到的圖換了一副面孔。

```
[wrapper] fw_compiler got:
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%arg0_1,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, 1), kwargs = {})
    return (add,)
```

node 的目標從 `torch.relu` 變成 `torch.ops.aten.relu.default`，這是一張走完展開、functionalization 和 decomposition 的 ATen 圖，純函數式、operator 集合小而穩定，正是編譯器該吃的形狀。需要訓練的話再多傳一個 backward 用的編譯函式，AOTAutograd 會把 joint graph 切成前後兩半，各自送進對應的編譯函式。aot_eager 就是這個包法的現成品，兩個編譯函式都原樣返回，而 Inductor 的正職也是當 `aot_autograd` 手下的編譯函式，我們此刻站的就是它平常站的月台。兩層各有客群，做貼著使用者程式碼的分析工具就留在 Dynamo 層，接真正的 code generator 就包一層下到 ATen 層。

## 把名字掛進註冊表

最後一步是掛牌。傳函式物件只有自己能用，用 `register_backend` 給它一個名字，任何人都能用字串指名。

```
list_backends(): ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
registered: True
observer calls via registry: 1
```

`list_backends()` 列出的就是這張註冊表，名單裡的 onnxrt、tvm、openxla 全是靠同一套機制接進來的第三方後端，只是它們透過 Python 套件的 entry point 註冊，套件裝好名字就自動出現，使用者不需要 import 那個套件。我們的 observer 掛上名字後，用 `backend="day28_observer"` 一樣叫得動，計數器如實跳了一格。所謂生態系插槽，拆開來就是一個 dict 加一點套件掃描，TensorRT 或自家硬體的編譯器要接進來，走的也是這扇門。

## 什麼時候值得自己寫

冷靜說，多數人永遠不需要寫 backend，Inductor 已經很好，加速交給它就是了。值得動手的場景有三種。第一種是接自家的編譯器或硬體，這個介面就是接進 torch.compile 生態的正門，名單上那幾位就是這麼進來的。第二種是做分析工具，反正 Dynamo 已經把圖抓好了，順手拿來統計 op 分布、估 FLOPs、比對兩個版本的圖，成本低得驚人。第三種就是今天做的事，教學與除錯，一個十行的 observer 勝過十頁文件，想知道 pipeline 某一站交出了什麼，寫個 backend 站在那裡看就是了。

## 結語

回頭看，今天沒有學新機制，而是把整個系列的知識換一個站位再用一次。契約是收 GraphModule 和 example inputs、回一個 callable。上游的 Guard、快取、dynamic shape 對自訂 backend 一視同仁。三個 backend 逐步升級，旁觀證明拿得到圖，改圖證明圖是活的，包上 `aot_autograd` 證明我們可以站上跟 Inductor 同一格月台。一開始畫地圖時說第三站可以換，今天真的換給你看。

不過 torch.compile 再怎麼換後端都還是一個 JIT，編譯發生在部署機的第一次呼叫，Python 直譯器也一直在場。想把編譯徹底搬到上線之前，甚至讓成品脫離 Python、在 C++ 環境裡直接執行，就需要另一條路。明天來看 torch.export 和 AOTInductor 這對搭檔，把整條 pipeline 從即時編譯改成事先出貨。那我們明天見！

## 參考資料

- [torch/_dynamo/backends/registry.py：register_backend 與 list_backends（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/backends/registry.py)
- [torch/_dynamo/backends/common.py：aot_autograd（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/backends/common.py)
- [torch/_dynamo/backends/debugging.py：eager 等 debug 後端（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/backends/debugging.py)
- [PyTorch Docs: Custom Backends](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)
