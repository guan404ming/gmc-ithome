# Day 1 | torch.compile 是怎麼長出來的？

## 前言

大家好，我是 Wesley，這是我第一次參加鐵人賽。這三十天的主題是「一行 torch.compile 背後發生了什麼？30 天深度拆解 PyTorch 編譯器」。

會想寫這個系列，是因為 `torch.compile` 大概是 PyTorch 2.0 之後最常被提到、也最常被當成魔法的一個功能。加一行模型就變快，但很少有人說得清楚它到底做了什麼。變快的原因是什麼？為什麼有時候沒效？為什麼換個 batch size 又慢一次？這些問題其實都有很明確的答案，只是要打開黑盒子才看得到。所以這個系列會直接讀 PyTorch 的原始碼，把每一層的中間產物 dump 出來看，用簡單的圖幫助理解，帶大家把 Dynamo、AOTAutograd、Inductor 三層一段一段拆開。

這篇文章會出現一些你可能還不熟的名詞，例如 FX Graph、Guard、Graph Break、Triton、Frame Evaluation Hook 等等。不用擔心，這些都不需要現在就懂，接下來的系列會一個一個講清楚。

第一天的目標不是理解細節，而是先建立一個整體輪廓：PyTorch 為什麼一開始選了 Eager Mode、為了拿到計算圖它試過哪些方法、為什麼那些方法沒有成功、以及最後 `torch.compile` 是被誰、用什麼想法做出來的。知道這段歷史，後面每一個設計決定就都有脈絡可循了。

正文開始！

## PyTorch 的起點：Eager Mode 是設計，不是妥協

PyTorch 的前身是 [Torch7](https://github.com/torch/torch7)，一個用 Lua 寫的科學計算框架。2016 年，Facebook AI Research（FAIR）的 Adam Paszke、Sam Gross、Soumith Chintala、Gregory Chanan 等人把它移植到 Python，2017 年 1 月公開釋出。

當時的主流是 TensorFlow 1.x 和 Theano。它們的模式是 Define-and-Run：你先用 Python「宣告」一張靜態的計算圖，再把資料丟進 Session 執行。這種做法對編譯器很友善，因為整張圖一開始就在手上，可以隨便最佳化。但對寫程式的人非常不友善：不能在中間 `print`、`if` 要寫成 `tf.cond`、Debugger 也進不去。

PyTorch 反其道而行，選了 Define-by-Run，也就是我們現在說的 Eager Mode。每一行 Python 執行的當下就真的算，計算圖是在執行過程中動態被記錄下來給 Autograd 用的，用完就丟。2019 年 NeurIPS 的論文 [*PyTorch: An Imperative Style, High-Performance Deep Learning Library*](https://arxiv.org/abs/1912.01703) 把這個設計哲學講得很白：把易用性和「Python 就是第一公民」放在第一位，效能靠底層的 C++ 與 CUDA Kernel 撐，而不是靠犧牲彈性換來。

這個選擇讓 PyTorch 在研究圈贏了。但它也埋下了一個結構性的問題：Eager Mode 一次只看到一個 Operator，永遠不知道下一步是什麼，所以做不了任何跨 Operator 的最佳化，例如把好幾個小運算融合成一個 Kernel；也很難把模型帶離 Python，部署到別的環境。整個系列後面講的一切，其實都是在解這一個問題：**在不放棄 Eager Mode 的前提下，怎麼把計算圖拿回來？**

## 第一次嘗試：TorchScript

PyTorch 1.0 在 2018 年帶來了 [TorchScript](https://pytorch.org/docs/stable/jit.html)，提供兩條路。

第一條是 `torch.jit.trace`，拿一組範例輸入實際跑一次，把碰到的 Tensor 運算錄下來。這種做法很快，但它看不到控制流程，`if` 只會錄到當時走過的那一條分支，換一組輸入結果可能就錯了。第二條是 `torch.jit.script`，直接解析 Python 原始碼，翻譯成 TorchScript 自己的 IR。這樣看得到控制流程，但它只支援 Python 的一個子集，你得把模型改寫成它看得懂的樣子。

TorchScript 的目標其實偏部署，重點是把模型序列化、離開 Python 執行，加速並不是它的強項。而且它對使用者的要求太高：一碰到 dict、任意 Python 物件、第三方函式庫，就是一連串的改寫。很多團隊試過、放棄，它現在也已經進入維護模式。

## 中間的探索：torch.fx 與 Lazy Tensor

2020 到 2021 年間，PyTorch 團隊在兩個方向上摸索。

一個是 [torch.fx](https://pytorch.org/docs/stable/fx.html)，由 James Reed 等人主導，論文 [*Torch.fx: Practical Program Capture and Transformation for Deep Learning in Python*](https://arxiv.org/abs/2112.08429) 發表在 MLSys 2022。它用 Python 層的 Symbolic Tracing 把 `nn.Module` 抓成一張很簡單的圖，並提供一套好寫的 Graph Transformation API。FX 後來成了 Dynamo 吐出的圖的格式，這點非常重要，但 FX 自己的 Tracer 一樣吃不下依賴資料的控制流程。

另一個是 Lazy Tensor，[PyTorch/XLA](https://github.com/pytorch/xla) 走的路。Tensor 運算先不真的算，累積成圖，等到有人要看結果才一次送給後端。它對使用者透明，但每一步都要重新 Trace 一次，Overhead 很高，而且一碰到需要看數值的地方就得 Flush。

這些嘗試合起來證明了同一件事：要求使用者改寫程式的方案不會贏，而能吃下任意 Python 的方案又抓不到完整的圖。要走出這個兩難，需要一個完全不同的切入點。

## 轉折：TorchDynamo 與 PyTorch 2.0

2021 年 9 月，Jason Ansel 在 PyTorch dev-discuss 上[發表了 TorchDynamo 的雛形](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)。它的核心想法很不一樣：不去解析 Python 原始碼，也不靠 Tracing 錄 Operator，而是利用 CPython 的 Frame Evaluation Hook（[PEP 523](https://peps.python.org/pep-0523/)），在 Python Bytecode 執行的當下把它攔下來，把能編的 Tensor 運算抓成 FX Graph，看不懂的地方就在那裡斷開（這就是 Graph Break），退回一般 Python 執行，之後再接回來。這樣使用者一行程式都不用改，而且永遠不會「不能跑」，最壞的情況只是沒加速而已。

Dynamo 解決的是「怎麼拿到圖」，但光有圖還不夠。同一時期還有幾塊拼圖陸續到位。[AOTAutograd](https://github.com/pytorch/pytorch/tree/main/torch/_functorch) 由 Horace He 等人主導，它拿到 forward 圖之後把 backpropagation 也一起 Trace 出來，讓訓練也能被整張圖編譯，並且把 In-place 修改、View 這些麻煩的東西正規化成純函數式。[TorchInductor](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) 由 Jason Ansel 主導，是預設的後端，把圖 Lower 成 Loop-level IR、做融合，然後生成 Triton（GPU）或 C++（CPU）程式碼；[Triton](https://github.com/triton-lang/triton) 是 OpenAI 的 Philippe Tillet 做的、用 Python 寫 GPU Kernel 的語言，讓「用 Python 生 GPU Kernel」這件事變得可行。還有 [PrimTorch](https://github.com/pytorch/pytorch/tree/main/torch/_prims)，把 PyTorch 兩千多個 Operator 拆解到幾百個基本 Operator，讓後端不用一一實作。

2022 年 12 月的 PyTorch Conference 上，這一整套以 PyTorch 2.0 的名義發表，2023 年 3 月[正式釋出](https://pytorch.org/blog/pytorch-2.0-release/)，對外的介面就是那一行 `torch.compile`。2024 年 ASPLOS 的論文 [*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*](https://pytorch.org/assets/pytorch2-2.pdf) 是這整套設計的正式論述，也是這個系列會反覆回頭看的一份文件。順帶一提，PyTorch 在 2022 年 9 月從 Meta 移交給 Linux Foundation 底下新成立的 [PyTorch Foundation](https://pytorch.org/blog/PyTorchfoundation/)，但核心開發至今仍以 Meta 的團隊為主。

## 把時間軸攤開

| 年份 | 事件 | 對「圖」的態度 |
|---|---|---|
| 2017 | PyTorch 0.1 釋出 | 純 Eager，沒有圖 |
| 2018 | PyTorch 1.0，TorchScript | 要使用者改寫成子集，才拿得到圖 |
| 2019 | NeurIPS 論文 | 明文把易用性放第一 |
| 2020 到 2021 | torch.fx、Lazy Tensor | 圖的格式與透明擷取的實驗 |
| 2021 年 9 月 | TorchDynamo 雛形 | 在 Bytecode 層攔截，抓得到就抓，抓不到就斷 |
| 2022 底到 2023 | PyTorch 2.0，`torch.compile` | 一行接上 Dynamo、AOTAutograd、Inductor |
| 2024 | ASPLOS 論文 | 整套設計的正式論述 |

如果把這條線縮成一句話：PyTorch 從頭到尾沒有放棄 Eager Mode，`torch.compile` 是在保留 Eager 語意的前提下，把圖偷偷抓出來的第三次嘗試，而前兩次的教訓決定了它的每一個設計。

## 這 30 天會怎麼走

接下來的內容大致分成四個部分。第一部分是 Dynamo，講它怎麼在 Bytecode 層攔截 Python、Guard 是什麼、為什麼會 Graph Break、吐出的 FX Graph 長什麼樣。第二部分是 AOTAutograd，講 backpropagation 怎麼被一起 Trace、In-place 怎麼被正規化、Operator 怎麼被拆成基本運算。第三部分是 Inductor，講圖怎麼變成 Loop、誰跟誰融合、以及怎麼讀它生出來的 Triton 和 C++。最後一部分是整合與實戰，包括 CUDA Graph、快取、Recompilation 爆炸，以及自己寫一個 Backend。

明天會從使用者的視角出發，把 `torch.compile` 的四段 pipeline攤開：Dynamo、AOTAutograd、Inductor、Runtime 各做什麼，並用 `backend` 參數把它們一段一段切開來親手驗證。那我們明天見！

## 參考資料

- Paszke et al., [*PyTorch: An Imperative Style, High-Performance Deep Learning Library*](https://arxiv.org/abs/1912.01703), NeurIPS 2019
- Reed et al., [*Torch.fx: Practical Program Capture and Transformation for Deep Learning in Python*](https://arxiv.org/abs/2112.08429), MLSys 2022
- Ansel et al., [*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*](https://pytorch.org/assets/pytorch2-2.pdf), ASPLOS 2024
- PEP 523, [*Adding a frame evaluation API to CPython*](https://peps.python.org/pep-0523/)
- [TorchDynamo 首次公開討論（PyTorch dev-discuss）](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)
- [PyTorch 2.0 正式釋出公告](https://pytorch.org/blog/pytorch-2.0-release/)
- [PyTorch Foundation 成立公告](https://pytorch.org/blog/PyTorchfoundation/)
