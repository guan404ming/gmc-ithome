# Day 29 | 不帶 Python 也能跑嗎？torch.export 與 AOTInductor

## 前言

昨天自己接管了一次 backend，整條 torch.compile 流水線算是摸過一遍。不過有一個前提我們從來沒有挑戰過，所有事情都發生在 Python process 裡。Dynamo 掛在直譯器上攔 frame，guard 每次呼叫都驗一輪，編好的 kernel 由 Python wrapper 負責調度。但部署現場常常不長這樣，線上服務可能是一台 C++ server，手機和車機上根本沒有直譯器，就算有，你也未必想把整套 JIT 機器搬進生產環境。今天來看 PyTorch 為這種場景鋪的另一條路，torch.export 加上 AOTInductor，把模型先編好、打包成一個 .pt2 檔，到了現場不需要 Python 也能跑。

今天的比喻是廚師與便當。torch.compile 像帶著廚師隨行，到現場看食材臨場開火，菜色隨時能調整，但整套廚房都得跟著走。export 加 AOTInductor 則是出發前把菜封進便當盒，現場打開就能吃，代價是菜單得先定案，出了門就不能改。

正文開始！

## 同一條流水線，兩種出貨方式

先把兩條路的定位擺清楚。torch.compile 是 JIT，編譯發生在模型第一次被呼叫時，之後每次呼叫都要經過 guard 檢查，遇到新的 shape 或分支就當場再編一份。這套設計的前提是 Dynamo 永遠在場，看不懂的程式碼就 graph break，斷掉的部分交還給直譯器用 eager 跑。彈性極高，但也意味著 Python、Dynamo、Inductor 整組人馬都是 runtime 的一部分。

export 加 AOTInductor 是 AOT，把同一條流水線搬到部署之前執行。torch.export 把模型收成一張完整的圖，AOTInductor 把這張圖編成機器碼並打包，產物是一個自足的 .pt2 檔。到了執行現場，沒有 trace、沒有 guard、沒有重編，只剩下載入和呼叫。這不是另一套編譯器，export 底下抓圖的是同一個 Dynamo，AOTInductor 就是 Inductor，講過的 lowering、fusion、codegen 一路照走，換掉的只是入口和出口。

兩條路付錢的時間點不一樣。JIT 把編譯成本攤在執行期，第一次呼叫慢，之後每次呼叫還要付一小筆 guard 檢查的手續費，換到的是遇到什麼都能應變。AOT 把成本搬到出貨之前，現場每一次呼叫都直達 kernel，沒有檢查也沒有應變，代價是所有的萬一都必須在匯出時想清楚。這個差別會貫穿今天的每一個實驗。

## 全圖承諾，沒有退路

兩條路最本質的差別在對 graph break 的態度。torch.compile 把 graph break 當日常，把函式切成幾段照樣跑。但 export 的產物要在沒有直譯器的地方執行，斷掉的部分沒有人接手，所以它承諾交出的是一張完整的圖，做不到就直接報錯。拿一個資料相依的分支實測，以下實驗都在本機 CPU 上跑（torch 2.8.0），完整程式在 `code/day29/`。

```python
class Branchy(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        return x - 1
```

torch.compile 的處理方式是斷開容忍。

```
[torch.compile on data-dependent branch]
  graph_count=2 graph_break_count=1
  break reason: generic_jump TensorVariable()
```

同一個 module 丟給 export，得到的是一個錯誤。

```
[torch.export on the same module]
  GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)
```

分支條件取決於張量的值，圖沒辦法在編譯期決定走哪一邊，compile 選擇切兩段，export 把問題丟回給你，要嘛改寫成 torch.where 這類圖內表達，要嘛用官方的控制流 op 把兩個分支都收進圖裡。動態的部分也一樣要先講好，compile 可以先當靜態編、變了再重編，export 沒有重編的機會，哪個維度會變、範圍多大，得在匯出時用 Dim 宣告清楚。

這份宣告是一紙雙向的合約。編譯器拿到範圍，就能把這個維度當符號處理，生成一份通吃整段範圍的程式碼，不必為每種 batch 各編一份。使用者這邊則是自我約束，執行期餵進超出範圍的輸入，成品會直接拒絕，而不是默默算錯。automatic dynamic 是跑了兩次之後的事後升格，這裡的 Dim 是事前簽字的條款，同一套 symbolic shape 機制，態度從寬鬆變成嚴格。

## 打開 ExportedProgram

export 的產物叫 ExportedProgram。拿一個小模型匯出，把 batch 維度宣告成動態。

```python
batch = Dim("batch", min=1, max=1024)
ep = export(m, (x,), dynamic_shapes={"x": {0: batch}})
```

印出來的內容節錄如下。

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_fc_weight: "f32[8, 16]", p_fc_bias: "f32[8]", x: "f32[s77, 16]"):
            linear: "f32[s77, 8]" = torch.ops.aten.linear.default(x, p_fc_weight, p_fc_bias);  x = p_fc_weight = p_fc_bias = None
            relu: "f32[s77, 8]" = torch.ops.aten.relu.default(linear);  linear = None
            add: "f32[s77, 8]" = torch.ops.aten.add.Tensor(relu, 1);  relu = None
            return (add,)

Graph signature:
    # inputs
    p_fc_weight: PARAMETER target='fc.weight'
    p_fc_bias: PARAMETER target='fc.bias'
    x: USER_INPUT

    # outputs
    add: USER_OUTPUT

Range constraints: {s77: VR[1, 1024]}
```

三個部分各有各的角色。第一部分是圖本身，op 全部落在 ATen 層，跟 AOTAutograd 展開出來的圖同一族，而且參數不再藏在 module 屬性裡，`fc.weight` 和 `fc.bias` 被抬升成圖的輸入，整張圖是一個沒有隱藏狀態的純函數。第二部分是 graph signature，記著每個輸入輸出的身分，哪些是參數、哪些是使用者輸入，沒有這份對照表，載入的人不知道哪個洞該塞權重、哪個洞該塞資料。第三部分是 range constraints，剛剛宣告的動態 batch 在圖裡以符號 s77 現身，範圍 1 到 1024 白紙黑字寫死，symbolic shape 在這裡變成對外的合約。這三件東西合在一起，就是一張不需要原始 Python 程式碼也能被理解、驗證、編譯的圖。

還有兩個性質值得記下。這張圖是 functionalize 過的，圖裡沒有突變也沒有 alias 陷阱，後面接手的編譯器和驗證工具都好做事。另外 ExportedProgram 可以序列化存檔，在另一個 process 甚至另一台機器讀回來再編，匯出和編譯可以拆成出貨流程裡的兩站，不必擠在同一個環境完成。

![同一個模型分兩條軌道，上軌 JIT 常駐 Python，下軌 export 收成一張圖再鑄成 .pt2](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day29/export_aoti.gif)

*圖一：同一個模型的兩條出路。上軌 torch.compile 常駐 Python runtime，每次呼叫都經過 guard，遇到新狀況當場重編。下軌 export 把全圖收成 ExportedProgram，AOTInductor 鑄成一顆 .pt2，Python 圖層淡出，便當獨自在 C++ runtime 上開飯。*

## 便當盒裡裝了什麼

有了 ExportedProgram，AOTInductor 接手把它變成成品。

```python
torch._inductor.aoti_compile_and_package(ep, package_path="tiny.pt2")
```

.pt2 其實是一個 zip，打開看看。

```
[pt2 package contents]
  tiny/data/aotinductor/model/cq5wuv454krsolah7rcae76pgdsnriveltbeyf4wsndgpgt76clw.wrapper.cpp  (27 KB)
  tiny/data/aotinductor/model/cocidwwsfww37pg7yy5b7f3fo4oy2lgne33ab6he52mejkbfc74o.kernel.cpp  (8 KB)
  tiny/data/aotinductor/model/cq5wuv454krsolah7rcae76pgdsnriveltbeyf4wsndgpgt76clw.wrapper.so  (229 KB)
  tiny/archive_format  (0 KB)
```

主角是那顆編好的 .so，旁邊躺著它的原始碼。kernel.cpp 是 Inductor codegen 生出來的運算 kernel，跟 C++ codegen 那章看過的是同一個產線的貨，檔名裡那串雜湊跟 cache 是同一套命名邏輯，都是拿內容算出來的指紋。wrapper.cpp 最值得停一下，Inductor 平常生成的 wrapper 是一段 Python 程式碼，負責配 buffer、按順序呼叫 kernel，而 AOTInductor 把這一層也翻成 C++，權重打包進檔案，動態 shape 的推導和範圍檢查也編進機器碼。原本 runtime 裡屬於 Python 的最後一份工作，就這樣被編譯期整個吃掉了。

載回來驗收。

```
[load and run]
  allclose(aoti, eager) = True
  batch=512 output shape = (512, 8)
  dynamo frames traced while running .pt2 = 0
  dynamo frames traced after one torch.compile call = 2
```

用 aoti_load_package 載回來跑，輸出跟 eager 完全一致，匯出時用 batch 4，餵 512 也照樣跑，動態維度的合約有效。最後兩行是今天最重要的證據，執行 .pt2 的整個過程 Dynamo 一個 frame 都沒有 trace，對照組 torch.compile 呼叫一次就 trace 了 2 個。runtime 真的完全繞開了 Dynamo，在 Python 裡載入只是方便驗證，同一顆 .so 用 libtorch 的 C++ 介面一樣能載。一個 .pt2 就是一份完整的交付物，wrapper、kernel、權重全在裡面，不必再隨身帶一份 Python 原始碼或 pickle 檔。

## 現場有沒有廚房

選哪條路，看的是執行現場。服務跑在有 Python 的 server 上、輸入形狀多變、模型還在快速迭代，torch.compile 是自然的選擇，寫法幾乎不用改，看不懂的地方自動 fallback，cache 還能把重啟的成本壓下來。反過來，目標是 C++ runtime、行動裝置這類沒有 Python 的環境，或者你要的是啟動即滿速、行為完全可預期的部署，那就走 export 加 AOTInductor，把不確定性全部留在出貨之前。折衷的場景也存在，有 Python 的推論服務可以先把模型用 AOTInductor 編好，Python 端只負責載入呼叫，rolling update 換一批機器也不必再付重編的帳。兩條路共用同一套編譯基礎設施，Dynamo 抓圖、AOTAutograd 攤平、Inductor 生碼，這個系列講過的每一站都沒有白學，變的只是圖從哪裡進來、成品往哪裡去。

## 結語

今天把部署這條路走完。torch.export 用全圖承諾換掉 graph break 的彈性，動態維度用 Dim 寫進合約，產物 ExportedProgram 是一張參數外露、簽名齊全的 ATen 純函數圖。AOTInductor 把它連 wrapper 一起編成 C++，打包成一顆自足的 .pt2，載回來輸出跟 eager 對得上，執行時 Dynamo 零參與。JIT 與 AOT 不是兩套系統，是同一條流水線的兩種出貨方式。

明天是最後一天，把三十天的路重新走一遍，從 bytecode 攔截到 kernel 出爐，把整個 torch.compile 的地圖攤開來總整理。那我們明天見！

## 參考資料

- [PyTorch Docs: torch.export](https://docs.pytorch.org/docs/2.8/export.html)
- [PyTorch Docs: AOTInductor](https://docs.pytorch.org/docs/2.8/torch.compiler_aot_inductor.html)
- [torch/_inductor/__init__.py：aoti_compile_and_package 與 aoti_load_package（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/__init__.py)
