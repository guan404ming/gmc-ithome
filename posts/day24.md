# Day 24 | 編譯費能只付一次嗎？torch.compile 的置物櫃與鑰匙

## 前言

昨天看了 autotune 怎麼幫一顆 kernel 海選出最快的寫法，答案很準，帳單也很驚人。不只 autotune，整條 pipeline 從 Dynamo trace 到 Inductor codegen 都是拿啟動時間換執行速度，如果每次重開 process 都得再付一次，這筆交易就很難划算。好消息是 torch.compile 把快取做得相當徹底，而且分了好幾層。今天就來看編譯成果存在哪裡、開櫃子的鑰匙是用什麼打出來的、什麼時候會突然開不了鎖、又要怎麼確認自己有沒有拿到便宜。對照的原始碼主要在 [`torch/_inductor/codecache.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codecache.py)。

正文開始！

## 一次編譯的帳單

先量一下多貴。實驗照舊在本機 CPU 上跑（torch 2.8.0），拿一個 matmul 加 gelu 加 sum 的小函式當實驗品，完整程式在 `code/day24/`。

```python
def f(x, y):
    return torch.nn.functional.gelu(x @ y + 1).sum(dim=1)
```

把快取目錄整個刪掉，從全新狀態編一次，log 節錄如下。

```
[run 1: fresh cache dir, n=512]
  first call (compile):   3.75 s
  second call (cached):   1.03 ms
  fxgraph_cache_miss=1 fxgraph_cache_hit=0
```

第一次呼叫的 3.75 秒裡疊著整個系列講過的每一站，Dynamo 逐條 bytecode trace、AOTAutograd 展開成 ATen 圖、Inductor 做 lowering 加 scheduling 加 codegen，最後還要請 g++ 把生成的 C++ 編成 `.so`，GPU 上再加上 Triton 編譯與 autotune。這還只是三行的玩具函式，真實模型有幾百張圖要走這條流水線，冷編譯輕鬆上到分鐘級，開了 max-autotune 還要再翻幾倍。而付錢的場合比想像中多，訓練 job 重啟一次付一次，推論服務 rolling update 又付一次。

同一個 process 裡的第二次呼叫只要 1.03 ms，不過這是 Guard 那一層，成品掛在 code object 上，guard 驗過就直接放行，連 trace 都不用重走。真正的問題是 process 一關，這層就跟著蒸發。

## 跨 process 的第二層

同一個快取目錄，重開一個 process，編同一個函式。

```
[run 2: same cache dir, n=512]
  first call (compile):   0.79 s
  second call (cached):   0.73 ms
  fxgraph_cache_miss=0 fxgraph_cache_hit=1
```

3.75 秒變 0.79 秒，這就是磁碟上的 FXGraphCache 在接手。它接手的位置在 Inductor 的正門口，拿到一張 ATen 圖後先不急著編，把圖和輸入的 metadata 拼成一把鑰匙，去磁碟上找現成的貨。找到了就把 pickle 過的成品反序列化、把編好的 `.so` 掛上，整段 lowering 到 codegen 直接跳過。找不到才走完全程，再用這把鑰匙把成品存進去，留給下一個 process。

`fxgraph_cache_hit=1` 來自 `torch._dynamo.utils.counters`，是確認快取有沒有吃到最直接的辦法。要看更細可以開 `TORCH_LOGS="+torch._inductor.codecache"`，log 會直接把鑰匙印給你。

```
fx graph cache hit for key fih5y2v32k3xkqrqilffnjrj36iym4ih5wnqc46hpvqv4bwj3hgr
```

值得注意的是 0.79 秒沒有歸零。Dynamo 那段沒得省，新 process 還是得重新 trace、重建 guard，磁碟快取救的是後半段，graph pass、fusion 決策、codegen、C++ 編譯全部變成一次磁碟讀取加反序列化。

## 打開置物櫃看看

快取目錄預設在 `/tmp/torchinductor_<user>`，可以用 `TORCHINDUCTOR_CACHE_DIR` 改位置。介紹 Inductor 時就在 log 裡瞥見過它，今天把門打開，跑完兩輪後裡面長這樣。

```
  3b/c3b5dchwczelzcuy6vgbf56y5ycgcs5f5imcnyqztbduxgwcqnhj.py  (5 KB)
  aotautograd/ae7ovyzz7gtp7s2l7imxtwrtul3tnk5iyrsuzsv7iofjnwjn4nfi/pvjhno2m3ulqefopwlqu7hl7wg5vewnw7m5vtofyduifarxr4ry  (42 KB)
  b7/cb7wkkioof5pkquoovazf5u7tubwi65tnzsp6psbxafpkeexwd2w.main.so  (53 KB)
  dq/cdqdltvbalg3t3on3nulfqdrac4xhwkkplet7ypqfruiazm33qkc.so  (49 KB)
  fxgraph/ih/fih5y2v32k3xkqrqilffnjrj36iym4ih5wnqc46hpvqv4bwj3hgr/3wllna56mmhxckjexy2lg7vlpok73hgbfrkepzzttbtcfkf4mzv  (47 KB)
```

一格一格認過去。`fxgraph/` 是 FXGraphCache 的本體，pickle 過的整份編譯成品，第一層目錄名就是那把 `fih5...` 開頭的鑰匙。`aotautograd/` 是 AOTAutogradCache，把 AOTAutograd 的展開結果也存了下來，hit 的時候連 joint graph 的 trace 和 partition 都不用重跑，原始碼在 [`autograd_cache.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/autograd_cache.py)。那個 `.py` 存的是 output code，就是 TORCH_LOGS 印出來的那份 wrapper 加 kernel。兩個 `.so` 屬於 CppCodeCache，g++ 編好的動態庫，檔名就是 C++ 原始碼的 hash。locks 目錄則是多個 process 同時編譯時上的鎖，免得搶著寫同一格櫃子。

GPU 上還會多一種住戶。autotune 海選出的 best config 會以小檔案的形式存在 kernel 旁邊，下一次遇到同一顆 kernel 直接讀答案，不再重新 benchmark，原始碼在 [`autotune_cache.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/runtime/autotune_cache.py)。昨天燒掉的那些海選時間，就是靠這格櫃子只付一次。順帶一提，這整個目錄可以放心刪掉，它是純粹的快取，最壞的結果就是下次啟動回到冷編譯的價錢。

## 鑰匙是怎麼打出來的

一把能跨 process 使用的鑰匙，必須保證「鑰匙相同就代表編譯結果相同」，所以材料比直覺想的多。`codecache.py` 裡把材料列得清清楚楚，圖本身、example inputs 的 metadata、編譯參數、整份 inductor config、deterministic 相關的全域設定，再加上 torch 版本號連同整份 Inductor 原始碼算出的 hash。全部序列化之後做 sha256，就是那串 `f` 開頭的 key。

幾個細節值得停一下。輸入進鑰匙的不是張量本身而是 shape、dtype、stride 這些 metadata，畢竟編譯結果只依賴形狀不依賴數值，FakeTensor 讓「只留形狀」有現成表示法。config 必須整份進鑰匙，因為同一張圖在不同開關下生成的程式碼真的不同，待會就有實驗作證。連 torch 原始碼都要 hash 進去，是因為 Inductor 改版之後同一張圖可能生出不同的 kernel，快取的底線是寧可白編，不可錯拿。材料裡任何一項變了，鑰匙齒形就變了，打開的會是另一格空櫃子。

整個機制用動畫走一遍。

![編譯成品被拼出一把鑰匙存進櫃子，第二次直接開櫃，shape 一變鑰匙就對不上](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day24/cache.gif)

*圖一：第一次編譯走完整條慢路，成品存進置物櫃，鑰匙由 graph、shape、dtype、config、torch 版本五塊碎片熔成。第二次同樣的請求打出同一把鑰匙，直接開櫃取貨。shape 一換，其中一塊碎片變形，鑰匙插不進鎖孔，只好重走慢路，再打一把新鑰匙。*

## 齒形對不上的時候

拿實驗驗證兩種失效。第一種是換 shape，輸入從 512 改成 768 再開一個 process。

```
[run 3: same cache dir, n=768]
  first call (compile):   3.12 s
  fxgraph_cache_miss=1 fxgraph_cache_hit=0
```

靜態圖的鑰匙裡嵌著具體的 shape，768 打出來的就是另一把，整段編譯重付。這也呼應 automatic dynamic，如果圖已經被標成 dynamic，shape 就以符號形式進鑰匙，這種 miss 自然不會發生，dynamic 的成品存進櫃子時還會帶著自己的 shape 條件，取貨前要再驗一次才算數。第二種是改 config，開 `TORCHINDUCTOR_CPP_WRAPPER=1` 再跑一次原本的 512。

```
[run 5: same cache dir, n=512, TORCHINDUCTOR_CPP_WRAPPER=1]
  first call (compile):   3.19 s
  fxgraph_cache_miss=1 fxgraph_cache_hit=0
```

函式沒動、shape 沒動，只是 config 變了，一樣 miss，因為 config 改變真的會改變生成的程式碼，鑰匙必須誠實。同理，升級 PyTorch 版本會讓 torch 版本這塊碎片整個換掉，一櫃子存貨瞬間全數作廢，這是部署時最常見的「怎麼今天啟動特別慢」的原因。實務上懷疑快取沒吃到，排查順序就出來了，先看 counters 是 hit 還是 miss，是 miss 就開 codecache 的 log 把兩次的鑰匙印出來對，log 的 hash details 會一項一項告訴你是 shape 變了、config 變了，還是有人偷偷升了版本。另外 miss 不是錯誤，新成品會用新鑰匙再存一格，舊的那格也還留著，同一張圖的 512 版和 768 版可以並存，代價只是櫃子越住越滿。

## 把這層關掉會發生什麼

最後一個對照組把 FXGraphCache 關掉，看少了它會退化到哪。

```
[run 4: same cache dir, n=512, TORCHINDUCTOR_FX_GRAPH_CACHE=0]
  first call (compile):   2.02 s
  fxgraph_cache_miss=0 fxgraph_cache_hit=0
```

counters 兩邊都是 0，這層被整個跳過，AOTAutograd 和 Inductor 的活全部重做。有趣的是 2.02 秒介於冷的 3.75 和熱的 0.79 之間，因為 CppCodeCache 還活著。重新生成的 C++ 內容一模一樣，hash 一樣，`.so` 直接複用，省下的正是 g++ 那段。這個數字把分層結構講得很白。同 process 內是 guard 守著的 code cache，跨 process 最上層是存整份成品的 FXGraphCache 和 AOTAutogradCache，再往下是按單顆產物計價的 CppCodeCache 和 autotune 的 best config。每一層失效的條件不同，打翻上層，下層還會各自接住能接的部分。

出了這台機器，同一套鑰匙還能再往外延伸。設 `TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE` 可以把 FXGraphCache 接上 Redis，整個 cluster 共享一格櫃子，一台機器編過的圖其他機器直接取貨，autotune 那邊也有對應的 `TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE`。另一條路是 `torch.compiler.save_cache_artifacts()`，把這次編譯碰過的快取打包成一坨 bytes，在 CI 裡先熱好，部署機啟動時 load 回來直接開跑。細節點到為止，知道櫃子可以搬出去就夠了。

## 結語

把今天的帳算一遍。一次冷編譯 3.75 秒，錢花在 trace、展開、codegen 和真正的編譯器上。同 process 內靠 guard 直接放行，跨 process 靠磁碟上的 FXGraphCache 降到 0.79 秒，鑰匙由圖、shape、dtype、config、torch 版本熔成，任何一項變動都是一次誠實的 miss。快取目錄裡分層住著 fxgraph、aotautograd、cpp 的產物，關掉一層還有另一層接著，整櫃還能搬上 Redis 共用。

不過快取省下的都是編譯期的錢，模型跑起來之後還有一種執行期的零碎開銷在偷時間，每呼叫一次 kernel 就要付一次 launch 手續費，模型越碎付得越多。明天來看 CUDA Graph 怎麼把一整串 kernel launch 錄成一卷帶子直接重播，`mode="reduce-overhead"` 背後就是這件事。那我們明天見！

## 參考資料

- [torch/_inductor/codecache.py：FxGraphCache 與 FxGraphHashDetails（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/codecache.py)
- [torch/_functorch/_aot_autograd/autograd_cache.py：AOTAutogradCache（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/_aot_autograd/autograd_cache.py)
- [torch/_inductor/runtime/autotune_cache.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/runtime/autotune_cache.py)
- [PyTorch Docs: Compile Time Caching in torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
