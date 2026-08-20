# Day 15 | Joint Graph 與 Partitioner：backward 該存什麼、該重算什麼

## 前言

Day 12 留下的懸案：forward 圖多輸出了 `le` 和 `permute` 給 backward，誰決定的？答案是 AOTAutograd 其實先把 forward 和 backward trace 成一張圖，再用一個切割演算法把它切成兩半。切線落在哪，決定了「forward 結束時要保存哪些中間值」，這是訓練記憶體的主要去向，也是速度與記憶體之間最重要的旋鈕。

正文開始！

![joint graph 上的一刀](https://raw.githubusercontent.com/guan404ming/gmc-ithome/main/assets/day15/min_cut.png)

*圖一：joint graph 與那一刀。forward 和 backward 先畫成一張圖，min-cut partitioner 以保存成本為邊權重找最小割：跨線的 `mm`、`permute` 成為保存值，便宜的 `tanh` 留給 backward 重算；checkpoint 是把切線推到極端、只存輸入。*

## 先看那張沒切開的圖

`TORCH_LOGS="aot_joint_graph"` 可以看到切割前的原貌。拿 `f(x, w) = tanh(x @ w).sum()` 實跑：

```python
 ===== Joint graph 0 =====
def forward(self, primals, tangents):
    mm = torch.ops.aten.mm.default(primals_1, primals_2)
    tanh = torch.ops.aten.tanh.default(mm)
    sum_1 = torch.ops.aten.sum.default(tanh)          # <- forward 的輸出
    expand = torch.ops.aten.expand.default(tangents_1, [64, 64])
    mul = torch.ops.aten.mul.Tensor(tanh, tanh)       # tanh 的導數 1 - tanh^2
    sub = torch.ops.aten.sub.Tensor(1, mul)
    mul_1 = torch.ops.aten.mul.Tensor(expand, sub)
    permute = torch.ops.aten.permute.default(primals_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(permute, mul_1)  # <- w 的梯度
    return pytree.tree_unflatten([sum_1, None, mm_1], self._out_spec)
```

輸入同時有 `primals`（forward 的輸入）和 `tangents`（上游梯度），輸出同時有 loss 和梯度。這張圖上 forward 和 backward 只是前後兩段節點，中間靠資料流相連：backward 用到了 `tanh`（算導數）和 `primals_1`（算 `w` 的梯度）。

## 切一刀

Partitioner（[`torch/_functorch/partitioners.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/partitioners.py)）要決定：哪些節點歸 forward、哪些歸 backward、跨過切線的值就是要保存的。看切完的結果：

```python
 ===== Forward graph 0 =====
    mm = aten.mm(primals_1, primals_2)
    tanh = aten.tanh(mm)
    sum_1 = aten.sum(tanh);  tanh = None
    permute = aten.permute(primals_1, [1, 0])
    return (sum_1, mm, permute)          # 保存 mm 和 permute

 ===== Backward graph 0 =====
def forward(self, mm, permute, tangents_1):
    tanh = aten.tanh(mm);  mm = None     # <- 在 backward 裡重算 tanh！
    mul = aten.mul(tanh, tanh)
    ...
```

有意思的地方來了：joint graph 裡 backward 直接用 `tanh`，但切完之後保存的是 `mm`，backward 開頭自己又算了一次 `tanh`。為什麼不直接存 `tanh`？因為存 `mm` 和存 `tanh` 記憶體一樣大，但這裡預設的 min-cut partitioner 在「可重算的便宜節點」上偏向重算：`tanh` 是 pointwise，重算幾乎免費，還可能跟 backward 的其他 op 融合掉；而少存一個值就少一份 activation 記憶體佔用的機會。切線的目標是讓「跨線傳輸的量」最小，這就是 min-cut 這個名字的來源：把保存成本當邊權重，在 joint graph 上求最小割。

`permute` 則相反：它只是 metadata 操作，「存」它不花記憶體，直接留在 forward。

## 把旋鈕轉到底：activation checkpointing

Min-cut 是自動的折衷，你也可以手動把「重算」開到最大。`torch.utils.checkpoint` 包住的區段，partitioner 一個中間值都不存：

```python
 ===== Forward graph 1 =====
    mm = aten.mm(primals_1, primals_2)
    tanh = aten.tanh(mm)
    sum_1 = aten.sum(tanh)
    return (sum_1, primals_1, primals_2)   # 只存輸入！

 ===== Backward graph 1 =====
def forward(self, primals_1, primals_2, tangents_1):
    mm = aten.mm(primals_1, primals_2)     # 整段 forward 重算一遍
    tanh = aten.tanh(mm)
    ...
```

Forward 只保存原始輸入，backward 開頭把 `mm`、`tanh` 全部重算。記憶體從「存 activation」變成「存輸入」，代價是 backward 多付一次 forward 的計算。大模型訓練的 activation checkpointing 在編譯棧裡就是這麼做出來的：不是魔法，只是 partitioner 換了一個策略。

## 這一刀為什麼重要

訓練的 GPU 記憶體大頭不是權重是 activation：每一層 forward 的中間值都得留到 backward 用完才能放。切線往 forward 靠（多存）記憶體漲、速度快；往 backward 靠（多算）記憶體省、速度慢。Min-cut partitioner 讓這個決策從「autograd 引擎存所有東西」的固定行為，變成一個圖上的最佳化問題，而且因為 backward 也是 Inductor 要編譯的圖，重算的 op 常常能融進 backward 的 kernel，重算的實際代價比 eager 世界裡低得多。這是編譯式訓練相對 eager 的一個結構性優勢。

## 結語

AOTAutograd 的全貌到此完整：FakeTensor 重跑 forward、autograd 引擎展開 backward，合成一張 joint graph；Functionalization 去掉 mutation、Decomposition 拆成基本 op；min-cut partitioner 一刀切開，切線上的值就是要保存的 activation，便宜的節點傾向重算，checkpointing 是這個旋鈕的極端值。兩張乾淨的 ATen 圖，一前一後交給 Inductor。

明天進入第三站。Inductor 拿到圖之後，第一步不是生程式碼，而是把每個 ATen op 翻譯成它自己的中間表示：一種「用 Python 函式描述的迴圈」。那我們明天見！

## 參考資料

- [torch/_functorch/partitioners.py（v2.8.0）](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_functorch/partitioners.py)
- [Min-cut recomputation 的設計討論（PyTorch dev-discuss）](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)
- [torch.utils.checkpoint 文件](https://pytorch.org/docs/stable/checkpoint.html)
