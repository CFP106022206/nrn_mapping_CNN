# per-EM 去偏實驗：結果是負的

**結論先講：per-EM 去偏無法回收 finetune 與 annotator 之間的差距，
「海綿效應／hubness 是主要機制」這個假設不成立。**

## 背景

`analysis_model_results/FINDINGS.md` §10 原本推測 rank-1 崩壞來自 **per-EM 偏移**——
某些 EM 不分 FC 一律被拉高。若成立，減去一個每顆 EM 各自不同的基線就能重排回來，
**不需要重訓**。這支實驗就是去驗證它。

（跨模態檢索的 hubness 有成熟解法：CSLS、NICDM、local scaling。
⚠️ CSLS 原式 `2s − r_e − r_f` 的 `r_f` 在**單一 FC 的池內是常數**，不改變該池排序，
所以對 rank-1 而言 CSLS ≡ `s − r_e/2`，只是置中搭配收縮係數 0.5。本實驗直接掃 α。）

## 作法

```bash
python3 analysis_hubness_debias/dump_pool_scores.py --model finetune   # 約 15 分鐘
python3 analysis_hubness_debias/dump_pool_scores.py --model annotator
python3 analysis_hubness_debias/debias.py --model finetune  --baseline-from all
python3 analysis_hubness_debias/debias.py --model annotator --baseline-from disjoint
```

- `dump_pool_scores.py` 把**整個候選池**的分數 dump 出來（`result/fc_all_top5*.csv`
  只存前 5 名，估不出基線）。不需改 `match_cli.py`：`NeuronMatchService.query()`
  本來就算完整池，`full_table` 保留全部。
  已驗證重算的 top5 與 `result/fc_all_top5.csv` 完全相同、分數差 3e-8、池大小逐顆吻合。
  2 735 顆 FC × 兩模型 = 各 **5 308 333 對**，各約 15 分鐘（GPU，約 5 800 pair/s）。
- 基線估計集分兩版：`all`（轉導式，不用標籤）與 `disjoint`
  （只用與評估集不相交的 FC，含全部 1 459 顆 non-winnable —— 零循環且免費）。

## 結果

只算 winnable FC，`none` 是未去偏的基線：

| 模型 | 基線 rank-1 | 最佳去偏 | Δ | 顯著性 |
|---|---|---|---|---|
| finetune（all，n=1276） | 21.97 % | 24.58 %（center α=0.25） | +2.61 pp | p = 0.055 |
| finetune（disjoint，n=638） | 21.17 % | 26.44 %（center α=0.25） | +5.27 pp | — |
| annotator（all） | 31.01 % | 31.28 %（center α=0.25） | +0.27 pp | p = 0.791 |
| annotator（disjoint） | 30.87 % | 31.68 %（center α=0.25） | +0.81 pp | — |

配對檢定（同一批 FC，去偏前後的 rank-1 命中）：
finetune 淨 +33（156 勝 / 123 敗，**p = 0.055**）、annotator 淨 +5（**p = 0.791**）。
**兩者都未達顯著。**

### 池內 AUC 才是關鍵數字

`pool_auc` 是每顆 FC 各自算「型別相符 vs 確定不同型」的 AUC 再平均。
它是**唯一不條件在模型自己 top-5 上**的排序品質量測：

| 模型 | 池內 AUC | p@5 | 正解 rank 中位 |
|---|---|---|---|
| finetune | **0.861** | 53.9 % | 5 |
| annotator | **0.898** | 67.1 % | 3 |

**annotator 在整個池的各個深度都排得比較好，不只是頂端。**
這不是校準假象，事後修正不了。

### 去偏一律降低池內 AUC

11 種設定、兩個模型、兩種基線來源，**沒有任何一組讓 pool AUC 上升**。
強去偏更是災難性：

| 設定 | finetune rank-1 | annotator rank-1 |
|---|---|---|
| 未去偏 | 21.97 % | 31.01 % |
| center α=1.0 | 13.62 % | 16.01 % |
| ranknorm | 6.99 % | 8.31 % |
| zscore | 2.57 % | 0.80 % |

→ **分數裡的 per-EM 成分本身帶有真實訊號**，拿掉它破壞的比修好的多。
「某些 EM 比較容易被配上」有一部分是生物事實（那些 EM 真的長得像很多東西），
不全是模型偏差。

## 這推翻了什麼

1. **§10 的「per-EM 偏移」不能再當主要機制。** 偏移確實存在
   （rank-1 集中度 2 401 vs 3 507 顆 EM），但修正它回收不了差距。
2. **「差距完全來自候選進場」也不成立。** 池內 AUC 顯示 annotator 在所有深度都較好，
   進場與排序不是兩件可分的事——只有一個排序，annotator 的整條比較好。
3. **階段 1 對線上服務 no-go。** 提升不顯著，不值得增加一層後處理。
   服務端的建議維持不變：**單用 annotator**（見 `RANKING_FINETUNE_PLAN.md` 階段 0）。

## 這強化了什麼

差距是**表徵／評分品質**的真實差異，不是可後處理的偏差。
所以「訓練還能不能榨出更多」變成真正的問題
→ **階段 2 的凍結 trunk 探針成為決定性測試**。

## 後續：EM 先驗來自哪裡（`expert_em_prior.py`）

階段 3 的排序微調（`data_process_rank.py`）留下一個問題：
EM 先驗隨「繼承多少 annotator 權重」單調上升（7.3 % → 11.4 % → 20.9 %），
那它是不是模型記住了專家標註過的 EM（identity shortcut）？

`expert_em_prior.py` 把每顆 EM 依「在 `train_split_0` 裡以什麼身分出現過」分組，
比較它們在全池的表現。FC 側幾乎不重疊（2 735 顆裡只有 6 顆被標註過），
所以兩邊能傳遞的只有 EM 側資訊。

**答案是否定的，而且方向相反：**

| 分組 | EM 數 | 全池平均分 | rank-1/EM |
|---|---|---|---|
| pos_only | 227 | 0.193 | 0.088 |
| neg_only | 241 | 0.127 | 0.012 |
| both | 75 | 0.078 | 0.000 |
| **unseen** | 12 201 | **0.281** | **0.222** |

Δ 平均分全部顯著為負（`pos_only` −0.088 [−0.103, −0.070]、`neg_only` −0.154、
`both` −0.202，P(Δ>0) 皆 0.000），且有單調的劑量反應但方向相反
（在專家訓練集出現愈多次，全池平均分愈低）。
專家見過的 EM 佔候選池 4.26 % 的 EM，只拿下 0.84 % 的 rank-1，**富集倍數 0.2x**。

⚠️ **判讀的限制**：`mean_score` 量的是「海綿程度」而非「品質」——
它是該 EM 在 2 735 顆**不相干** FC 上的平均分，一顆只對某顆 FC 是好配對的 EM，
平均分照樣很低。所以這張表真正說的是「專家標註過的 EM 比池子平均更不海綿」，
這合理（會被送去人工判讀的本來就是形態較有辨識度的神經），
但它**不能**反過來當成「專家標註品質差」的證據。

→ 先驗不是身分記憶。現行的機制假設（型別層級標籤無法懲罰通用好度）
與驗證方式見 `RANKING_FINETUNE_PLAN.md` §機制假設。

## 專家標註與型別判定有沒有衝突（`expert_type_consistency.py`）

問題：排序微調的守門 AUC（專家測試集）一直低於 annotator，
會不會是型別層級的排序訊號與個體層級的專家標註互相矛盾？

**答案：兩者幾乎沒有交集，衝突無從發生。**

- 型別判定只涵蓋 KC / LC / ALPN 三族；專家標註的 FC 多是 PPL1 多巴胺神經、MBON14、
  fruitless 神經等。訓練集 1 097 對只有 **30 對**、測試集 122 對只有 **6 對**
  套得上型別判定（缺口在 FC 側：沒有可用的 VFB 型別預期；EM 側全都有型別）。
- **同型（1.0）配對一對都沒有**，所以「同一顆 FC 內，同型的專家分數高於不同型」
  找不到可以比的 FC。
- 在這點交集裡：型別 0.0 的 22 對專家分數全是 0（零硬衝突）；
  型別 0.5 的 14 對有 7 對被專家判為正例（0.5 = 同亞族無法斷定，不算衝突）。

同一支腳本也把各輪的守門 AUC 做配對 bootstrap（122 對），發現兩件事：
**0.9293 是十折門檻，annotator 在 fold 0 自己只有 0.9050**；
修好 bug 的組態（0.856–0.865）與 0.9050 的差距 CI 都包含 0，**不顯著**。

## 輸出

- `scores/pool_scores_{model}.parquet` — 全池分數（各 530 萬對，各 34 MB）
- `results/debias_{model}_{all,disjoint}.csv` — 11 種設定 × 6 個指標
- `results/expert_em_prior_{per_em,groups}.csv` — 每顆 EM 的全池表現與分組摘要
- `results/expert_type_consistency_{train,test,guard}.csv` — 專家配對的型別判定與守門 AUC 拆解
