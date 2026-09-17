# 用外部資料庫的策展型別驗證模型輸出

拿兩個與本專案 CNN 完全獨立的權威來源，判定全庫掃描的配對是否可能是同一型神經，
並據此比較 finetune 與 annotator 兩個模型。

| 端 | 來源 | 取得方式 |
|---|---|---|
| FlyCircuit | Virtual Fly Brain 的 FBbt 策展型別（文獻定義，如 LC12 出自 Wu et al. 2016） | `pdb.virtualflybrain.org` Neo4j，公開帳密 `neo4j:neo4j` |
| hemibrain | neuPrint `hemibrain:v1.2.1` 的 `type` / `instance`（FlyEM 策展） | `neuprint.janelia.org/api/custom/custom`，**不需 token** |

## 執行

```bash
python3 analysis_external_validation/build_pool_ceiling.py   # 先跑，建立隨機基準
python3 analysis_external_validation/run_type_check.py
python3 analysis_external_validation/unseen_stratum.py       # 乾淨分層上的測試集比較
python3 analysis_external_validation/label_rank1_pairs.py [掃描檔]  # 任一份掃描的 rank-1 標同型與否
```

被評估的兩份掃描分別由 `FineTune_miniLR_D1-D6_0`（`result/fc_all_top5.csv`）與
`Annotator_D1-D6_0`（`result/fc_all_top5_annotator_notrunc.csv`）產生，
即 `nrn_service/config.py` 線上服務實際使用的模型。

第一次執行會抓外部資料並存進 `cache/`，之後直接重用。`build_pool_ceiling.py`
會呼叫 `candidate_matching.run_matching` 重建 stage-1 候選池（輸出到暫存目錄，
不動 `data/pairs_label/`）。

## 判定規則

`type_match.py`。四級，仿照專家信心度：

| `type_label` | 意義 |
|---|---|
| `1.0` | 兩邊策展型別相符 → **應該是同一型**（可當正例） |
| `0.5` | 同亞族不同亞型，或 VFB 只標到 lineage 層級 → 形態近親，無法斷定 |
| `0.0` | 兩邊都有策展型別且確定不同 → **一定不是同一顆**（可當負例） |
| 空白 | hemibrain 未給正式 `type`，無法判斷 |

「亞族」的粒度跟著參考資料走，避免用比資料更細的標準去苛求模型：

- **LC**：`LC12` 已是最細單位，LC12 vs LC10 判 `0.0`
- **Kenyon cell**：VFB 只分 core / surface / posterior，hemibrain 另有 `KCab-m`
  等切法，同屬 alpha/beta 的互配判 `0.5`
- **嗅覺投射神經**：以腎小球為亞族，`DL2d_adPN` vs `DL2d_vPN` 判 `0.5`，
  `DL2d` vs `DL2v` 判 `0.0`

## 為什麼要有「可贏性」欄位

命中率單看沒有意義：若某顆 FC 的正確型別根本沒進 stage-1 候選池，兩個模型都不可能答對。
`build_pool_ceiling.py` 重建候選池後算出 `n_correct_in_pool`，`winnable = n_correct_in_pool > 0`。
主要結論一律只看 `winnable` 的子集，並附上 `chance_pct`（從同一個池隨機抽一顆就答對的機率）當基準。

2,735 顆可評估的 FC 中，只有 **1,276 顆（46.7%）**的正解有進候選池。
ALPN 特別慘——hemibrain 裡 `DL2d_adPN` 只有 5 顆、`VM5d_adPN` 只有 8 顆。

## 結果

只算 `winnable` 的 1,276 顆 FC，隨機基準 3.16%：

| 模型 | rank-1 命中 | top5 命中 | rank-1 確定錯誤 |
|---|---|---|---|
| finetune | 22.0 % | 23.7 % | 59.1 % |
| **annotator** | **31.0 %** | **32.5 %** | **46.1 %** |

兩者都遠高於隨機，但 **annotator 明顯較佳**（McNemar 配對檢定 p = 8.8e-10）。

分族群看，差距完全來自 Kenyon cell：

| 族群 | n | finetune rank-1 | annotator rank-1 | p | 結論 |
|---|---|---|---|---|---|
| KC | 846 | 18.1 % | **32.2 %** | 1.6e-12 | annotator 顯著較佳 |
| LC | 241 | 44.9 % | 42.0 % | 0.42 | **無顯著差異** |
| ALPN | 138 | 3.6 % | 2.9 % | 1.0 | 無顯著差異，兩者都近乎無效 |

## 與專家標註的交叉驗證

拿 `labeled_info/D*_conf.csv` 的人工標註回頭驗證判定規則。兩邊都有型別可查的重疊只有
36 筆，樣本很小，但足以抓出規則的錯誤：

| 專家判定 | n | 本規則的判定 |
|---|---|---|
| 同一顆（conf ≥ 0.5） | 7 | 全部 `0.5`（相容，**0 筆被誤判為確定不同**） |
| 不同 | 29 | 22 筆 `0.0`（75.9 %），其餘 `0.5` |

這一步抓到並修掉一個真的 bug：hemibrain 的多腎小球 PN 寫成 `M_<譜系><編號>`
（如 `M_l2PNl20`），譜系 token 在字串**中間**而不是字尾，原本用 `endswith('_lPN')`
判斷 ALl1 譜系會全部漏掉，把 4 筆專家確認的正例錯標成「確定不同型」。
改用 `pn_lineage()` 解析譜系 token 後，「確定不同型」從 14,647 筆降到 14,270 筆
（移除 377 筆偽陰性），模型比較的數字完全不變——因為 lineage 層級的判定本來就只給
`0.5`，不影響 `winnable` 與命中率。

後來又修了反方向的寬鬆：VFB 只標到「adult uniglomerular antennal lobe projection
neuron」時既沒有 prefix 也沒有譜系可查，舊規則對任何 EM 都給 `0.5`，連 PFNd、DNp23
這種根本不是投射神經的也算「相容」。現在 lineage 層級的判定至少要求 EM 端同屬 ALPN
（連帶讓 `em_family` 認得 `VP1d+VP4_l2PN1` 這種字尾帶編號的 PN），173 筆從 `0.5`
改為 `0.0`。這 173 筆全落在不可贏的 FC，`winnable` 範圍的數字與上表的專家交叉驗證
都不變；只有全體範圍的 rank-1 確定錯誤率上升（finetune 74.5 → 75.5 %、
annotator 64.3 → 65.1 %）。

## 輸出

- `results/pair_labels.csv` — 25,341 筆配對，每筆含 `type_label`、`relation`、
  `basis`（判定依據的中文說明）、兩個模型的 rank/score、候選池資訊。
  其中確定同型 3,274 筆、確定不同型 14,443 筆、不確定 3,330 筆、無法判斷 4,294 筆。
- `results/<掃描檔名>_rank1_type.csv` — `label_rank1_pairs.py` 的輸出，欄位比照
  `pair_labels.csv`，另加 `same_type`（True / False / Uncertain / Unknown）。涵蓋全部 rank-1，
  判定不了的也保留並在 `relation` 註明原因。一顆 FC 有多個 VFB 標註時全部都判
  （任一個判確定不同即為不同），所以與 `pair_labels.csv` 在這類 FC 上可能不同。
- `results/model_summary.csv` — 兩模型比較（全體與 winnable 兩種範圍）
- `results/by_family.csv` — 分族群比較
- `results/mcnemar.csv` — 配對顯著性檢定
- `results/unseen_stratum.csv` — 十折測試集依「神經有沒有在訓練時見過」分層的 AUC
  （含 bootstrap CI）。「兩側都沒看過」那層 n = 69：annotator 0.887、
  `FineTune_miniLR` 0.955，ΔAUC +0.068 CI [+0.001, +0.146]。
  **fine-tune 的二分類優勢在未見神經上仍成立**，不能歸因於洩漏；
  它在全庫檢索上輸掉的原因是分數膨脹（對「確定不同型」配對有 18.8 % 給 ≥0.95，
  annotator 僅 1.6 %），詳見 `analysis_model_results/FINDINGS.md` §10。

## 限制

- **`1.0` 與 `0.0` 的強度天差地遠。** `0.0`（兩邊都有策展型別且不同）接近硬證據；
  `1.0` 只是「沒有被排除」。型別相符之後還剩多少模糊度，各族群差很多：

  | 族群 | 候選池中位 | 其中同型別者中位 |
  |---|---|---|
  | ALPN | 3,109 | **1** ← 型別相符幾乎等於個體確認 |
  | LC | 1,036 | 31 |
  | KC | 3,075 | **138** ← 仍是 138 選 1 |

  專案的專家標註是**個體層級**的（每顆 FC 中位數只給 1 顆正例、最多 8 顆），
  比型別層級嚴格得多，所以 `1.0` 不能直接當正例訓練標籤用。
- `0.0` 也不是邏輯上的鐵證，它預設兩邊的策展標註都正確。已知的失效模式是
  跨資料集的型別命名不對齊（見上節那個 `M_l2PNl20` 的 bug），以及 FlyCircuit 端
  的型別本來就是從光學影像判讀、自帶不確定性。
- 可評估的族群限於型別名稱能明確對應 hemibrain 的三類，佔 VFB 有標註 FC 的一部分，
  不是全庫的隨機抽樣。KC 佔了 1,638 顆，總體數字被它主導，所以務必看分族群的表。
- FlyCircuit 含雄性神經與左半腦神經，hemibrain 是單一雌性右半腦；
  這類 FC 本來就沒有正確答案，多半落在「不可贏」而被排除，但不保證完全乾淨。
- `0.0` 的可信度高於 `1.0`：兩邊都有策展型別且不同，是硬證據。
