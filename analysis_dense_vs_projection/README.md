# D1 (projection) vs D2 (dense) 的量化分析

論文中兩個 sub dataset 的定義依據, 以及 NBLAST 在 D2 上表現不佳的機制。
所有程式只讀 repository 內既有的資料, 輸出寫進 `results/`。

## 執行

```bash
# 一次性: 建立 NBLAST 環境並算出基準分數
conda create -n nblast python=3.10 -y && conda run -n nblast pip install navis
conda run -n nblast python analysis_dense_vs_projection/run_nblast_official.py --emit-figure-csv

# 主流程 (約 12 分鐘)
bash analysis_dense_vs_projection/run_all.sh
```

## 輸入

| 檔案 | 用途 |
|---|---|
| `labeled_info/D5_conf.csv` | D1 配對 216 組 (109 FC / 143 hemibrain), 含專家信心 |
| `labeled_info/{D2,D6}_conf.csv` | D2 配對 682 組 (150 FC / 349 hemibrain), 含專家信心 |
| `labeled_info/D2+D6_ID.csv` | D2 的 ID 清單 (與上者同一批配對) |
| `data/neuron1x1Coding_Ver2.csv` | 58 個 compartment 的佔位點數。**僅 FlyCircuit** |
| `data/SWC/{FC,EM}/*.swc` | 骨架, 兩個資料庫已對齊到同一標準腦 (µm) |
| `data/descriptors_{FC,EM}/` | 質心與轉動慣量比, 供 prescreening 對照用 |

D1 與 D2 有 3 個 FC、14 個 hemibrain 神經重疊; 所有組間比較都只用互斥名單
(`exclusive == True`)。

## 流程

| 程式 | 產出 |
|---|---|
| `run_nblast_official.py` | `nblast_official.csv` — **獨立於主流程**, 需 navis。用官方 `smat.fcwb` 與雙向平均重算 898 組標註配對 |
| `s01_build_neuron_lists.py` | `neuron_roster.csv`, `pairs.csv` |
| `s02_neuropil_metrics.py` | `neuropil_metrics.csv` — 佔位集中度, side/region, 分母一律含 other |
| `s03_morphology_metrics.py` | `morphology_metrics.csv` — 骨架幾何、多尺度密度、三視圖自我遮蔽 |
| `s04_group_contrast.py` | `contrast_*.csv` — 每個描述子的 AUC / Cliff's δ / 最佳切點 |
| `s05_size_control.py` | 尺寸配對後的 neuropil 訊號與敏感度掃描 |
| `s06_region_criteria.py` | `region_*.csv` — D1 的 MB↔DFP 身分與門檻掃描 |
| `s07_selection_rule.py` | `selection_rule_*.csv` — 交叉驗證的納入條件 |
| `s08_expert_and_nblast.py` | `confidence_by_group.csv`, `nblast_separability.csv` |
| `s09_soma_and_strahler.py` | `soma_*.csv`, `nblast_cable_bins.csv` |
| `s10_em_sponge_effect.py` | `sponge_*.csv` — 海綿效應的九項驗證 |
| `s11_figures.py` | `figures/fig1..fig5` (PDF 向量 + PNG 300 dpi) |

## 方法上必須注意的事

* **不要把 FlyCircuit 與 hemibrain 混在一起比形態。** 同一顆神經在 hemibrain 量到
  的 cable 是 FlyCircuit 的 1.5 倍 (線段中位 0.35 µm vs 4.54 µm), 但空間跨度幾乎
  相同 (0.97)。差的是追蹤精細度。所有骨架比較都在各資料庫內部做, 兩邊都重現才採信。
* **佔比的分母要含 `other`。** 檔案中 58 個 neuropil 總和 + `other` = `volume`,
  48633 筆 100 % 吻合。`other` (纖維束等) 佔比兩組本就不同 (D1 0.225 vs D2 0.161),
  排除它會把 `other` 多的那組灌大。「前兩腦區合計」在排除 other 時 AUC 0.716、
  含 other 時掉到 0.557, 不可作為準則。程式已不再計算排除 other 的版本。
* **密度指標要掃尺度。** 1 µm 重採樣搭配 2 µm 格子時, cable 幾乎不會重複經過同一格,
  指標飽和在 1 而測不出東西。`s03` 因此掃 4/8/16 µm 並加上三視圖投影。
* **NBLAST 分數只用 `run_nblast_official.py` 的輸出。** repository 內原有兩套分數
  彼此不一致 (Pearson 0.93); 腦科中心版的 (similarity+inverse)/2 與官方實作
  Pearson 0.995, 舊版 D2p 家族僅 0.897, 舊版已刪除。
* **`neuron1x1Coding_Ver2.csv` 只有 FlyCircuit** (48633 筆全是 FC id), 且其數值是
  與 cable 長度成正比的點數 (volume / cable ≈ 20 µm⁻¹, Spearman 0.906), 不是解剖體積。
