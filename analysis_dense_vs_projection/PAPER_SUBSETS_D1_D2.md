# 論文 Results：D1 / D2 子資料集的定義與用詞依據

> 對應論文 Results 的兩個小節：
> **D1** = 程式中 `labeled_info/D5_conf.csv`（組名 `projection`）；
> **D2** = 程式中 `labeled_info/D2_conf.csv` + `D6_conf.csv`（組名 `dense`）。
>
> 本文件只整理**子資料集怎麼定義、標題與用詞能說到哪裡**。NBLAST 比較見 `FINDINGS.md` §3–4。
> 數字可由 `s06_region_criteria.py`（→ `results/region_wording_support.csv`）與
> `s07_selection_rule.py`（→ `results/selection_rule_thresholds.csv`）重現。

---

## 0. 資料與定義（引用任何數字前先讀）

| 項目 | 說明 |
|---|---|
| 腦區資料 | `data/neuron1x1Coding_Ver2.csv`，**只有 FlyCircuit**。以下腦區數字全為 FC 端，hemibrain 無對應表 |
| 數值意義 | 每個 neuropil 內的重建點數，與 cable 長度成正比（volume / cable ≈ 20 µm⁻¹，Spearman 0.906）。所以「佔比」= **該神經有多少比例的神經突落在此腦區** |
| 量不到的 | 突觸（是否真的 innervate）、soma 位置、投射方向 |
| 分母 | 除特別註明外皆為 `volume` = 58 個 neuropil + `other`（已驗證 48 633 筆完全相等） |
| 左右 | 腦區佔比將左右半腦合併（`mb_4_l` + `mb_4_r` → MB） |
| **主腦區** | 佔比最大的腦區 |
| MB 與 calyx | FlyCircuit 將 calyx 獨立編碼為 `cal_18`，因此 MB（`mb_4`）指 calyx 以外的 MB |

**樣本數對帳**

| | 論文數字 | 在 neuron1x1Coding 內 | 僅屬單一子集（分析用） |
|---|---|---|---|
| D1 FC | 109 | 106 | **103** |
| D2 FC | 150 | 150 | **147** |

- 3 顆 D1 神經不在 neuron1x1Coding 內。
- `TH-F-100083`、`TH-F-100099`、`TH-F-200081` **同時出現在 D1 與 D2**，分析時排除。

論文若引用百分比，建議註明分母為 103 / 147，或說明兩子集在 FC 端不完全互斥。

---

## 1. D1：主腦區為 MB 或 DFP

### 1.1 核心指標

| 條件 | D1 (n = 103) | D2 (n = 147) |
|---|---|---|
| **主腦區為 MB 或 DFP** | **98.1 %**（101） | 34.0 %（50） |
| 　主腦區為 MB | 61.2 %（63） | **0.0 %**（0） |
| 　主腦區為 DFP | 36.9 %（38） | 34.0 %（50） |
| 主腦區種類數 | **4** | **12** |

D1 的佔比中位：MB 34.8 %、DFP 31.4 %。

不符合的 2 顆都有不少 MB，只是 MB 不是第一名：

| 神經 | 主腦區 | MB 佔比 |
|---|---|---|
| `VGlut-F-300680` | EB | 27.1 % |
| `VT43401-F-700002` | LH | 17.0 % |

### 1.2 預期會被追問：D2 也有 34 % 以 DFP 為主

以 DFP 為主的比例兩組幾乎一樣（36.9 % vs 34.0 %）。**DFP 本身不區分兩組，區分兩組的是 MB**：

| 主腦區為 DFP 的神經 | D1 (n = 38) | D2 (n = 50) |
|---|---|---|
| 同時 MB ≥ 5 % | **92.1 %** | 48.0 % |
| MB 佔比中位 | **31.0 %** | 4.5 % |

D1 裡以 DFP 為主的神經，仍有約三成神經突在 MB；D2 的則大多只擦過 MB。

**建議寫法**：用「主腦區為 MB 或 DFP」描述 D1 本身；需要與 D2 對照時，補一句
「97.1 % 的 D1 神經有 ≥ 5 % 的神經突位於 MB（D2 為 22.4 %）」。

### 1.3 其他候選指標（參考，不建議當標題）

| 條件 | D1 | D2 | 評語 |
|---|---|---|---|
| MB ≥ 5 % | 97.1 % | 22.4 % | 最能區分兩組的單一腦區 |
| MB 與 DFP **皆** ≥ 5 % | 92.2 % | 21.8 % | 5 % 門檻為人為選定；且暗示「兩區都要有」，比挑選動機更強 |
| MB 或 DFP **任一** ≥ 5 % | 100.0 % | 60.5 % | 太鬆，D2 也過半 |
| 前兩名剛好是 {MB, DFP} | 85.4 % | 13.6 % | 最嚴，但覆蓋率較低 |

「皆 ≥ X %」與「任一 ≥ X %」的門檻敏感度（D1 / D2）：

| 門檻 | 任一 | 皆 |
|---|---|---|
| 1 % | 100.0 / 72.1 | 97.1 / 42.9 |
| 5 % | 100.0 / 60.5 | 92.2 / 21.8 |
| 10 % | 100.0 / 50.3 | 84.5 / 12.9 |
| 20 % | 98.1 / 42.9 | 67.0 / 4.8 |

### 1.4 用詞：可以說「經過」，不能說「從 MB 出發」

| 用詞 | 可否 | 理由 |
|---|---|---|
| 經過 / have neurites in / arborize in | ✅ | 資料量的就是神經突落在哪 |
| innervate | ⚠️ | 領域慣用，但嚴格來說暗示有突觸，骨架量不到 |
| 從 MB 出發 / originate from | ❌ | soma 位置與投射方向都不在這份資料裡 |
| 全部都跨 MB 與 DFP | ❌ | 皆 ≥ 5 % 只有 92.2 %；另有 8.7 % 的第二腦區佔比 < 10 % |
| 同側投射 | ❌ | 前兩名 compartment 在同一半腦者 D1 90.3 %、D2 82.3 %，差距不大，不是 D1 的特徵（見 §5） |

### 1.5 標題候選

- `Sub-dataset D1: Neurons with the MB or DFP as the primary neuropil`
- `Sub-dataset D1: MB-associated neurons`

（挑選動機仍待與合作者確認。）

---

## 2. D2：大、集中於單一腦區、分布全腦

D2 由 stage-1 prescreening 擴充（`candidate_matching.py`：質心距離 ≤ 100 µm、
轉動慣量比 (r21, r31) 距離 ≤ 0.4、rod/disk 主軸一致）。

### 2.1 兩條獨立的形態軸

| 軸 | 描述子 | D1 | D2 | AUC |
|---|---|---|---|---|
| **大小** | Cable length，FC | 1 722 µm | **4 817 µm** | **0.905** |
| | Cable length，EM | 2 318 µm | **6 584 µm** | 0.819 |
| **佔位集中度** | 第二 compartment 佔比，FC（`side_top2`） | **0.325** | 0.162 | 0.884 |

- **大小**：兩個資料庫一致，D2 約為 D1 的 2.8 倍。cable 長度、分支點數、末梢數、佔用體積相互
  Spearman ≥ 0.96，是**同一個軸**；分支數扣除 cable 後殘餘 AUC 僅 0.569。
- **佔位集中度獨立於大小**：依 cable 長度 1:1 配對（±25 %）後 AUC 仍有 0.808。
- `side_top2` = 佔比第二大的 compartment（左右分開計）的佔比，分母為具名 neuropil 總和。
  數值越小代表越集中在單一 compartment。

### 2.2 主腦區分布

D2 的主腦區共 12 種：

| DFP | AL | VLP | DLP | FB | LH | EB | SPP | DMP | CCP | CMP | LOB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 34.0 % | 16.3 % | 12.9 % | 12.2 % | 12.2 % | 4.1 % | 2.7 % | 2.0 % | 1.4 % | 0.7 % | 0.7 % | 0.7 % |

對照 D1 只有 4 種（MB 61.2 %、DFP 36.9 %、EB 與 LH 各 1 顆）。

### 2.3 可引用的量化規則

10-fold × 10 重複分層交叉驗證，門檻在每個訓練 fold 內重新擬合。

| 規則（判為 D2） | 佔比分母 | CV balanced accuracy | precision / recall |
|---|---|---|---|
| cable ≥ 2 000 µm **且** `side_top2` ≤ 0.32 | 具名 neuropil（**不含** other） | **0.881 ± 0.071** | 0.907 / 0.925 |
| cable ≥ 2 000 µm **且** `sidetot_top2` ≤ 0.244 | `volume`（含 other） | 0.861 ± 0.068 | 0.905 / 0.905 |
| 只用 cable ≥ 2 474 µm | — | 0.871 ± 0.066 | — |

⚠️ 兩件要注意的事：

1. **分母一致性**：論文其他地方用「佔總重建量」（含 other）的話，這裡應引用第二條。
2. **佔位集中度對規則的增益很小**：只用 cable 就有 0.871。不含 other 的版本只多 +0.010，
   含 other 的版本反而低 0.010。佔位集中度的價值在於它是**獨立的第二條軸**（§2.1），
   而不是讓分類變得更準。

### 2.4 標題候選

- `Sub-dataset D2: Large, regionally confined neurons across the brain`
- `Sub-dataset D2: Brain-wide neurons with large single-region arbors`

**不建議在標題用 dense**：控制大小後，「D2 整體比 D1 密」目前不成立。海綿效應講的是
D2 **內部**的密度差異（`FINDINGS.md` §4）。另外，組間比較還沒用凸包密度重做過。

---

## 3. 為什麼要改「especially for the olfactory system」

### 3.1 草稿原句

> These projection neurons are important, especially for the olfactory system.

這句話讓讀者以為 D1 是嗅覺 projection neuron（PN）。果蠅嗅覺 PN 的經典路徑是
**AL（觸角葉）→ MB calyx → LH（側角）**，所以檢查這三個腦區。

### 3.2 數據

各腦區佔總重建量（分母含 other）：

| 腦區 | D1 中位 | D1 ≥ 5 % | D2 中位 | D2 ≥ 5 % |
|---|---|---|---|---|
| MB（不含 calyx） | 34.8 % | 97.1 % | 0.7 % | 22.4 % |
| DFP | 31.4 % | 95.1 % | 9.4 % | 59.9 % |
| **AL** | 0.0 % | **0.0 %** | 0.0 % | 16.3 % |
| **Calyx** | 0.0 % | **1.0 %** | 0.0 % | 0.7 % |
| **LH** | 0.0 % | **5.8 %** | 0.1 % | 16.3 % |

### 3.3 結論

1. **D1 沒有嗅覺 PN 的型態。** 103 顆中 AL ≥ 5 % 的有 0 顆、calyx ≥ 5 % 的只有 1 顆。
   D1 的 MB 神經突在 calyx 以外的 MB，另一端在 DFP。
2. **D2 也不以嗅覺 PN 為主。** D2 有 24 顆 AL ≥ 5 %，但**其中 calyx ≥ 5 % 的是 0 顆**，
   比較像侷限在 AL 內的神經。
   - **更正先前的說法**：`GH146-M-000001` 雖然名稱帶有經典 PN driver，實際分布是
     AL 75.3 %、calyx 0.0 %、LH 2.0 %，**不是 PN 的投射型態**。不能拿它當「D2 含嗅覺 PN」的例子。
3. D1「MB（非 calyx）+ DFP」的分布比較符合 MB extrinsic neurons（如 MBON、DAN）的型態，
   但**這份分析判定不了細胞型別**。

### 3.4 替代寫法

「MB 是嗅覺學習與記憶的中樞」是文獻共識，可以引用，但不要暗示 D1 是 PN。

- 原句：*These projection neurons are important, especially for the olfactory system.*
- 建議：*The MB is a well-established center for olfactory learning and memory, which makes
  MB-associated neurons a natural starting point.*

另外建議把 **projection neurons** 從 D1 的描述中拿掉。這個詞在果蠅文獻中幾乎專指嗅覺 PN。

---

## 4. 待確認

| 項目 | 狀態 | 如何確認 |
|---|---|---|
| D1 的挑選動機 | 與合作者討論中 | — |
| D1 的細胞型別（MBON / DAN？） | 未驗證 | 用 D1 的 143 個 hemibrain ID 查 neuPrint 的 cell type（需線上存取，本地沒有這份資料） |
| DFP 的命名對應 | 未驗證 | DFP 是 FlyCircuit 採用的名稱，與現行系統命名（Ito et al. 2014）的對應需確認，審稿人可能會問 |
| 3 顆同屬 D1 / D2 的 `TH-F` 神經 | 已知 | 論文需決定歸屬或加註 |

---

## 5. 本次修正的舊敘述

| 位置 | 舊 | 新 |
|---|---|---|
| `FINDINGS.md` §1、`HANDOVER.md` §1 | 「93.2 % 為同側投射」 | 93.2 % 其實是「主 compartment 在右半腦」的比例，反映的是資料的左右分布。同側性（前兩名 compartment 在同一半腦）是 **90.3 %**，D2 為 82.3 % |
| `FINDINGS.md` §1、`HANDOVER.md` §1 表格 | 以「MB 與 DFP 各 ≥ 5 %」為主指標 | 改以「主腦區為 MB 或 DFP」為主指標 |
| `FINDINGS.md` §2、`HANDOVER.md` §2 | 第二 compartment「佔總體積」 | 規則用的 `side_top2` 分母不含 other；已補上含 other 的版本 |
| 先前對話 | 以 GH146-M-000001 暗示 D2 含嗅覺 PN | 該神經沒有 calyx 分布，不是 PN 型態 |

---

## 重現

```bash
cd analysis_dense_vs_projection
python s06_region_criteria.py   # §0、§1、§3 的腦區數字 → results/region_wording_support.csv
python s07_selection_rule.py    # §2.3 的規則          → results/selection_rule_thresholds.csv
python s04_group_contrast.py    # §2.1 的形態數字      → results/contrast_*.csv
```
