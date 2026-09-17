# 模型訓練與預測流程交接說明

這份文件說明從人類標註資料切分、初始模型訓練、pseudo labeling、pre-train、fine-tune，到使用模型輸出 prediction 結果的流程。

前半段 `SWC -> descriptor -> candidate pair -> standard view` 請先看 [DRAW_PIPELINE_HANDOVER.md](DRAW_PIPELINE_HANDOVER.md)。
線上服務版（使用者上傳單一 SWC，即時回傳相似度 CSV）見 [SERVICE_HANDOVER.md](SERVICE_HANDOVER.md) ——
它重用本文件的模型與 `Model_predict.py` 的前處理，只是把批次流程改成單次查詢。

本文件假設三視圖已經存在於：

- `data/standard_views/FC/*.npz`
- `data/standard_views/EM/*.npz`

每個 `npz` 需要包含 `views`，模型端會讀取 FC/EM 各自的三視圖，padding 到同尺寸後下採樣成 `50x50`，再組成 Siamese CNN 的輸入。

## 1. 流水線總覽

主流程可以分成七段：

1. 用人類專家標註結果產生 train/test cross-validation split。
2. 針對每個 fold 訓練初始 annotator model。
3. 用 `result_analysis.py` 檢查初始模型結果。
4. **從 prescreening 候選池剔除所有被人工標註過的神經**（只要有一邊出現就剔除整筆配對）。
5. 用 10 個初始模型對剩下的未標註 pair 產生 pseudo label prediction。
6. 合併 10 個 fold 的 pseudo label，保留標準差低的 high-confidence pair。
7. 用 pseudo label pre-train 一個模型，再用人類標註資料 fine-tune。

第 4 步是後來補上的。annotator 的十個 fold 模型合起來看過**全部**專家標註，若候選池裡留著
「某一側被標註過」的配對，那份身分記憶會經由 pseudo label 流進 pre-train，使預訓練階段
不再是 fold-clean。只排除專家配對本身不夠——記憶正是經由「同一顆神經的其他候選」外溢的。

## 2. 關鍵腳本

- [make_cross_val_set.py](make_cross_val_set.py)：產生 cross-validation train/test split。
- [Data_process_Train.py](Data_process_Train.py)：讀取 split 和三視圖，訓練 annotator 或 fine-tune model。
- [Data_process_Train.sh](Data_process_Train.sh)：批次跑 10 個 fold。
- [result_analysis.py](result_analysis.py)：分析模型結果、畫 loss curve、violin plot、ROC、confusion matrix。
- [make_pseudo_candidates.py](make_pseudo_candidates.py)：從 prescreening 候選池剔除所有被人工標註過的神經，產生 pseudo labeling 專用的乾淨候選名單。
- [Model_predict.py](Model_predict.py)：用指定 fold 模型對未標註 pair 做 prediction。
- [Model_predict.sh](Model_predict.sh)：批次用 10 個 fold 模型做 prediction。
- [merge_pseudo_label.py](merge_pseudo_label.py)：合併 10 個 fold 的 prediction，篩選 high-confidence pseudo label。
- [data_process_preTrain.py](data_process_preTrain.py)：用 pseudo label 從頭 pre-train。
- [merge_Model_predict.py](merge_Model_predict.py)：將 prediction CSV 合併後，按 `fc_id` 或 `em_id` 取 top rank。
- [data_process_rank.py](data_process_rank.py)：**排序微調**（§14）。用外部型別標籤的
  InfoNCE + 專家標註的 BCE 聯合重訓，針對全庫檢索而非成對二分類。

## 3. 資料與目錄約定

### 3.1 人類標註資料

`make_cross_val_set.py` 預設使用 soft label：

- `labeled_info/D1_conf.csv`
- `labeled_info/D2_conf.csv`
- `labeled_info/D3_conf.csv`
- `labeled_info/D4_conf.csv`
- `labeled_info/D5_conf.csv`
- `labeled_info/D6_conf.csv`

每份標註表至少需要包含：

- `fc_id`
- `em_id`
- `label`

### 3.2 Cross-validation split

預設輸出在：

- `train_test_split/train_split_{fold}_D1-D6.csv`
- `train_test_split/test_split_{fold}_D1-D6.csv`

其中 `fold` 為 `0..9`。

### 3.3 模型與結果輸出

常用輸出目錄：

- 初始 annotator / fine-tune 權重：依 `Data_process_Train.py` 的 `Config.save_model_dir`
- pre-train 權重：`PreTrain_Model/*.weights.h5`
- 測試與訓練歷史：`result/*.pkl`、`result/test_label_*.csv`
- pseudo label prediction：`result/unlabel_data_predict/*.csv`
- 圖表：`Figure/*`

## 4. Step 1：產生 train/test split

使用：

```bash
python3 make_cross_val_set.py
```

目前預設設定：

- `mode = 1`：使用 `KFold`
- `cross_validation_num = 10`
- `used_label = "soft_label"`
- `seed = 7`
- `out_dir = "./train_test_split"`

輸出會是 10 組 `train_split` / `test_split`。後續訓練腳本會根據 fold number 自動讀取對應 split。

## 5. Step 2：訓練初始 annotator model

使用：

```bash
bash Data_process_Train.sh
```

`Data_process_Train.sh` 會依序執行：

```bash
for i in {0..9}; do
  python3 Data_process_Train.py "$i"
done
```

`Data_process_Train.py` 的主要流程：

1. 讀取 `train_test_split/train_split_{fold}_D1-D6.csv` 和 `test_split_{fold}_D1-D6.csv`。
2. 從 `data/standard_views/FC` 和 `data/standard_views/EM` 讀取對應三視圖。
3. 將每一對 FC/EM padding 到同尺寸，再下採樣到 `50x50`。
4. 從 train split 再切出 validation set，預設 `val_ratio = 0.15`。
5. 使用 `MVCNN_Siamese` 建模。
6. 對 training data 做 augmentation：FC/EM swap、4 種旋轉、左右翻轉。
7. 以 `BinaryFocalCrossentropy` 和 `AdamW` 訓練。
8. 保存最佳 validation loss 的 `.weights.h5`。
9. 重新載入 best weights，在 validation / test set 上產生 report 和 test prediction CSV。

注意：目前 `Data_process_Train.py` 的 `Config` 已經切到 fine-tune 設定：

- `use_pretrain_model = True`
- `pretrain_model = "./PreTrain_Model/pre_train_model_by_EMxFC_high_confidence.weights.h5"`
- `save_model_dir = "./FineTune_Model"`
- `model_name = "FineTune_miniLR_e7"`
- `initial_lr = 1e-7`
- `train_epochs = 600`

如果要訓練最初的 annotator model，需要把 `Config` 中 annotator 相關設定切回來，也就是不載入 pretrain weights，並調整 `save_model_dir` / `model_name` / learning rate / epoch。

## 6. Step 3：檢查訓練結果

使用：

```bash
python3 result_analysis.py
```

這支腳本是分析用腳本，通常會根據當次測試目的手動調整。常改的設定包括：

- `model_name`
- `test_mode`
- `cross_num`
- `selected_test_set`
- `label_csv_name`
- `nblast_path`

目前它可以做：

- cross-validation loss curve
- selected test set 篩選
- model score / NBLAST score 分布比較
- violin plot
- ROC curve
- confusion matrix
- threshold / F1 探索

輸出圖通常保存到 `Figure/`。

## 7. Step 4：用初始模型產生 pseudo label prediction

### 7.0 前置（必要）：剔除被人工標註過的神經

```bash
python3 make_pseudo_candidates.py
```

輸入 `data/pairs_label/EMxFC_6KK_last.csv`（prescreening 候選），依 `D1-D6_total_conf.csv`
與 `labeled_info/D*_conf.csv` 的聯集，把**任一側**為人工標註神經的配對整筆剔除，輸出
`data/pairs_label/EMxFC_6KK_last_noexpert.csv`。

目前候選池的實測：6,482,684 組中有 544,868 組（8.40 %）至少一側是被標註過的神經
（393 / 453 顆專家 FC、454 / 587 顆專家 EM 出現在池中），剔除後剩 5,937,816 組（91.60 %）。

⚠️ **只在配對層級排除是不夠的。** `EMxFC_6KK_last.csv` 是由 `preTrain_label/Annotator_D1-D6_*.csv`
拼出來後**手動**剔除人工標註配對而成的，所以它一組專家配對都沒有；但這個動作當初沒有寫進
pipeline，而且它只擋掉專家配對本身，擋不住「同一顆神經的其他候選」——annotator 的身分記憶正是
經由後者外溢。本步驟把排除補進流程，並從配對層級提升到神經層級。

### 7.1 十折預測

跑之前把 `Model_predict.py` 的 `Config.pairs_csv` 指向
`./data/pairs_label/EMxFC_6KK_last_noexpert.csv`，`model_dir` / `model_prefix` / `out_dir`
切回 annotator 那一組。

使用：

```bash
bash Model_predict.sh
```

`Model_predict.sh` 會依序執行：

```bash
for i in {0..9}
do
    python3 Model_predict.py $i
done
```

`Model_predict.py` 的預設輸入：

- 模型目錄：`./Annotator_Model`
- 模型前綴：`Annotator_D1-D6_`
- 未標註 pair 名單：`./data/pairs_label/EMxFC_6KK_last_noexpert.csv`（Step 7.0 的輸出；
  **不要**直接用未過濾的 `EMxFC_6KK_last.csv`）
- FC 三視圖：`./data/standard_views/FC`
- EM 三視圖：`./data/standard_views/EM`

未標註 pair CSV 需要包含：

- `fc_id`
- `em_id`

可選欄位如 `score`、`rank` 可以存在，但目前 prediction 輸出不會保留它們。

`Model_predict.py` 會用 chunk 方式讀取 pairs，預設每個 chunk `80000` rows。每個 fold 的輸出保存到：

- `result/unlabel_data_predict/Annotator_D1-D6_{fold}_chunk{chunk_idx}.csv`

輸出欄位：

- `fc_id`
- `em_id`
- `model_predict`

缺失或讀取失敗的三視圖會追加到：

- `result/unlabel_data_predict/missing_Annotator_D1-D6_{fold}.csv`

## 8. Step 5：合併 pseudo label

使用：

```bash
python3 merge_pseudo_label.py
```

`merge_pseudo_label.py` 會：

1. 讀取 `result/unlabel_data_predict` 下每個 fold 的 prediction CSV。
2. 根據 `fc_id`、`em_id` 橫向合併 10 個 fold 的 `model_predict`。
3. 計算每個 pair 的 `predict_mean` 和 `predict_std`。
4. 保留 `predict_std < 0.05` 的 high-confidence pair。
5. 用 `predict_mean >= 0.5` 作 positive，其餘作 negative。
6. 因為 negative 通常遠多於 positive，會隨機抽樣 negative，使正負樣本數量平衡。
7. 將 `predict_mean` 改名為 `label`。
8. **把關**：呼叫 `make_pseudo_candidates.expert_neurons()`，確認輸出中沒有任何一組配對的
   某一側是被人工標註過的神經；只要有就 `raise`，不會產出檔案。

輸出：

- `data/pairs_label/EMxFC_all_high_confidence.csv`

這份 CSV 是下一步 pre-train 的輸入。

⚠️ 現存的這份檔案（178,278 組）是**舊流程**的產物，其中 12,876 組（7.22 %）至少一側是被
標註過的神經，新的把關會擋下來。重跑前請先備份，因為目前所有已完成的模型與論文數字都是
用它 pre-train 出來的；覆蓋掉就回不去了。

## 9. Step 6：用 pseudo label pre-train

使用：

```bash
python3 data_process_preTrain.py
```

或直接跑：

```bash
bash data_process_preTrain_fineTune.sh
```

`data_process_preTrain.py` 的預設輸入：

- pseudo label：`./data/pairs_label/EMxFC_all_high_confidence.csv`
- FC 三視圖：`./data/standard_views/FC`
- EM 三視圖：`./data/standard_views/EM`

主要流程：

1. 讀取 pseudo label CSV。
2. 去除無效 label 和重複 pair。
3. 載入三視圖並轉成 `50x50` pair tensor。
4. 切 train / validation，預設 `val_ratio = 0.1`。
5. 從頭訓練 `MVCNN_Siamese`。
6. 保存最佳 validation loss 的 weights。
7. 保存 pre-train history、validation report 和 validation prediction CSV。

目前 pre-train augmentation 只保留 FC/EM swap，不做 rotation / flip。

預設輸出：

- `PreTrain_Model/pre_train_model_by_EMxFC_180K.weights.h5`
- `result/PreTrain_History_pre_train_model_by_EMxFC_180K.pkl`
- `result/PreTrain_Val_Result_pre_train_model_by_EMxFC_180K.pkl`
- `result/pretrain_val_pred_pre_train_model_by_EMxFC_180K.csv`

## 10. Step 7：用人類標註資料 fine-tune

使用：

```bash
for i in {0..9}
do
    python3 Data_process_Train.py "$i"
done
```

或：

```bash
bash data_process_preTrain_fineTune.sh
```

目前建議直接使用 `Data_process_Train.py` 做 fine-tune。它的 `Config` 已經支援：

- `use_pretrain_model = True`
- `pretrain_model = "...weights.h5"`
- `save_model_dir = "./FineTune_Model"`
- `model_name = "FineTune..."`

流程如下：

1. 讀取 cross-validation split。
2. 載入 `data/standard_views/FC` 和 `data/standard_views/EM`。
3. 建立 `MVCNN_Siamese`。
4. 載入 pre-train 權重。
5. 用人類標註資料 fine-tune。
6. 保存每個 fold 的 best weights、history、test report、test prediction CSV。

目前 `Data_process_Train.py` 預設 pre-train 權重是：

- `PreTrain_Model/pre_train_model_by_EMxFC_high_confidence.weights.h5`

這裡需要和 `data_process_preTrain.py` 的輸出名稱對齊；如果 pre-train 實際輸出是 `pre_train_model_by_EMxFC_180K.weights.h5` 或其他名稱，要同步修改 `Config.pretrain_model`。

## 11. 使用模型 prediction 結果

如果目標是把多個 prediction CSV 合併後，對每個 `fc_id` 或 `em_id` 取排名，可以使用：

```bash
python3 merge_Model_predict.py
```

目前預設：

- 讀取目錄：`./result/unlabel_data_predict/`
- 排名依據：`find_from = 'em_id'`
- 每個 group 保留前 `rank_n = 20`
- 輸出：`result/unlabel_data_predict/2024-04-01_EMxFC_rk20.csv`

如果要從 FC 找 EM，將 `find_from` 改為 `fc_id`；如果要從 EM 找 FC，使用 `em_id`。

## 12. 建議的完整運行順序

第一次重跑整個模型流程時，建議按下面順序：

```bash
python3 make_cross_val_set.py
bash Data_process_Train.sh
python3 result_analysis.py
python3 make_pseudo_candidates.py   # 剔除被標註過的神經，Step 7.0
bash Model_predict.sh               # pairs_csv 要指向 *_noexpert.csv
python3 merge_pseudo_label.py
python3 data_process_preTrain.py
bash Data_process_Train.sh
python3 result_analysis.py
```

## 13. 需要留意的地方

### 13.0 ⛔ `MVCNN_Siamese` 只讀到三視圖中的第 3 張（2026-09-18 發現）

`model.MVCNN_Siamese` 的視角切片寫成
`[Lambda(lambda x: tf.expand_dims(x[..., i], -1)) for i in range(3)]`。
這是 Python 閉包延遲綁定：三個 lambda 共用同一個 `i`，模型執行時 `i` 已經是 2，
**三個「視角」全部讀到第 3 張圖（channel 2），另外兩張對輸出完全沒有影響。**

實測（`model.py` 內的測試方式可重現）：
- 三個 channel 各填 1、2、3，三個切片都取到 3。改 channel 0、1，模型輸出一位數都不變；改 channel 2 輸出才變。
- **現有所有權重都是在這個狀態下訓練的**：Annotator fold 0、fold 5、FineTune fold 0 的
  6 個逐視角 BN，moving_mean / moving_var 在三個視角之間**逐位元相同**
  （gamma / beta 不同，是各視角後面獨立的 Dropout 造成的，符合預期）。
- 所以舊模型本身是一致的，**線上服務與各預測腳本沒有訓練與推論不一致的問題**，
  只是從來沒用到另外 2/3 的視角資訊。
- 反過來，**不能拿舊權重去配修正後的切片**：Annotator fold 0 守門 AUC 0.905 → 0.836、
  fold 5 0.900 → 0.857、FineTune fold 0 0.962 → 0.875。

處理方式：
- `MVCNN_Siamese` **暫時保留不改**（只加註解），現有權重、`nrn_service`、`Model_predict.py`、
  `tools/build_confident_list.py` 等照舊運作。
- 新增 `MVCNN_Siamese_3View`：修正切片，建層順序與舊函數相同；另有 `trunk_norm="gn"` 選項。
  `MVCNN_Siamese_Advanced` 有同樣的錯誤，但專案沒有使用，只加了註解。
- 舊的 annotator 權重已備份到 `Annotator_Model_backup_singleview_20260918/`（原檔也保留）。
- 重訓入口：`python3 train_annotator_3view.py <fold>`，沿用 `Data_process_Train.main`，
  只改 `model_builder="MVCNN_Siamese_3View"`，存成**新檔名**
  `Annotator_Model/Annotator3v_D1-D6_{fold}.weights.h5`，不覆蓋舊權重。
  `Data_process_Train.Config` 新增 `model_builder` 欄位，預設 `"MVCNN_Siamese"` 維持舊行為。
- **PreTrain / FineTune 模型同樣受影響**，尚未重訓。新 annotator 驗證過之前，
  不要把服務切到新權重。

### 13.1 Config 目前主要靠手動改

多數腳本的路徑、模型名稱、learning rate、epoch 都寫在 `Config` 或腳本頂部變數中。交接時務必先確認：

- 要訓練 annotator 還是 fine-tune。
- `model_name` 是否和 `result_analysis.py` 讀取的名稱一致。
- pre-train 權重檔名是否真的存在。
- `Model_predict.py` 的 `model_dir`、`model_prefix` 是否和實際初始模型輸出一致。

### 13.2 建議統一訓練入口

目前 annotator 訓練和 fine-tune 都走 `Data_process_Train.py`。後續如果要把切換方式做得更乾淨，可以替 `Data_process_Train.py` 加命令列參數，例如：

```bash
python3 Data_process_Train.py 0 --stage annotator
python3 Data_process_Train.py 0 --stage finetune --pretrain PreTrain_Model/xxx.weights.h5
```

### 13.3 三視圖讀取與 preprocessing 重複很多

以下函式在多個檔案中幾乎重複：

- `_to_uint8_views`
- `_ensure_3hw_views`
- `_pad_to_same_size`
- `_resize_to_50`
- `_load_views_from_npz`
- `make_numpy_from_standard_views`

建議抽到共用模組，例如 `view_pair_dataset.py`，讓 train、pre-train、fine-tune、predict 共用同一套 preprocessing。這是最值得優先做的整理。

### 13.4 pseudo label 篩選閾值目前是固定值

`merge_pseudo_label.py` 目前固定：

- `predict_std < 0.05`
- `predict_mean >= 0.5` 作 positive
- negative downsample 到 positive 數量

如果資料分布改變，建議把這些值改成 `Config` 或命令列參數，並在輸出檔名中記錄閾值，例如 `EMxFC_high_conf_std005_balanced.csv`。

### 13.5 shell 腳本目前沒有錯誤中止

`Data_process_Train.sh`、`Model_predict.sh` 建議也加上 `set -euo pipefail`。如果中間某個 fold 失敗，後面可能繼續跑，最後不一定容易發現缺少某個 fold。`data_process_preTrain_fineTune.sh` 目前已經加上這個設定。

建議加上：

```bash
set -euo pipefail
```

並把 stdout/stderr 保存到帶日期或 stage 名稱的 log。

### 13.6 建議保存 not-found 報告

訓練腳本目前會 print `not_found_train` / `not_found_test` 數量，但沒有把 missing details 保存成 CSV。建議和 `Model_predict.py` 一樣，保存 missing report，方便排查三視圖缺失或 ID 對不上。

## 14. Step 8：排序微調（`data_process_rank.py`）

§1–§13 的流程產出的模型是為「**這一對像不像**」訓練的二分類器。
但線上服務要的是「**從 700–3 000 個候選裡把正解排第一**」。
`analysis_model_results/FINDINGS.md` §10 證明這兩件事在本任務上會分岔——
`FineTune_miniLR` 在十折測試集上贏 annotator（AUC 0.978 vs 0.929，10/10 折），
在全庫檢索上卻輸（rank-1 22.0 % vs 31.0 %）。

這一步就是針對檢索任務重訓。**它是 §10 的延伸，不取代 §1–§13**——
專家標註與 annotator 模型仍是它的輸入。

### 14.1 架構的變化（很小）

模型結構仍是 `model.MVCNN_Siamese`，**沒有改動 `model.py`**。差異只有三處：

| | 原本（`Data_process_Train.py` / `data_process_fineTune.py`） | 排序微調 |
|---|---|---|
| 輸出層 | `Dense(1, activation="sigmoid")` | `Dense(1)`，**輸出 logit**；sigmoid 只在推論與 BCE 時套 |
| head 的正規化 | `BatchNormalization`（`momentum=0.99`） | **`LayerNormalization`** |
| head 初始權重 | 沿用載入的權重 | **預設隨機初始化**（`--head-init existing` 可沿用） |

兩個修改都有實測依據：

- **logit**：排序損失的有趣區間在 0.88–0.99，套在 sigmoid 上等於跟被壓扁的梯度打架。
  （拆分類頭不影響既有權重載入——實測 `load_weights()` 舊檔成功且輸出**逐位元相同**，
  TF 2.19.1 / Keras 3.12.0。）
- **LayerNorm 取代 BatchNorm**：用於檢索的打分函數，對同一對神經的輸出不該取決於
  「同批次裡還有誰」，而 BN 在訓練時正是批次相依的。這個落差咬過三次：
  (1) `momentum=0.99` 讓 moving 統計嚴重滯後，實測池內 AUC 掉到 **0.33**；
  (2) 正例/負例/專家三次 forward 各自更新 moving 統計（穩態 29.9/33.2/**36.9** %），
  而檢索只打 pool、守門只打專家，兩個指標都在錯配的正規化下量測；
  (3) 同一個 InfoNCE 比較裡正例與負例用不同批次的統計做正規化。
  第 (3) 點事後無法補救（重校實測反而更差），只能換掉正規化層。
  **實測效果**：探針組態下 rank-1 從 53.46 → **66.72**，超過階段 2 探針的 63.20。
  完整的回歸測試階梯見 `RANKING_FINETUNE_PLAN.md` §階段 3 結果。
  ⚠️ LN 移除了 BN 批次噪音的隱性正則，訓練損失收斂快一個數量級，要留意過擬合。

**可訓練範圍**是個開放問題，兩種都能跑：

| | 可訓練參數 | 說明 |
|---|---|---|
| 預設（凍結 trunk） | 4 719 617 | 只訓練分類頭 |
| `--tune-trunk` | 4 785 377 | 連 conv1-4 + BN 一起訓練 |

⚠️ 參數分布是反直覺的：**trunk 只佔 1.4 %（66 528），head 佔 98.6 %（4 720 129）**。
「凍結 trunk」凍的是那 1.4 %。預設凍結是因為階段 2 證明卷積特徵**夠用**，
但那沒有證明它**最好**——那個 trunk 當初只用 1 097 對訓練，
而現在的監督量多了約 170 倍（見 §14.2），值得兩種都跑再定。

### 14.2 訓練資料：兩個來源，同步聯合訓練

**(a) 排序項（InfoNCE）——外部型別標籤**

來自 `analysis_external_validation/`：FC 端用 Virtual Fly Brain 的 FBbt 策展型別，
EM 端用 neuPrint `hemibrain:v1.2.1` 的 `type`。兩者都**獨立於本專案的 CNN**。

- 母體：1 276 顆 winnable FC（池中確有同型 EM），依族群分層切一半 →
  **訓練 638 顆 / 評估 638 顆**
- 候選池與分數：`analysis_hubness_debias/scores/pool_scores_annotator.parquet`
  （530 萬對，`candidate_matching.run_matching` 的完整池）
- 每 epoch 重抽：每顆 FC 取 4 個正例當錨點 → 約 **2 379 個錨點**
- 每個錨點的負例 = **24 個池內型別不符** + **8 個 cross**
  → 每 epoch 約 **78 500 對**

⚠️ **cross 負例是最關鍵的設計**。若負例只從同一顆 FC 的池抽，
「把某顆 EM 全域推高」對損失是免費的午餐——實測階段 2 的某個變體，
其 rank-1 有 **39.4 %** 可被「完全忽略 FC、只用每顆 EM 平均分」的虛無模型重現。
cross 負例 = 這顆 FC 池內、是**別顆** FC 的正例、但對本 FC 型別不符的 EM
（每顆 FC 中位有 673 個可選）。

**(b) 二元項（一般 BCE）——專家標註**

`train_test_split/train_split_{fold}_D1-D6.csv`，fold 0 為 **1 097 對**。
用 `data_process_fineTune.make_numpy_from_standard_views` 載入，與既有流程一致。

**不能拿掉。** 排序損失只看分數差、對全域平移無感，模型可把分數壓進 `[0.30, 0.52]`，
排序完全正確但 0.5 門檻就廢了。專家標註同時也是**個體層級**的
（每顆 FC 中位只給 1 個正例），比型別層級嚴格。

⚠️ **不用 focal loss。** `BinaryFocalCrossentropy(gamma=2.0)` 會**降低**容易樣本的權重，
而這批負例是「模型極度自信但錯」的（中位分數 0.878），focal 會**放大**其梯度。

**(c) 增強**：**每個樣本**各自抽一組 `(rot, flip)`，對配對兩側同步施加。
`PairEncoder` 為每顆神經快取 8 個變體（4 個旋轉 × 翻轉與否），抽樣時查表。
⚠️ 早期版本是「每 epoch 抽一組、整個 epoch 共用」，等於完全沒有樣本級多樣性。
（不做 FC/EM swap：這裡每一對只 forward 一次，沒有探針那種兩次 forward 的對稱性問題。）

### 14.3 執行

```bash
python3 data_process_rank.py                        # fold 0，凍結 trunk
python3 data_process_rank.py --tune-trunk           # 連卷積層一起訓練
python3 data_process_rank.py --head-init existing   # head 沿用既有權重
python3 data_process_rank.py --fold 3 --epochs 80
python3 data_process_rank.py --w-bce 0              # 消融：關掉專家二元項
python3 data_process_rank.py --epochs 35 --pick-epoch 35   # 強制評估指定 epoch
python3 data_process_rank.py --trunk-init scratch   # 完全從頭，不繼承 annotator
# 診斷：載入既有 head、跳過訓練、把 BN moving 統計重校到指定分布後重評
python3 data_process_rank.py --epochs 0 --dev-ratio 0 \
        --load-head RankTune_Model/<檔名>.weights.h5 --recalib 50 --recalib-src pool
```

```bash
# EM 型別 hold-out：這些型別的 EM 完全不進訓練（決定性的泛化測試）
python3 data_process_rank.py --holdout-emtypes "LC12,KCab-s,DL2d_adPN"
# 控制組：只在報表多列這些型別的分層，不影響訓練
python3 data_process_rank.py --report-emtypes "LC12,KCab-s,DL2d_adPN"
# 正例只抽一次、整個訓練固定（探針的作法，實測 rank-1 +6.6 pp、ALPN +42 pp）
python3 data_process_rank.py --fixed-anchors
```

⚠️ **FC 層級的切分不足以測泛化**：實測每顆 eval FC 的正例中位 100 % 也是訓練正例。
`--holdout-emtypes` 把整個型別拿掉才問得到「沒見過的神經型別能不能排對」。
注意被 hold out 的 FC 其正例就是那些 EM，所以 FC 型別與 EM 型別必然一起被拿掉，
結論只能說到「無法泛化到訓練時缺席的型別」。

`--load-head` + `--epochs 0` 跳過訓練只做評估；`--recalib N --recalib-src {pool,expert}`
用 N×128 筆該分布的配對重跑 `training=True` forward，把 BN moving 統計收斂過去。
換成 LayerNorm 之後這組旋鈕對新模型已無作用（LN 沒有 moving 統計），
保留是為了診斷舊的 BN checkpoint。

`--trunk-init scratch` 連 trunk 都隨機初始化，整個打分器不繼承任何 annotator 的東西，
用途是把「先驗是不是繼承來的」問到底。它會**自動啟用 `--tune-trunk`**
（凍結一個隨機特徵抽取器只是隨機投影），且不能配 `--head-init existing`
（head 權重是針對 annotator 的特徵訓練的，接到隨機 trunk 上沒有意義，程式會報錯）。
⚠️ **目前不能用於真正的從頭訓練**：`--tune-trunk` 會把 trunk 的 BN 鎖在推論模式，從頭訓練時它們停在初始統計量、從未正規化。要從頭訓練，得先改掉 trunk 的正規化處理（改用 GroupNorm/LayerNorm，或把正例、負例、專家配對合成一次 forward）。

`--pick-epoch N` 繞過「挑 dev 最佳」的預設，直接評估第 N 個 epoch 的權重。
訓練是**確定性的**（同 seed 下逐 epoch 損失可逐字重現），
所以 `--epochs N --pick-epoch N` 能精確重現任一 epoch 的狀態拿去做完整檢索評估。
會這樣設計，是因為 dev 只有 133 顆 FC、KC 又佔多數，
單一全域 checkpoint 未必對每個族群都是最佳時點——這個假設後來被否證（見 PLAN），
但這個旋鈕本身是診斷後期 checkpoint 的唯一手段。

⚠️ **`--fold` 會連帶決定 trunk 的來源**：`Annotator_D1-D6_{fold}.weights.h5`。
`Annotator_D1-D6_{fold}` 是在 `train_split_{fold}` 上訓練的，沒看過
`test_split_{fold}`；若固定用 fold 0 的權重跑其他 fold，專家測試集那道守門會失效。

### 14.4 輸出與守門指標

- `RankTune_Model/{stem}_head.weights.h5`（`--tune-trunk` 時另存 `_trunk.weights.h5`）
- `result/retrieval_{stem}.csv` —— 型別層級檢索，**分族群**報
- `result/test_label_{stem}.csv` —— 專家測試集，與原模型並排

`{stem}` 帶上 `head_init` / `frozen|tuned` / fold，非預設的 `--w-bce` 會加 `_wbce{N}`、
`--pick-epoch` 會加 `_ep{N}`，不同設定不會互相覆蓋。
⚠️ 早期版本沒有後兩段，消融跑會直接蓋掉正式權重。

**判讀的三條規則**（都是踩過坑得出來的）：

1. **一定要看 `em_prior_rank1_pct`。** 那是本任務的虛無模型——完全忽略 FC、
   只用每顆 EM 的平均分排序。某個變體若這欄偏高，它的增益就不可信，
   不管 rank-1 多漂亮。階段 2 就是靠它抓到一個看似 +32 pp 的假增益。
2. **主指標用 MRR 與「最佳正解的 rank 中位」，不要用池內 AUC。**
   每顆 FC 的正例數差兩個數量級（KC 中位 111、ALPN 只有 1），
   AUC 對全部正例取平均、又看不到未定型候選，跨族群不可比——
   實測 ALPN 基線 AUC 0.997 但 rank-1 只有 5.8 %，兩者甚至反向。
3. **rank-1 的執行間變異約 ±10 pp**，單次結果不可過度解讀。

**守門**：專家測試集 AUC 與**同一折**的 annotator 在同一批測試配對上做配對 bootstrap。
⚠️ 原寫「不得低於 annotator 的 0.9293」是錯的：0.9293 是 annotator **十折合併**的 AUC，annotator 自己在 fold 0 只有 0.9050。fold 0 測試集只有 122 對，修好 bug 的組態（0.856–0.865）與 0.9050 的差距 CI 都包含 0，**不顯著**。詳見 `RANKING_FINETUNE_PLAN.md` §階段 3 結果的更正。

⚠️ **兩個檢索診斷指標各有限制，不可單獨採信**（2026-09-16 量到）：

- **`em_prior_*` 被評估集的同質性污染。** 先驗虛無模型對每顆 FC 幾乎都推同一顆 EM
  （實測 319 顆 FC 的 top-1 只用到 30 顆不同的 EM，一顆 KCab-m 獨佔 208 顆），
  所以數值取決於「那顆居首的 EM 其型別在評估集裡多常見」。KC 佔評估集 66 %
  且型別層級高度同質，因此 KC 先驗在三輪不同的模型上都固定是 49.52 %——
  那是評估集的結構，不是模型性質。它混淆了「模型 hubness」與「評估集同質性」。
  報表現在同時給 `em_prior_MRR` 與 `em_prior_p@5_pct`，看深度比看 rank-1 單點可靠。
- **per-EM 置中（`analysis_hubness_debias/center_diag.py`）會給出虛假的安心。**
  它只移除「與 FC 無關的常數偏置」，移除不了「這顆 FC 對應那顆特定 EM」
  這種 FC-conditional 的關聯。實測某個模型置中後 rank-1 仍是基線的 2.6 倍，
  但同一個模型在 hold-out 層上是 0.00。**置中不是泛化測試。**

→ **真正的泛化判準是 `--holdout-emtypes`。**

⚠️ **守門基線曾經是錯的（已修）。** 原本用訓練中的 `net` 當「原模型」對照，
但 `net` 與 `trunk` 共用圖層：`--tune-trunk` 時 trunk 已被訓練過（卻還掛著原 head），
`--trunk-init scratch` 時它根本沒載入過 annotator 權重——實測誤報 AUC **0.4627**，
照字面讀會變成「RankTune 0.7920 大勝原模型 0.4627」，其實是輸給真正的 0.9050。
現在每次都從 `cfg.base_weights` 重建乾淨的參照模型，並在輸出標上權重檔名。
**凍結 trunk 的跑法不受影響**，先前那些數字有效。

### 14.5 fold 0 的實測結果（2026-09-15）⏸️ **待重測**

> ⏸️ **本節數字全部待重測。** 增強是逐對施加的，一個錨點的正例與它的 32 個負例
> 各自抽到不同的 `(rot, flip)` 就被拿來比大小。實測 **45.9 % 的時候
> 「哪個負例最難」是朝向運氣決定的**（最難與次難差距 0.265 < 噪音 0.449），
> 梯度有近一半時間在追朝向而非形態。
> 完整的量測、修法、受影響的結論清單與重跑順序見
> `RANKING_FINETUNE_PLAN.md` §階段 3 結果開頭的方框。
> 尤其：**「專家 BCE 項不能拿掉」與「訓練更久沒有用」目前未證實。**


**目前狀態：尚未進入生產，守門未過，全折未跑。**
完整的表、消融與撤回聲明見 `RANKING_FINETUNE_PLAN.md` §階段 3 結果。摘要：

| | 全部 MRR | 全部 rank-1 | p@5 | em_prior | 守門 AUC |
|---|---|---|---|---|---|
| 基線（annotator head） | 0.464 | 31.7 [26.9, 36.6] | **66.8** | 3.9 % | **0.9050** |
| RankTune scratch e10 | **0.489** | 34.6 [29.7, 39.8] | **70.2** | 11.4 % | 0.8122 |
| RankTune existing e20 | 0.458 | **34.8** [29.8, 39.8] | 59.3 | **20.9 %** | 0.8561 |

- KC 有真實增益（CI 幾乎不重疊），但 **LC 明確退步**（45.5 → 29.0），
  整體 CI 大幅重疊，沒有可宣稱的整體增益。
- **守門沒過**：0.9050 → 0.8122，低於 annotator 的 0.9293。
- `--w-bce 0` 的消融證明**專家二元項不能拿掉**：拿掉後全面低於基線，
  守門 AUC 0.5311 等於擲骰子。那 1 097 對只佔訓練對數的 1.4 %，
  卻是唯一的個體級監督。
- `--pick-epoch 35` 的消融證明**訓練更久沒有用**：除 ALPN 外全面差於 e10，
  守門 AUC 幾乎不動。
- `--head-init existing` 守門最好（0.8561）、LC 也保住（43.3 vs 基線 45.5），
  但 **EM 先驗退化到 20.9 %**（全場最高），扣掉先驗後主指標反而低於基線。
  ⚠️ cross 負例**沒有**堵住這條退化路徑——若你在別處讀到相反的說法，那是錯的。
- `--trunk-init scratch`（完全從頭，不繼承任何 annotator 權重）**全面輸給基線**
  （MRR 0.350 vs 0.464、p@5 46.0 vs 66.8）。但 dev 最佳只有 0.369
  （annotator trunk 版本 0.635），760 顆 FC 端到端餵不飽，
  這個否定有強混淆，不能直接推論成「方法有根本問題」。
  ⚠️ **2026-09-18 補充**：這輪還有另一個障礙。`--tune-trunk` 會把 trunk 的 6 個 BN 鎖在推論模式；從頭訓練時它們的 moving 統計量停在初始值（均值 0、變異數 1），**等於從未做過正規化**，而 annotator 的 BN 是正常訓練的。加上這輪早於三個 bug 的修正，「資料不足」這個解讀不成立，這輪結果不能拿來判斷從頭訓練的能力。
- **六個組態沒有一個通過守門**（門檻 0.9293），主指標上也沒有一個乾淨勝過基線。

**EM 先驗的劑量反應**（繼承愈多 annotator，先驗愈高；但排序微調本身就會增加它）：

| 繼承多少 annotator | em_prior |
|---|---|
| 基線（沒做排序微調） | 3.9 % |
| 完全從頭 | 7.3 % |
| 只繼承 trunk | 11.4 % |
| 繼承 trunk + head | 20.9 % |

先驗**不是**記住專家標註過的 EM（`analysis_hubness_debias/expert_em_prior.py` 已否證：
那些 EM 在全池反而被壓低，富集倍數 0.2x）。目前的機制假設是
「型別層級標籤無法懲罰通用好度」，驗證方式與死結見
`RANKING_FINETUNE_PLAN.md` §機制假設。

⚠️ **不要引用切分改為隨 fold 轉之前的 fold 0 數字**（ALPN 46.4 %、KC 先驗 41.7 %），
它們在新切分下沒有重現，是舊切分的產物。理由見 PLAN 的撤回段。

### 14.6 最終結論（2026-09-16）：**no-go**，型別泛化失敗

三個訓練 bug 都修掉之後（見 14.1 與下方），檢索數字大幅提升——
生產組態全體 rank-1 31.72 → **55.53**、MRR 0.4638 → **0.6258**、
守門 0.9050 → **0.8585**（配對 bootstrap 差距 +0.047 [−0.008, +0.105]，不顯著），EM 先驗也壓在 8.90 %（基線 3.87 %）。

**但 `--holdout-emtypes` 揭露這些增益不能外推**（⚠️ n 只有 27–57，
CI 很寬——三層裡只有 **KCab-s** 在兩種組態下都與基線 CI 不重疊，
LC12 只在樣本較大的 638 eval 那輪顯著，DL2d_adPN 兩輪都無顯著差異）：

| 層 | 基線 MRR | RankTune | p@5 |
|---|---|---|---|
| 全部 (383) | 0.4638 | **0.6258** | 66.84 → 68.15 |
| holdout:LC12 (32) | 0.3441 | **0.1678** | 56.25 → 25.00 |
| holdout:KCab-s (57) | 0.5097 | **0.1009** | 73.68 → **7.02** |
| holdout:DL2d_adPN (27) | 0.1898 | **0.1460** | 44.44 → 25.93 |

兩種獨立的失效：

1. **型別缺席 → 崩潰**。控制組（訓練時看過同一批型別）LC12 rank-1 **98.15 %**，
   hold-out 後 **0.00 %**；DL2d_adPN 94.12 % → 0.00 %。
2. **少數姊妹型別 → 即使見過也崩潰**。KCab-s 在見過的情況下 rank-1 仍是 0.00 %
   （基線 29.41 %）。KCab-c 有 376 顆 FC、KCab-s 只有 180 顆且形態相近，
   訓練權重壓倒性投給多數型別。**族群報表把它藏在 KC 整體的 73 % 底下。**

⚠️ **dev 偵測不到**：該輪 dev 三元組正確率 0.855 且一路平滑上升。
dev 的負例來自訓練見過的型別分布。**任何以 dev 挑 checkpoint 的流程，
都會選中一個在缺席型別上崩潰的模型而毫無警訊。**

→ **重跑六個組態沒有意義**：`head_init` / `w_bce` / `epochs` / `trunk_init`
沒有一個碰得到型別泛化。要處理的是訓練集的型別涵蓋率，或換方法。
完整的實驗階梯與撤回聲明見 `RANKING_FINETUNE_PLAN.md` §階段 3 結果。

### 14.7 尚未補上的

- 專家標註那 453 顆 FC 還沒 dump 過全池分數，所以
  「專家配對的**個體層級** rank 不得退步」這道守門目前**無法執行**。
  依 §10 的切分，`pool_scores` 涵蓋的 2 735 顆 FC 沒有一顆進過專家標註。
- 按 **EM 型別** hold out 的切分（比 FC 層級切分更嚴格）尚未實作。
- 多 seed × 多切分的穩健性檢驗尚未做。

---

## 15. 建議的交接閱讀順序

1. 先看 [PIPELINE_HANDOVER.md](PIPELINE_HANDOVER.md)，理解三視圖如何生成。
2. 再看 [make_cross_val_set.py](make_cross_val_set.py)，理解 train/test split 如何來。
3. 再看 [Data_process_Train.py](Data_process_Train.py)，理解模型訓練主流程。
4. 再看 [Model_predict.py](Model_predict.py) 和 [merge_pseudo_label.py](merge_pseudo_label.py)，理解 pseudo label 如何來。
5. 再看 [data_process_preTrain.py](data_process_preTrain.py) 和 [result_analysis.py](result_analysis.py)。
6. 若要理解為什麼會有 §14 的排序微調，先讀
   [analysis_model_results/FINDINGS.md](analysis_model_results/FINDINGS.md) §10
   （三輪機制檢驗：全域膨脹、海綿效應都被否證，訓練目標錯配成立），
   再讀 [RANKING_FINETUNE_PLAN.md](RANKING_FINETUNE_PLAN.md)（分階段計畫與 go/no-go）。
