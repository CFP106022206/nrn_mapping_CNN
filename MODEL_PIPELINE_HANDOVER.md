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

主流程可以分成六段：

1. 用人類專家標註結果產生 train/test cross-validation split。
2. 針對每個 fold 訓練初始 annotator model。
3. 用 `result_analysis.py` 檢查初始模型結果。
4. 用 10 個初始模型對未標註 pair 產生 pseudo label prediction。
5. 合併 10 個 fold 的 pseudo label，保留標準差低的 high-confidence pair。
6. 用 pseudo label pre-train 一個模型，再用人類標註資料 fine-tune。

## 2. 關鍵腳本

- [make_cross_val_set.py](make_cross_val_set.py)：產生 cross-validation train/test split。
- [Data_process_Train.py](Data_process_Train.py)：讀取 split 和三視圖，訓練 annotator 或 fine-tune model。
- [Data_process_Train.sh](Data_process_Train.sh)：批次跑 10 個 fold。
- [result_analysis.py](result_analysis.py)：分析模型結果、畫 loss curve、violin plot、ROC、confusion matrix。
- [Model_predict.py](Model_predict.py)：用指定 fold 模型對未標註 pair 做 prediction。
- [Model_predict.sh](Model_predict.sh)：批次用 10 個 fold 模型做 prediction。
- [merge_pseudo_label.py](merge_pseudo_label.py)：合併 10 個 fold 的 prediction，篩選 high-confidence pseudo label。
- [data_process_preTrain.py](data_process_preTrain.py)：用 pseudo label 從頭 pre-train。
- [merge_Model_predict.py](merge_Model_predict.py)：將 prediction CSV 合併後，按 `fc_id` 或 `em_id` 取 top rank。

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
- 未標註 pair 名單：`./data/pairs_label/EMxFC_6KK_last.csv`
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

輸出：

- `data/pairs_label/EMxFC_all_high_confidence.csv`

這份 CSV 是下一步 pre-train 的輸入。

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
bash Model_predict.sh
python3 merge_pseudo_label.py
python3 data_process_preTrain.py
bash Data_process_Train.sh
python3 result_analysis.py
```

## 13. 需要留意的地方

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

## 14. 建議的交接閱讀順序

1. 先看 [PIPELINE_HANDOVER.md](PIPELINE_HANDOVER.md)，理解三視圖如何生成。
2. 再看 [make_cross_val_set.py](make_cross_val_set.py)，理解 train/test split 如何來。
3. 再看 [Data_process_Train.py](Data_process_Train.py)，理解模型訓練主流程。
4. 再看 [Model_predict.py](Model_predict.py) 和 [merge_pseudo_label.py](merge_pseudo_label.py)，理解 pseudo label 如何來。
5. 最後看 [data_process_preTrain.py](data_process_preTrain.py) 和 [result_analysis.py](result_analysis.py)。
