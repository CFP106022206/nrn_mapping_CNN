# 全流程使用說明
整個流程分成兩個部分：1、將swc 數據做初步篩選配對以及畫出三視圖。2、用模型預測每一對三視圖相似度。
這份使用說明為第一部分

這份文檔說明從 `SWC` 原始數據到 `descriptor`、`pair matching`、`standard view` 的完整流程。

## 1. 流水線總覽

主入口腳本是 [swc_pair_and_draw.py](swc_pair_and_draw.py)。它按順序做四件事：

1. 讀取 FC / EM 的原始 `SWC` 文件，生成 descriptor。
2. 用 descriptor 做 candidate matching，輸出 pair CSV。
3. 根據 pair CSV，渲染 FC 的標準三視圖。
4. 根據同一份 pair CSV，渲染 EM 的標準三視圖。

整個過程不會覆蓋原始 `SWC` 數據；只會在你指定的輸出目錄里生成中間文件和結果文件。

## 2. 關鍵目錄

默認約定如下：

- 原始 `SWC`：`./data/SWC/FC`、`./data/SWC/EM`
- FC descriptor 輸出：`./data/descriptors_FC`
- EM descriptor 輸出：`./data/descriptors_EM`
- pair 輸出：`./data/pairs_label`
- 標準視圖輸出：`./data/standard_views/FC`、`./data/standard_views/EM`

如果你的目錄結構不同，可以在命令行參數里改掉。

## 3. 每一步分別做什麼

### 3.1 Descriptor 提取

這一步把每個 `SWC` 文件轉換成 descriptor 文件，典型輸出包括：

- `descriptors_FC.parquet` / `descriptors_EM.parquet`
- `centroids_FC.npy` / `centroids_EM.npy`
- `eigvecs_FC.npy` / `eigvecs_EM.npy`
- `eigvals_ratio_FC.npy` / `eigvals_ratio_EM.npy`
- `neuron_ids_FC.npy` / `neuron_ids_EM.npy`

這一步由 [swc_descriptor_batch.py](swc_descriptor_batch.py) 里的 `batch_run()` 完成，主腳本里是第 1、2 步。

### 3.2 Candidate Matching

這一步由 [candidate_matching.py](candidate_matching.py) 完成。

它會讀取 FC / EM descriptor，先按 centroid 距離過濾，再按 inertia ratio 距離過濾，最後再做 orientation 過濾，輸出 pair CSV：

- `pairs_FC_EM.csv`


### 3.3 Standard View 渲染

這一步由 [standard_draw.py](standard_draw.py) 完成。

它讀取 pair CSV，然後分別從 FC / EM 的原始 `SWC` 里把 pair 對應的神經元渲染成標準三視圖，輸出為 `npz` 文件：

- `data/standard_views/FC/*_views.npz`
- `data/standard_views/EM/*_views.npz`

每個 `npz` 里通常包含：

- `nid` 神經ID
- `views` 三視圖


## 4. 一鍵運行全流程

### 4.1 命令行入口

這個全流程是通過 [swc_pair_and_draw.py](swc_pair_and_draw.py) 直接用命令行啟動的。最基本的調用形式是：

```bash
python swc_pair_and_draw.py \
    --fc_swc_dir ./data/SWC/FC \
    --em_swc_dir ./data/SWC/EM
```

建議在倉庫根目錄下執行，因為默認的輸入和輸出路徑都是按當前目錄解析的。
這條命令會使用默認輸出目錄，依次生成 descriptor、pair、standard view。


## 5. 常用參數

`swc_pair_and_draw.py` 的主要參數：

- `--fc_swc_dir`：FC 原始 `SWC` 目錄
- `--em_swc_dir`：EM 原始 `SWC` 目錄
- `--descriptor_root`：descriptor 輸出根目錄，默認 `./data`
- `--pairs_out_dir`：pair CSV 輸出目錄，默認 `./data/pairs_label/`
- `--views_root`：標準視圖輸出根目錄，默認 `./data/standard_views/`
- `--centroid_th`：candidate matching 的 centroid 距離閾值，默認 `100.0`
- `--ratio_th`：candidate matching 的 ratio 距離閾值，默認 `0.4`
- `--no-recursive`：只掃描頂層，不遞歸子目錄
- `--fail-fast`：遇到單個 `SWC` 錯誤時立即停止
- `--scale_um_per_px`：標準視圖縮放參數，默認 `5.0`
- `--normalize`：歸一化方式，`max` 或 `p99`
- `--no-skip-existing`：即使文件已存在也重新渲染

## 6. 輸出結果怎麼檢查

如果全流程成功，通常你會看到以下類型的文件：

- `data/descriptors_FC/descriptors_FC.parquet`
- `data/descriptors_EM/descriptors_EM.parquet`
- `data/pairs_label/pairs_FC_EM.csv`
- `data/standard_views/FC/*.npz`
- `data/standard_views/EM/*.npz`

你可以用下面的方式快速確認：

```bash
ls data/descriptors_FC
ls data/descriptors_EM
ls data/pairs_label
ls data/standard_views/FC | head
ls data/standard_views/EM | head
```
## 7. （已刪除）

## 8. 只跑某一段時怎麼做

### 8.1 只跑 candidate matching

前提是 descriptor 已經存在：

```bash
python candidate_matching.py \
    --fc_dir ./data/descriptors_FC \
    --em_dir ./data/descriptors_EM \
    --out_dir ./data/pairs_label
```

### 8.2 只跑 standard view 渲染

前提是 pair CSV 已經存在：

```bash
python standard_draw.py \
    --swc_dir ./data/SWC/FC \
    --neuron_list ./data/pairs_label/pairs_FC_EM.csv \
    --csv_id_col fc_id \
    --output_dir ./data/standard_views/FC \
    --format npz
```

EM 側同理，把 `--swc_dir` 和 `--csv_id_col` 改成 `./data/SWC/EM` 和 `em_id`。

## 9. 回歸測試腳本

倉庫里還有 [swc_pair_and_draw_test.py](swc_pair_and_draw_test.py)，它的目標不是覆蓋原始數據，而是在臨時目錄里覆跑完整流程，然後和現有歸檔結果做比較。

如果你想確認這條 pipeline 有沒有被改壞，可以用：

```bash
python swc_pair_and_draw_test.py \
    --fc_swc_dir ./data/SWC/FC \
    --em_swc_dir ./data/SWC/EM
```

這個測試腳本會：

- 在臨時目錄里生成 descriptor、pair、standard views
- 不覆蓋原始歸檔
- 默認只檢查 pair CSV 的 schema，如果你另外提供真正的 pair 歸檔路徑，也可以做嚴格對比

## 10. 常見問題

### Q: 為什麼會報找不到 pair CSV？

先確認有沒有跑過 [swc_pair_and_draw.py](swc_pair_and_draw.py) 的 matching 階段。pair 文件是 `candidate_matching.py` 生成的，不是 `standard_draw.py` 生成的。

### Q: 為什麼只有 descriptor 有文件，但 pair 沒有？

通常是 matching 階段沒跑，或者 `--pairs_out_dir` 指到了別的目錄。

### Q: 為什麼 standard view 沒有生成？

通常是 pair CSV 沒有正確生成，或者 `--swc_dir` / `--neuron_list` 傳錯。

### Q: 運行時為什麼不要直接覆蓋原始目錄？

因為這條 pipeline 的中間產物很多，建議全部寫到輸出目錄或臨時目錄里，便於排查和回滾。

## 11. 建議的交接順序

按這個順序理解代碼：

1. 先看 [swc_pair_and_draw.py](swc_pair_and_draw.py)
2. 再看 [candidate_matching.py](candidate_matching.py)
3. 再看 [standard_draw.py](standard_draw.py)
4. 最後看 [swc_descriptor_batch.py](swc_descriptor_batch.py) 和 [swc_descriptor.py](swc_descriptor.py)

