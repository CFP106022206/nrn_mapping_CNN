# Standard Draw 使用指南

## 概述

`standard_draw.py` 是用來渲染單一神經元的標準三視圖（XYZ方向垂直投影）的腳本。

**主要改進點：**
- ✅ 每個神經元單獨繪製三視圖（不再是配對）
- ✅ 視角固定為標準腦座標的三個垂直方向（XYZ）
- ✅ 不需要讀取 eigenvector 或 centroid 數據
- ✅ 圖片大小可變，但保證相同的 scale 比例
- ✅ 可通過 `scale_um_per_px` 參數調整縮放因子
- ✅ 代碼更簡潔，只需要 SWC 檔案和神經元列表

## 基本用法

### 1. 準備神經元 ID 列表

**方式 A：文本檔案（每行一個 ID）**
```
cat > neuron_ids.txt << EOF
12345
67890
11111
EOF
```

**方式 B：.npy 陣列**
```python
import numpy as np
neuron_ids = np.array([12345, 67890, 11111])
np.save('neuron_ids.npy', neuron_ids)
```

### 2. 運行腳本

```bash
# 基本使用
python3 standard_draw.py \
    --swc_dir ./data/SWC/FC \
    --neuron_list neuron_ids.txt \
    --output_dir ./standard_views/

# 自定義 scale（1.0 微米/像素 = 預設）
python3 standard_draw.py \
    --swc_dir ./data/SWC/EM \
    --neuron_list neuron_ids.npy \
    --scale_um_per_px 2.0 \
    --output_dir ./standard_views_large/

# 輸出為 PNG 格式（分別保存三個視圖）
python3 standard_draw.py \
    --swc_dir ./data/SWC/FC \
    --neuron_list neuron_ids.txt \
    --format png \
    --output_dir ./standard_views_png/

# 調試模式（只處理前 10 個神經元）
python3 standard_draw.py \
    --swc_dir ./data/SWC/FC \
    --neuron_list neuron_ids.txt \
    --max_neurons 10 \
    --output_dir ./test_output/
```

## 參數說明

| 參數 | 說明 | 默認值 | 必需 |
|------|------|--------|------|
| `--swc_dir` | SWC 檔案所在目錄 | - | ✅ |
| `--neuron_list` | 神經元 ID 列表（.txt 或 .npy） | - | ✅ |
| `--scale_um_per_px` | 微米/像素，控制圖像大小 | 1.0 | ❌ |
| `--normalize` | 歸一化方式：max 或 p99 | p99 | ❌ |
| `--output_dir` | 輸出目錄 | ./standard_views/ | ❌ |
| `--format` | 輸出格式：npz 或 png | npz | ❌ |
| `--max_neurons` | 最大處理神經元數（0=全部） | 0 | ❌ |

## 輸出格式

### NPZ 格式（默認）
```
output_dir/
  12345_views.npz     # 包含三個視圖的壓縮陣列
  67890_views.npz
  11111_views.npz
```

NPZ 文件包含的內容：
- `nid`：神經元 ID（字符串）
- `views`：(3, H, W) uint8 陣列，三個視圖
  - view 0：YZ 平面
  - view 1：XZ 平面
  - view 2：XY 平面
- `grid_size`：圖像大小（int32）

### PNG 格式
```
output_dir/
  12345_view_0.png    # YZ 平面
  12345_view_1.png    # XZ 平面
  12345_view_2.png    # XY 平面
  67890_view_0.png
  ...
```

## 讀取 NPZ 結果

```python
import numpy as np

# 載入單個神經元的三視圖
data = np.load('output_dir/12345_views.npz')
nid = data['nid']
views = data['views']  # (3, H, W) uint8
grid_size = data['grid_size']

# views[0] = YZ 平面投影
# views[1] = XZ 平面投影
# views[2] = XY 平面投影
```

## scale_um_per_px 參數詳解

`scale_um_per_px` 控制了多少微米的距離對應一個像素。

- **小值**（如 0.5）：更多的標準腦座標被映射到像素上 → **更大的圖**
- **大值**（如 2.0）：更少的標準腦座標被映射到像素上 → **更小的圖**

例如：
```
SWC 坐標範圍：500 微米 × 600 微米 × 700 微米

scale_um_per_px = 1.0   → grid ≈ 700×700 像素
scale_um_per_px = 2.0   → grid ≈ 350×350 像素（max(500, 600, 700) / 2.0）
scale_um_per_px = 0.5   → grid ≈ 1400×1400 像素
```

## 常見問題

**Q: 為什麼有些神經元報錯？**
A: 可能是 SWC 檔案不存在、檔案格式不正確，或檔案為空。檢查 `--swc_dir` 路徑和神經元 ID 是否與 SWC 檔案名稱相符。

**Q: 如何獲得與舊 pair_draw 類似的效果？**
A: 舊代碼為配對神經元使用相同的座標系和視角。新代碼支持為單一神經元生成標準三視圖，無需配對。

**Q: 三個視圖的大小不一樣？**
A: 代碼會自動將三個視圖調整為相同大小（取最大值）。這保證了一致的視角。

## 效能提示

- 使用 `--max_neurons` 進行小規模測試
- 緩存機制會自動保存 SWC 檔案的處理結果，加速後續重複處理
- 大批量處理時，考慮增加 `scale_um_per_px` 以減少記憶體使用

## 簡化代碼與原始 pair_draw 的主要差異

| 功能 | pair_draw | standard_draw |
|------|-----------|---------------|
| 處理對象 | 配對神經元 | 單個神經元 |
| 座標系 | 使用 eigenvector 旋轉 | 標準腦座標（XYZ） |
| 依賴數據 | 需要 eigenvector, centroid | 只需要 SWC 檔案 |
| 輸出格式 | 配對 NPZ 分片 | 單個神經元 NPZ/PNG |
| 視角 | 根據配對調整 | 固定（XYZ 垂直方向） |
| 圖像大小 | 固定 grid | 可變（但 scale 固定） |

