"""dense / projection 兩組神經量化研究的共用設定。

所有路徑都相對於 repository 根目錄解析, 因此 pipeline 可以從任何位置執行:
    python analysis_dense_vs_projection/s01_build_neuron_lists.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ 輸入資料
DENSE_CSV = ROOT / "labeled_info" / "D2+D6_ID.csv"   # 論文的 sub dataset D2 (dense)
PROJ_CSV = ROOT / "labeled_info" / "D5_conf.csv"     # 論文的 sub dataset D1 (projection)
NEUROPIL_CSV = ROOT / "data" / "neuron1x1Coding_Ver2.csv"
SWC_FC = ROOT / "data" / "SWC" / "FC"
SWC_EM = ROOT / "data" / "SWC" / "EM"

# -------------------------------------------------------------------- 組別名稱
GROUP_DENSE = "dense"        # 即 D2+D6
GROUP_PROJ = "projection"    # 即 D5
GROUP_ORDER = [GROUP_PROJ, GROUP_DENSE]

# ---------------------------------------------------------------------- 分析參數
# 58 個 neuropil 欄位 = 29 個解剖腦區 x {左, 右}; `volume` 與 `other` 是
# neuron1x1Coding_Ver2.csv 的統計欄位, 另外處理。
META_COLS = ["volume", "other"]

VOXEL_UM = 2.0          # 佔位/密度指標所用的 voxel 邊長 (um)
RESAMPLE_UM = 1.0       # 點雲類指標的骨架重採樣步長 (um)
MIN_SHARE = 0.05        # 一個 neuropil 佔到 arbor 的 5 % 以上才算數
POS_CONF = 0.5          # 專家信心 (0-1) >= 此值為正例; 與論文和 result_analysis_make_figure.py 一致
RANDOM_STATE = 0
