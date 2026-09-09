"""服務端與離線 pipeline 共用的設定。

⚠️ 這個檔案是「畫圖 / 前處理 / 模型」參數的唯一真實來源。
   standard_draw.py 的函式預設值雖然也是 5.0，但任何新的呼叫點都應該
   從這裡取值，不要依賴函式預設值 —— 尺度不一致不會報錯，只會讓分數失去意義。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Side = Literal["FC", "EM"]
SIDES: tuple[str, ...] = ("FC", "EM")


def other_side(side: str) -> str:
    """FC <-> EM。目前資料庫只有這兩側。"""
    s = str(side).strip().upper()
    if s == "FC":
        return "EM"
    if s == "EM":
        return "FC"
    raise ValueError(f"unknown side: {side!r} (expected one of {SIDES})")


def normalize_side(side: str) -> str:
    s = str(side).strip().upper()
    if s not in SIDES:
        raise ValueError(f"unknown side: {side!r} (expected one of {SIDES})")
    return s


@dataclass(frozen=True)
class RenderConfig:
    """三視圖渲染參數。

    scale_um_per_px: 幾微米壓成一個 pixel。圖的像素數正比於神經元的實際
        bounding box 大小，所以所有神經元的圖是同一個物理尺度。
        資料庫裡歸檔的三視圖全部是用 5.0 畫的（已用 bit-for-bit 重畫驗證），
        改這個值等於讓新圖跟舊圖不能比較。
    normalize: "p99" 或 "max"，單張圖的灰階正規化方式。
    out_hw:    送進模型前下採樣的目標大小。
    render_version: 寫進新產生的 npz，讓日後可以判斷某張圖是用哪組參數畫的。
        改了上面任何一個參數就要改這個字串，並讓快取失效。
    """

    scale_um_per_px: float = 5.0
    normalize: str = "p99"
    out_hw: tuple[int, int] = (50, 50)
    render_version: str = "v1_scale5.0_p99"


@dataclass(frozen=True)
class ModelConfig:
    """打分模型。

    weights 是 weights-only 檔，必須先用 MVCNN_Siamese(input_size) 建好圖再載入。
    模型有兩個具名輸入 "FC" / "EM"，查詢側的圖餵進自己那一側。
    """

    weights: str = "./FineTune_Model/FineTune_miniLR_D1-D6_0.weights.h5" #'./Annotator_Model/Annotator_D1-D6_0.weights.h5' 
    input_size: tuple[int, int, int] = (50, 50, 3)
    batch_size: int = 512
    # 寫進結果快取的 key，換模型必須換這個字串，否則會端出用舊模型算的分數
    model_id: str = "FineTune_miniLR_D1-D6_0" #'Annotator_D1-D6_0'


@dataclass(frozen=True)
class MatchConfig:
    """候選初篩參數，與 candidate_matching.py 的離線設定對齊。"""

    centroid_th: float = 100.0        # 質心距離門檻 (um)
    ratio_th: float = 0.4             # (r21, r31) 2D 距離門檻
    # rod 用 35° 而非 30°：v3 在 FC/EM 之間有系統性抖動，用 D1-D6 人工標註量測，
    # 30° 會切掉 9 對真 pair（6 對信心度 >=0.8，含 2 對 1.0），35° 讓高信心損失歸零，
    # 代價是候選池少砍 3.3pp。disk 維持 30°（標註中 disk 正樣本角度 max 僅 26.7°）。
    rod_angle_th_deg: float = 35.0    # rod-like 的方向夾角門檻
    disk_angle_th_deg: float = 30.0   # disk-like 的方向夾角門檻
    # 通過三段過濾後再按 descriptor 距離取前 K，用來保證單次查詢的延遲上限。
    # 實測每次查詢的候選數 p90 約 2920、最大約 6498。
    top_k_candidates: int = 2000


@dataclass(frozen=True)
class ValidationConfig:
    """上傳檔的合法性檢查門檻，數值取自現有資料庫的實測範圍。

    實測節點座標範圍：
        FC  x[-458, 470]  y[-341, 144]  z[-185, 108]
        EM  x[-285, 102]  y[-353, 140]  z[-213, 100]
    """

    # 超過這個值幾乎確定不是 standard brain 的微米座標（多半是原始 nm 或 voxel）
    max_abs_coord: float = 5000.0
    # bounding box 中心必須落在這個範圍內，否則視為沒有 registration
    center_lo: tuple[float, float, float] = (-600.0, -500.0, -350.0)
    center_hi: tuple[float, float, float] = (600.0, 300.0, 250.0)
    # 單顆神經元的 extent 上限（超過只警告不擋，實測 p99 約 455 um）
    warn_extent_um: float = 1000.0
    max_nodes: int = 500_000
    max_file_bytes: int = 64 * 1024 * 1024
    # neuron_ids_*.npy 的 dtype 是 <U64
    max_id_len: int = 64


@dataclass(frozen=True)
class Paths:
    """資料位置。

    data/ 是 curated 資料庫，服務執行期間只讀不寫。
    user_data/ 是使用者上傳的暫存區，未經人工確認前不會進入 curated 資料庫，
    也不會出現在任何人的候選名單裡。
    """

    root: Path = Path(".")

    @property
    def swc_root(self) -> Path:
        return self.root / "data" / "SWC"

    @property
    def descriptor_root(self) -> Path:
        return self.root / "data"

    @property
    def views_root(self) -> Path:
        return self.root / "data" / "standard_views"

    @property
    def index_dir(self) -> Path:
        return self.root / "data" / "index"

    @property
    def view_store_dir(self) -> Path:
        return self.root / "data" / "view_store"

    @property
    def user_data_root(self) -> Path:
        return self.root / "user_data"

    def swc_dir(self, side: str) -> Path:
        return self.swc_root / normalize_side(side)

    def descriptor_dir(self, side: str) -> Path:
        return self.descriptor_root / f"descriptors_{normalize_side(side)}"

    def views_dir(self, side: str) -> Path:
        return self.views_root / normalize_side(side)


@dataclass(frozen=True)
class ServiceConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    paths: Paths = field(default_factory=Paths)

    # 輸出 CSV 預設保留幾對
    default_top_n: int = 5
    # 同一份檔案（sha256 + render_version + model_id 都相同）是否重用上次的結果
    reuse_cached_result: bool = True


# 輸出 CSV 的欄位，順序固定
RESULT_COLUMNS: tuple[str, ...] = ("source_id", "target_id", "similarity_score", "rank")
