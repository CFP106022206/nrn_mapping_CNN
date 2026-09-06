"""上傳檔的合法性檢查。

擋兩類問題：
1. 根本不是 standard brain 座標的檔案（沒 registration、單位是 nm 或 voxel）。
   這種檔案跑完整條流程不會報錯，只會安靜地給出沒有意義的分數，所以必須擋在最前面。
2. 會讓後面爆掉或拖垮延遲的檔案（節點過多、檔案過大、檔名有路徑字元）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from swc_util import Swc
from .config import ValidationConfig

# 檔名只允許這些字元：資料庫裡的 ID 形如 104198-F-000000（FC）或 1001453586（EM）
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._\-]+$")


class ValidationError(ValueError):
    """上傳檔無法處理。訊息會直接回給前端。"""


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_nodes: int = 0
    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValidationError("; ".join(self.errors))


def sanitize_neuron_id(name: str, cfg: ValidationConfig) -> str:
    """把上傳檔名轉成可以安全組進路徑的 neuron id。

    neuron id 會被拿去組 `<dir>/<id>.swc`、`<dir>/<id>_views.npz`，
    所以必須先擋掉 `/`、`..` 這類字元，長度也要符合 neuron_ids_*.npy 的 <U64。
    """
    stem = Path(str(name)).name          # 去掉任何目錄部分
    if stem.lower().endswith(".swc"):
        stem = stem[:-4]
    stem = stem.strip()

    if not stem:
        raise ValidationError("檔名為空")
    if stem in (".", ".."):
        raise ValidationError(f"不合法的檔名: {name!r}")
    if not _SAFE_ID_RE.match(stem):
        raise ValidationError(
            f"檔名含有不允許的字元: {name!r}（只接受英數字、'.'、'_'、'-'）"
        )
    if len(stem) > cfg.max_id_len:
        raise ValidationError(
            f"檔名過長: {len(stem)} > {cfg.max_id_len}（資料庫 neuron id 的上限）"
        )
    return stem


def load_swc_or_raise(path, original_name: str = "") -> Swc:
    """讀取 SWC，把底層的解析錯誤轉成可以直接回給前端的訊息。

    load_swc_fast 的錯誤訊息帶的是內部落地後的檔名（neuron.swc），
    直接回給使用者會看不懂是自己上傳的哪個檔。
    """
    from swc_util import load_swc_fast

    shown = original_name or str(path)
    try:
        return load_swc_fast(path)
    except Exception as e:
        raise ValidationError(
            f"{shown} 不是可解析的 SWC 檔（{type(e).__name__}）。"
            f"SWC 每一行需要 7 個數值欄位：id type x y z radius parent"
        ) from e


def validate_file_size(n_bytes: int, cfg: ValidationConfig) -> None:
    if n_bytes <= 0:
        raise ValidationError("上傳檔案是空的")
    if n_bytes > cfg.max_file_bytes:
        raise ValidationError(
            f"檔案過大: {n_bytes / 1024 / 1024:.1f} MB > {cfg.max_file_bytes / 1024 / 1024:.0f} MB"
        )


def validate_swc(swc: Swc, cfg: ValidationConfig) -> ValidationReport:
    """檢查骨架是否落在 standard brain 的合理範圍內。"""
    errors: list[str] = []
    warnings: list[str] = []

    n = int(swc.xyz.shape[0])
    if n == 0:
        return ValidationReport(ok=False, errors=["SWC 沒有任何節點"], n_nodes=0)

    if n > cfg.max_nodes:
        errors.append(f"節點數過多: {n} > {cfg.max_nodes}")

    xyz = np.asarray(swc.xyz, dtype=np.float64)
    if not np.isfinite(xyz).all():
        errors.append("座標含有 NaN 或 Inf")
        return ValidationReport(ok=False, errors=errors, n_nodes=n)

    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) * 0.5
    extent = hi - lo

    max_abs = float(np.abs(xyz).max())
    if max_abs > cfg.max_abs_coord:
        errors.append(
            f"座標數量級不對（max|coord| = {max_abs:.0f} > {cfg.max_abs_coord:.0f}）。"
            f"這通常表示 SWC 還是原始的 nm 或 voxel 單位，"
            f"必須先 warp 到 standard brain 的微米座標再上傳"
        )
    else:
        # 只有在數量級看起來對的時候，檢查位置才有意義
        c_lo = np.asarray(cfg.center_lo, dtype=np.float64)
        c_hi = np.asarray(cfg.center_hi, dtype=np.float64)
        if np.any(center < c_lo) or np.any(center > c_hi):
            ctr = [round(float(v), 1) for v in center]
            rng = ", ".join(
                f"{ax}[{float(lo):.0f}, {float(hi):.0f}]"
                for ax, lo, hi in zip("xyz", c_lo, c_hi)
            )
            errors.append(
                f"神經元中心 {ctr} 落在 standard brain 範圍之外（允許 {rng}）。"
                f"請確認 SWC 已經做過 registration"
            )

    if float(extent.max()) > cfg.warn_extent_um:
        warnings.append(
            f"神經元跨度異常大（{np.round(extent, 1).tolist()} um），請確認是否為單一神經元"
        )

    if float(extent.max()) <= 0.0:
        errors.append("所有節點座標相同，無法計算幾何特徵")

    # descriptor 需要至少一段有長度的 parent-child segment
    if int((swc.parent != -1).sum()) == 0:
        errors.append("SWC 沒有任何 parent-child 連線，無法計算 descriptor")

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        n_nodes=n,
        bbox_min=tuple(float(v) for v in lo),
        bbox_max=tuple(float(v) for v in hi),
    )


def validate_views(views: np.ndarray, neuron_id: str) -> None:
    """三視圖健檢。

    ⚠️ 全零圖一定要擋掉：實測 FineTune_miniLR_D1-D6_0 對全零輸入會輸出 0.678，
       等於一張空圖會直接排到結果前段。
    """
    v = np.asarray(views)
    if v.ndim != 3 or v.shape[0] != 3:
        raise ValidationError(f"{neuron_id}: 三視圖形狀不對 {v.shape}，預期 (3, H, W)")
    if v.shape[1] != v.shape[2]:
        raise ValidationError(f"{neuron_id}: 三視圖不是正方形 {v.shape}")
    if int(v.max()) == 0:
        raise ValidationError(
            f"{neuron_id}: 渲染出來是全黑的圖。模型對空白輸入會給高分，因此拒絕評分"
        )
