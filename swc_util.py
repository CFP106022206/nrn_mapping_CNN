from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Swc:
    nid: np.ndarray
    ntype: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent: np.ndarray


def load_swc_fast(path: str | Path) -> Swc:
    path = Path(path)
    buf = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("#", "//", ";")):
                continue
            if any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in line):
                continue
            buf.append(line.replace(",", " "))

    if not buf:
        raise ValueError(f"{path.name}: no numeric SWC rows found")

    data = np.fromstring(" ".join(buf), sep=" ", dtype=np.float64)
    if data.size % 7 != 0:
        raise ValueError(f"{path.name}: parsed values not divisible by 7 (got {data.size})")

    arr = data.reshape(-1, 7)
    return Swc(
        nid=arr[:, 0].astype(np.int32, copy=False),
        ntype=arr[:, 1].astype(np.int16, copy=False),
        xyz=arr[:, 2:5].astype(np.float32, copy=False),
        radius=arr[:, 5].astype(np.float32, copy=False),
        parent=arr[:, 6].astype(np.int32, copy=False),
    )


def load_swc(path: str | Path) -> Swc:
    return load_swc_fast(path)


def _to_uint8_views(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    if v.dtype == np.uint8:
        return v
    vf = v.astype(np.float32, copy=False)
    vmax = float(np.nanmax(vf)) if vf.size else 0.0
    if vmax <= 1.0:
        vf = np.round(vf * 255.0)
    vf = np.clip(vf, 0.0, 255.0)
    return vf.astype(np.uint8)


def _ensure_3hw_views(v: np.ndarray) -> np.ndarray:
    """Normalize view array to shape (3,H,W)."""
    v = np.asarray(v)
    if v.ndim != 3:
        raise ValueError(f"Expect 3D views, got shape={v.shape}")
    if v.shape[0] == 3:
        return v
    if v.shape[-1] == 3:
        return np.transpose(v, (2, 0, 1))
    raise ValueError(f"Cannot interpret views shape as (3,H,W): {v.shape}")


def _pad_to_same_size(fc_views: np.ndarray, em_views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the smaller square view stack to the larger stack size. Input/output: (3,H,W)."""
    fc = _ensure_3hw_views(_to_uint8_views(fc_views))
    em = _ensure_3hw_views(_to_uint8_views(em_views))
    target = max(int(fc.shape[1]), int(em.shape[1]))

    def _pad(v: np.ndarray) -> np.ndarray:
        _, h, w = v.shape
        if h != w:
            raise ValueError(f"Expect square views (H==W). got {(h, w)}")
        pad = target - h
        if pad < 0:
            raise ValueError(f"target smaller than current: current=({h},{w}) target=({target},{target})")
        top = pad // 2
        bottom = pad - top
        left = pad // 2
        right = pad - left
        return np.pad(v, ((0, 0), (top, bottom), (left, right)), mode="constant", constant_values=0)

    return _pad(fc), _pad(em)


def _resize_to_50(views: np.ndarray, out_hw: tuple[int, int] = (50, 50)) -> np.ndarray:
    """Downsample (3,H,W) to (3,out_h,out_w) with adaptive max pooling; smaller inputs are padded."""
    v = _ensure_3hw_views(_to_uint8_views(views))

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    h, w = int(v.shape[1]), int(v.shape[2])

    if h == out_h and w == out_w:
        return v

    if h < out_h or w < out_w:
        pad_h = max(out_h - h, 0)
        pad_w = max(out_w - w, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        vv = np.pad(v, ((0, 0), (top, bottom), (left, right)), mode="constant", constant_values=0)
        return vv[:, :out_h, :out_w].astype(np.uint8, copy=False)

    y_starts = (np.arange(out_h, dtype=np.int64) * h) // out_h
    x_starts = (np.arange(out_w, dtype=np.int64) * w) // out_w

    tmp = np.maximum.reduceat(v, y_starts, axis=1)
    out = np.maximum.reduceat(tmp, x_starts, axis=2)
    return out.astype(np.uint8, copy=False)


def _load_views_from_npz(npz_path: Path | str) -> np.ndarray:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        if "views" not in z.files:
            raise KeyError(f"Missing key 'views' in {npz_path}. keys={list(z.files)}")
        return z["views"]


def make_numpy_from_standard_views(
    pair_df: pd.DataFrame,
    fc_dir: Path | str = "./data/standard_views/FC",
    em_dir: Path | str = "./data/standard_views/EM",
    out_hw: tuple[int, int] = (50, 50),
    no_valid_message: str = "No valid pairs loaded from standard_views. Check directories, ids, and label CSVs.",
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, tuple[int, int, int]]:
    """Build Siamese pair tensors from standard-view npz files.

    Returns:
      x: (M,2,out_h,out_w,3) float32 in [0,1]
      found_df: fc_id, em_id, label
      not_found_df: fc_id, em_id, label, reason
      resolutions: (V,H,W) = (3,out_h,out_w)
    """
    fc_dir = Path(fc_dir)
    em_dir = Path(em_dir)

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid out_hw={out_hw}")

    if not fc_dir.exists():
        raise FileNotFoundError(f"FC standard views dir not found: {fc_dir}")
    if not em_dir.exists():
        raise FileNotFoundError(f"EM standard views dir not found: {em_dir}")

    fc_cache: dict[str, np.ndarray] = {}
    em_cache: dict[str, np.ndarray] = {}

    x_list: list[np.ndarray] = []
    found_rows: list[tuple[str, str, float]] = []
    not_found_rows: list[tuple[str, str, float, str]] = []

    required_cols = {"fc_id", "em_id", "label"}
    if not required_cols.issubset(set(pair_df.columns)):
        raise KeyError(f"pair_df must contain columns {sorted(required_cols)}. got={list(pair_df.columns)}")

    for row in pair_df.itertuples(index=False):
        fc_id = str(getattr(row, "fc_id")).strip()
        em_id = str(getattr(row, "em_id")).strip()
        label = float(getattr(row, "label"))

        fc_npz = fc_dir / f"{fc_id}_views.npz"
        em_npz = em_dir / f"{em_id}_views.npz"

        if not fc_npz.exists() and not em_npz.exists():
            not_found_rows.append((fc_id, em_id, label, "missing_fc_and_em_npz"))
            continue
        if not fc_npz.exists():
            not_found_rows.append((fc_id, em_id, label, "missing_fc_npz"))
            continue
        if not em_npz.exists():
            not_found_rows.append((fc_id, em_id, label, "missing_em_npz"))
            continue

        try:
            if fc_id in fc_cache:
                fc_v = fc_cache[fc_id]
            else:
                fc_v = _load_views_from_npz(fc_npz)
                fc_cache[fc_id] = fc_v

            if em_id in em_cache:
                em_v = em_cache[em_id]
            else:
                em_v = _load_views_from_npz(em_npz)
                em_cache[em_id] = em_v
        except Exception as e:
            not_found_rows.append((fc_id, em_id, label, f"load_error:{type(e).__name__}:{e}"))
            continue

        try:
            fc_pad, em_pad = _pad_to_same_size(fc_v, em_v)
            fc_50 = _resize_to_50(fc_pad, (out_h, out_w))
            em_50 = _resize_to_50(em_pad, (out_h, out_w))
        except Exception as e:
            not_found_rows.append((fc_id, em_id, label, f"preprocess_error:{type(e).__name__}:{e}"))
            continue

        x_pair = np.empty((2, out_h, out_w, 3), dtype=np.float32)
        x_pair[0] = np.transpose(fc_50, (1, 2, 0))
        x_pair[1] = np.transpose(em_50, (1, 2, 0))
        x_pair /= 255.0

        x_list.append(x_pair)
        found_rows.append((fc_id, em_id, label))

    if not x_list:
        raise RuntimeError(no_valid_message)

    x = np.stack(x_list, axis=0)
    found_df = pd.DataFrame(found_rows, columns=["fc_id", "em_id", "label"])
    not_found_df = pd.DataFrame(not_found_rows, columns=["fc_id", "em_id", "label", "reason"])
    return x, found_df, not_found_df, (3, out_h, out_w)

