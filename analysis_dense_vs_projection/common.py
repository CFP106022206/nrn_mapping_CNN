"""pipeline 各步驟共用的資料讀取與統計工具。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo 根目錄, 為了 swc_util
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 本資料夾

import study_config as C  # noqa: E402
from swc_util import load_swc_fast  # noqa: E402


# ------------------------------------------------------------------- SWC 相關 --
def swc_path(neuron_id: str, source: str) -> Path:
    base = C.SWC_FC if source == "FC" else C.SWC_EM
    return base / f"{neuron_id}.swc"


def _parent_index(swc) -> np.ndarray:
    """每個節點的 parent 所在列索引; 沒有 parent (root) 時為 -1。"""
    order = np.argsort(swc.nid)
    sorted_nid = swc.nid[order]
    pos = np.searchsorted(sorted_nid, swc.parent)
    pos = np.clip(pos, 0, len(sorted_nid) - 1)
    idx = order[pos]
    valid = sorted_nid[pos] == swc.parent
    return np.where(valid, idx, -1)


def segments(swc) -> tuple[np.ndarray, np.ndarray]:
    """回傳樹上每一段 parent-child 線段的 (中點, 長度)。"""
    xyz = swc.xyz.astype(np.float64, copy=False)
    pidx = _parent_index(swc)
    m = pidx >= 0
    if not m.any():
        return np.empty((0, 3)), np.empty(0)
    a, b = xyz[m], xyz[pidx[m]]
    length = np.linalg.norm(a - b, axis=1)
    keep = length > 0
    return (a[keep] + b[keep]) * 0.5, length[keep]


def resample_cable(swc, step: float = C.RESAMPLE_UM) -> tuple[np.ndarray, np.ndarray]:
    """把骨架重採樣成近似等距的點, 並附上該點的局部切線方向。

    回傳 (points (N,3), 單位切向量 (N,3)) -- 這是 NBLAST 類分數所操作的表示法。
    每段線段貢獻 ceil(長度/step) 個點, 因此點密度正比於 cable length, 而不是
    正比於 FlyCircuit 與 hemibrain 兩者差異極大的節點密度。
    """
    xyz = swc.xyz.astype(np.float64, copy=False)
    pidx = _parent_index(swc)
    m = pidx >= 0
    if not m.any():
        return np.empty((0, 3)), np.empty((0, 3))
    a, b = xyz[m], xyz[pidx[m]]
    d = a - b
    L = np.linalg.norm(d, axis=1)
    keep = L > 0
    b, d, L = b[keep], d[keep], L[keep]
    if L.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    n = np.maximum(np.ceil(L / step).astype(np.int64), 1)
    idx = np.repeat(np.arange(n.size), n)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    k = np.arange(int(n.sum())) - np.repeat(starts, n)
    t = (k + 0.5) / n[idx]
    pts = b[idx] + t[:, None] * d[idx]
    vecs = (d / L[:, None])[idx]
    return pts, vecs


# ------------------------------------------------------------------ 統計工具 --
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = 2*AUC-1, 由 Mann-Whitney U 統計量換算。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2.0 * (u / (x.size * y.size)) - 1.0


def auc_from_u(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y), 平手算 1/2 -- 也就是單一特徵的 ROC AUC。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return u / (x.size * y.size)


def best_threshold(pos: np.ndarray, neg: np.ndarray) -> dict:
    """讓 Youden's J (= balanced accuracy * 2 - 1) 最大的單一特徵切點。

    `pos` 是規則應該命中的那一組。兩個方向 (>= t 與 <= t) 都會嘗試, 回報較好的一個。
    """
    pos = np.asarray(pos, float)[np.isfinite(np.asarray(pos, float))]
    neg = np.asarray(neg, float)[np.isfinite(np.asarray(neg, float))]
    if pos.size == 0 or neg.size == 0:
        return {}
    cand = np.unique(np.concatenate([pos, neg]))
    mids = np.concatenate([[cand[0] - 1e-9], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1e-9]])
    best = {"youden": -np.inf}
    for direction in (">=", "<="):
        for t in mids:
            tp = (pos >= t).sum() if direction == ">=" else (pos <= t).sum()
            fp = (neg >= t).sum() if direction == ">=" else (neg <= t).sum()
            sens = tp / pos.size
            spec = 1 - fp / neg.size
            j = sens + spec - 1
            if j > best["youden"]:
                best = {
                    "youden": j, "threshold": float(t), "direction": direction,
                    "sensitivity": sens, "specificity": spec,
                    "balanced_acc": (sens + spec) / 2,
                    "accuracy": (tp + (neg.size - fp)) / (pos.size + neg.size),
                }
    return best


def describe_split(df: pd.DataFrame, feature: str, group_col: str = "group",
                   pos_group: str = C.GROUP_DENSE, neg_group: str = C.GROUP_PROJ) -> dict:
    """單一特徵的完整對照: 位置統計量、效果量、檢定與最佳切點。"""
    pos = df.loc[df[group_col] == pos_group, feature].to_numpy(float)
    neg = df.loc[df[group_col] == neg_group, feature].to_numpy(float)
    pos_f, neg_f = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos_f.size < 3 or neg_f.size < 3:
        return {}
    p = stats.mannwhitneyu(pos_f, neg_f, alternative="two-sided").pvalue
    auc = auc_from_u(pos_f, neg_f)
    row = {
        "feature": feature,
        f"n_{pos_group}": pos_f.size, f"n_{neg_group}": neg_f.size,
        f"{pos_group}_median": np.median(pos_f), f"{pos_group}_iqr_lo": np.percentile(pos_f, 25),
        f"{pos_group}_iqr_hi": np.percentile(pos_f, 75),
        f"{neg_group}_median": np.median(neg_f), f"{neg_group}_iqr_lo": np.percentile(neg_f, 25),
        f"{neg_group}_iqr_hi": np.percentile(neg_f, 75),
        f"{pos_group}_mean": pos_f.mean(), f"{pos_group}_sd": pos_f.std(ddof=1),
        f"{neg_group}_mean": neg_f.mean(), f"{neg_group}_sd": neg_f.std(ddof=1),
        "mannwhitney_p": p,
        "cliffs_delta": 2 * auc - 1,
        "auc": max(auc, 1 - auc),          # 不分方向的可分離度
        "auc_signed": auc,
    }
    row.update({f"cut_{k}": v for k, v in best_threshold(pos_f, neg_f).items()})
    return row
