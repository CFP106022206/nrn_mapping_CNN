"""步驟 3 - 由 SWC 檔計算的骨架幾何描述子。

輸入: data/SWC/{FC,EM}/<neuron_id>.swc  (兩個資料庫都已對齊到同一個標準腦
      座標系, 因此長度可以直接比較)
輸出: results/morphology_metrics.csv

描述子家族
----------
尺寸    : cable_length_um, n_branch_points, span_um
緊緻度  : hull_volume_um3, occupied_volume_um3, fill_ratio,
          cable_per_occupied_um2 (= 每單位佔用體積裡的 cable 長度),
          cable_per_rg (以空間範圍正規化後的 cable 長度)
擁擠度  : local_density_r5 / r10 -- 以骨架上的點為球心, 半徑 5/10 um 球內
          cable 長度的中位數
          revisit_r{4,8,16}um -- cable 長度 / (佔用格子數 x 格子邊長), 也就是
          一個被佔用的格子平均被 cable 穿過幾次。1.0 = 從不重複經過。
          細尺度 (1-2 um) 會飽和在 1, 粗尺度才看得出纏繞程度。
          overdraw_2d -- 投影到三個正交平面後的 cable / 佔用像素, 取平均。
          這是模型的三視圖與 NBLAST 的最近鄰計分實際感受到的密度,
          也是 hemibrain 側「海綿效應」的主要指標 (見 s10)
形狀    : elongation, flatness (共變異數矩陣特徵值比)
投射結構: arbor_separation, arbor_balance, bridge_fraction -- 由 cable 長度
          加權的 2-means 分割得到的雙葉結構
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K

N_PROBE = 3000      # 計算局部擁擠度中位數時的探測點數
REVISIT_VOXELS = (4.0, 8.0, 16.0)   # 重複穿越指標的格子邊長 (um)
PIXEL_UM = 1.0      # 三視圖投影的像素邊長
N_KMEANS = 20000    # 送進雙葉分割的線段中點數
N_HULL = 50000      # 送進凸包計算的點數


def _weighted_kmeans2(X: np.ndarray, w: np.ndarray, iters: int = 50, seed: int = 0):
    """以 cable 長度加權的 2-means, 起始點取自最遠的兩個端點。"""
    rng = np.random.default_rng(seed)
    c0 = X[np.argmax(np.linalg.norm(X - np.average(X, axis=0, weights=w), axis=1))]
    c1 = X[np.argmax(np.linalg.norm(X - c0, axis=1))]
    cen = np.vstack([c0, c1])
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - cen[None, :, :], axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for k in (0, 1):
            m = lab == k
            if m.sum() == 0:
                cen[k] = X[rng.integers(len(X))]
            else:
                cen[k] = np.average(X[m], axis=0, weights=w[m])
    return lab, cen


def descriptors(path: Path) -> dict:
    swc = K.load_swc_fast(path)
    mid, seg = K.segments(swc)
    if seg.size < 10:
        return {}
    L = float(seg.sum())

    # ---- 拓撲
    id2idx = {int(v): i for i, v in enumerate(swc.nid)}
    deg = np.zeros(len(swc.nid), dtype=int)
    for pid in swc.parent:
        j = id2idx.get(int(pid))
        if j is not None:
            deg[j] += 1
    n_branch = int((deg >= 2).sum())
    n_tips = int((deg == 0).sum())

    # ---- 空間範圍
    cen = np.average(mid, axis=0, weights=seg)
    d2 = ((mid - cen) ** 2).sum(axis=1)
    rg = float(np.sqrt(np.average(d2, weights=seg)))

    cov = np.cov((mid - cen).T, aweights=seg)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.clip(ev, 1e-12, None)
    elongation = float(np.sqrt(ev[1] / ev[0]))   # 1 = 各向同性, 0 = 一條線
    flatness = float(np.sqrt(ev[2] / ev[0]))

    try:
        hull = ConvexHull(mid if len(mid) <= N_HULL else
                          mid[np.random.default_rng(C.RANDOM_STATE).choice(len(mid), N_HULL, replace=False)])
        hull_v, hull_a = float(hull.volume), float(hull.area)
        hv = mid[hull.vertices]
        span = float(np.linalg.norm(hv[:, None, :] - hv[None, :, :], axis=2).max())
    except Exception:
        hull_v = hull_a = span = np.nan

    # ---- 固定 voxel 網格上的佔位 (不受節點數多寡影響)
    pts, _ = K.resample_cable(swc, C.RESAMPLE_UM)
    vox = np.unique(np.floor(pts / C.VOXEL_UM).astype(np.int64), axis=0)
    n_vox = int(len(vox))
    occ_v = n_vox * C.VOXEL_UM ** 3

    # ---- 擁擠度: 骨架上一個小球內有多少 cable。
    # KD-tree 建在所有重採樣點上 (所以估計值是精確的), 但查詢點數設上限 --
    # 幾千個探測點的中位數已經很穩定, 且讓計算成本不隨重建規模成長。
    tree = cKDTree(pts)
    rng = np.random.default_rng(C.RANDOM_STATE)
    probe = pts if len(pts) <= N_PROBE else pts[rng.choice(len(pts), N_PROBE, replace=False)]
    dens5 = np.asarray(tree.query_ball_point(probe, 5.0, return_length=True), float) * C.RESAMPLE_UM
    dens10 = np.asarray(tree.query_ball_point(probe, 10.0, return_length=True), float) * C.RESAMPLE_UM

    # ---- 多尺度的重複穿越次數與三視圖自我遮蔽
    revisit = {}
    for v in REVISIT_VOXELS:
        nv = len(np.unique(np.floor(pts / v).astype(np.int64), axis=0))
        revisit[f"revisit_r{v:g}um"] = L / (nv * v) if nv else np.nan
    over = []
    for ax0, ax1 in ((0, 1), (0, 2), (1, 2)):
        px = np.unique(np.floor(pts[:, [ax0, ax1]] / PIXEL_UM).astype(np.int64), axis=0)
        over.append(L / (len(px) * PIXEL_UM) if len(px) else np.nan)

    # ---- 雙葉 / 投射結構
    if len(mid) > N_KMEANS:
        sel = np.random.default_rng(C.RANDOM_STATE).choice(len(mid), N_KMEANS, replace=False)
        mid_k, seg_k = mid[sel], seg[sel]
    else:
        mid_k, seg_k = mid, seg
    lab, cc = _weighted_kmeans2(mid_k, seg_k, seed=C.RANDOM_STATE)
    mid, seg = mid_k, seg_k
    w = np.array([seg[lab == k].sum() for k in (0, 1)])
    if w.min() > 0:
        s = np.array([np.sqrt(np.average(((mid[lab == k] - cc[k]) ** 2).sum(axis=1),
                                         weights=seg[lab == k])) for k in (0, 1)])
        pooled = float(np.sqrt(np.average(s ** 2, weights=w)))
        d_cc = float(np.linalg.norm(cc[0] - cc[1]))
        arbor_sep = d_cc / pooled if pooled > 0 else np.nan
        arbor_balance = float(w.min() / w.max())
        # 不屬於任一 arbor 核心的 cable -> 連接兩者的纖維束
        r = np.linalg.norm(mid - cc[lab], axis=1)
        bridge = float(seg[r > s[lab]].sum() / L)
    else:
        arbor_sep = arbor_balance = bridge = d_cc = np.nan

    return {
        "cable_length_um": L,
        "n_nodes": int(len(swc.nid)),
        "n_branch_points": n_branch,
        "n_tips": n_tips,
        "branch_per_100um": 100.0 * n_branch / L,
        "span_um": span,
        "radius_gyration_um": rg,
        "hull_volume_um3": hull_v,
        "hull_area_um2": hull_a,
        "occupied_volume_um3": occ_v,
        "n_voxels": n_vox,
        "fill_ratio": occ_v / hull_v if hull_v and np.isfinite(hull_v) and hull_v > 0 else np.nan,
        "cable_per_occupied_um2": L / occ_v if occ_v > 0 else np.nan,
        "cable_per_hull_um2": L / hull_v if hull_v and hull_v > 0 else np.nan,
        "cable_per_rg": L / rg if rg > 0 else np.nan,
        "cable_per_span": L / span if span and span > 0 else np.nan,
        "local_density_r5": float(np.median(dens5)),
        "local_density_r10": float(np.median(dens10)),
        **revisit,
        "overdraw_2d_mean": float(np.mean(over)),
        "overdraw_2d_max": float(np.max(over)),
        "elongation": elongation,
        "flatness": flatness,
        "arbor_separation": arbor_sep,
        "arbor_balance": arbor_balance,
        "arbor_centroid_dist_um": d_cc,
        "bridge_fraction": bridge,
    }


def main() -> pd.DataFrame:
    roster = pd.read_csv(C.OUT / "neuron_roster.csv")
    roster["neuron_id"] = roster["neuron_id"].astype(str)
    rows = []
    for i, r in enumerate(roster.itertuples(), 1):
        p = K.swc_path(r.neuron_id, r.source)
        try:
            d = descriptors(p)
        except Exception as e:                       # 單顆失敗不中斷整條 pipeline
            print(f"  ! {r.source}/{r.neuron_id}: {e}")
            d = {}
        if d:
            rows.append({"neuron_id": r.neuron_id, "source": r.source,
                         "group": r.group, "exclusive": r.exclusive, **d})
        if i % 100 == 0:
            print(f"  {i}/{len(roster)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "morphology_metrics.csv", index=False)
    print(df.groupby(["group", "source"])[
        ["cable_length_um", "n_branch_points", "revisit_r16um",
         "overdraw_2d_mean", "branch_per_100um"]].median().round(3))
    return df


if __name__ == "__main__":
    main()
