# %%
# stage1_candidate_matching.py
'''
python candidate_matching.py
    --fc_dir data/descriptors_FC
    --em_dir data/descriptors_EM
    --out_dir data/pairs_label
    --centroid_th 100
    --ratio_th 0.4
'''


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial import cKDTree  # type: ignore

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceDesc:
    source: str
    centroids: np.ndarray  # (N,3) float32
    ratios2d: np.ndarray   # (N,2) float32 -> [r21, r31]
    eigvecs: np.ndarray    # (N,3,3) float32
    neuron_ids: np.ndarray    # (N,) str

def load_source(out_dir: str | Path, source: str) -> SourceDesc:
    out_dir = Path(out_dir)

    cent_path = out_dir / f"centroids_{source}.npy"
    ratio_path = out_dir / f"eigvals_ratio_{source}.npy"
    eigvecs_path = out_dir / f"eigvecs_{source}.npy"
    neuron_ids_path = out_dir / f"neuron_ids_{source}.npy"
    if not cent_path.exists():
        raise FileNotFoundError(f"Missing {cent_path}")
    if not ratio_path.exists():
        raise FileNotFoundError(f"Missing {ratio_path}")
    if not eigvecs_path.exists():
        raise FileNotFoundError(f"Missing {eigvecs_path}")
    if not neuron_ids_path.exists():
        raise FileNotFoundError(f"Missing {neuron_ids_path}")

    centroids = np.load(cent_path).astype(np.float32)
    ratios = np.load(ratio_path).astype(np.float32)
    eigvecs = np.load(eigvecs_path).astype(np.float32)
    neuron_ids = np.load(neuron_ids_path, allow_pickle=True)

    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"{cent_path} invalid shape: {centroids.shape}")

    if ratios.ndim != 2:
        raise ValueError(f"{ratio_path} invalid shape: {ratios.shape}")

    if eigvecs.ndim != 3 or eigvecs.shape[1:] != (3, 3):
        raise ValueError(f"{eigvecs_path} invalid shape: {eigvecs.shape}")

    # Accept (N,3)=[r11,r21,r31] OR (N,2)=[r21,r31]
    if ratios.shape[1] == 3:
        ratios2d = ratios[:, 1:3]
    elif ratios.shape[1] == 2:
        ratios2d = ratios
    else:
        raise ValueError(f"{ratio_path} expected (N,3) or (N,2), got {ratios.shape}")

    return SourceDesc(source=source, centroids=centroids, ratios2d=ratios2d, eigvecs=eigvecs, neuron_ids=neuron_ids)

# 利用質心距離進行第一步過濾
def candidate_pairs_by_centroid_distance(cent_a: np.ndarray, cent_b: np.ndarray,
    threshold: float,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (ia, ib, dist_centroid) for pairs with centroid distance <= threshold.
    Uses cKDTree for fast spatial search.
    """

    tree = cKDTree(cent_b.astype(np.float64, copy=False))
    neigh = tree.query_ball_point(cent_a.astype(np.float64, copy=False), r=float(threshold))

    counts = np.fromiter((len(x) for x in neigh), dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    ia = np.repeat(np.arange(len(neigh), dtype=np.int32), counts.astype(np.int32))

    ib = np.empty((total,), dtype=np.int32)
    pos = 0
    for xs in neigh:
        n = len(xs)
        if n:
            ib[pos : pos + n] = np.asarray(xs, dtype=np.int32)
            pos += n

    d = cent_a[ia] - cent_b[ib]
    dist = np.sqrt(np.einsum("ij,ij->i", d, d)).astype(np.float32)

    # numeric safety
    m = dist <= float(threshold)
    return ia[m], ib[m], dist[m]

# 第二步過濾：歸一化轉動慣量, 利用ratio2d坐标之间的距离判断相似性
def filter_pairs_by_ratio2d_distance(
    ia: np.ndarray,
    ib: np.ndarray,
    ratios2d_a: np.ndarray,  # (NA,2) [r21,r31]
    ratios2d_b: np.ndarray,  # (NB,2) [r21,r31]
    threshold: float,
    chunk: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep pairs whose Euclidean distance in (r21,r31) 2D <= threshold.
    Returns (ia2, ib2, dist_ratio2d).
    """
    n = ia.shape[0]
    if n == 0:
        return ia, ib, np.empty((0,), dtype=np.float32)

    keep_ia = []
    keep_ib = []
    keep_dr = []

    thr = float(threshold)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        a = ratios2d_a[ia[s:e]]  # (m,2)
        b = ratios2d_b[ib[s:e]]  # (m,2)

        diff = a - b
        dr = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
        m = dr <= thr

        if np.any(m):
            keep_ia.append(ia[s:e][m])
            keep_ib.append(ib[s:e][m])
            keep_dr.append(dr[m])

    if not keep_ia:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    return np.concatenate(keep_ia), np.concatenate(keep_ib), np.concatenate(keep_dr)

# 第三步過濾：判斷方向性是否明顯(r21-r31)，符合條件者計算cosine similarity
def filter_pairs_by_orientation_rod_disk(
    ia: np.ndarray,
    ib: np.ndarray,
    ratios2d_a: np.ndarray,   # (NA,2) [r21,r31]
    ratios2d_b: np.ndarray,   # (NB,2) [r21,r31]
    eigvecs_a: np.ndarray,    # (NA,3,3) columns v1,v2,v3 (λ1>=λ2>=λ3)
    eigvecs_b: np.ndarray,    # (NB,3,3)
    *,
    # rod thresholds
    rod_r31_max: float = 0.35,  # 判斷是否是rod-like，需要r21接近1且r31接近0
    rod_gap_min: float = 0.4,       # r21 - r31
    # 35 而非 30：v3 跨 modality 有系統性抖動。用 D1-D6 人工標註 (label>=0.5) 量測，
    # 被 gate 管到的 361 對真 pair 角度 p95=21.5°、p99=33.4°；30° 會切掉 9 對，
    # 其中 6 對信心度 >=0.8（含 2 對 1.0），且已確認它們真的從 top5 輸出消失。
    # 35° 讓高信心損失歸零，代價只是候選池少砍 3.3pp（-42.2% -> -38.9%）。
    rod_angle_th_deg: float = 35.0,
    # disk thresholds
    # 盤狀要比的軸是 v1 (最大慣量 = 盤面法線)，它良好定義的條件是 λ1 與 λ2 分開，
    # 即 1 - r21 >= disk_gap_min，與 rod 用 r21 - r31 判斷 v3 是對稱的。
    # 理想薄盤落在 (r21, r31) = (0.5, 0.5)；慣量張量的三角不等式 λ2 + λ3 >= λ1
    # 保證 r21 + r31 >= 1，因此 r21 >= 0.5，1 - r21 的上限就是 0.5。
    # (舊版寫 r21 <= 0.45 落在可及區域之外，此分支永遠不會啟動。)
    disk_gap_min: float = 0.30,     # 1 - r21
    # disk 維持 30°：標註中 disk 正樣本角度最大只有 26.7°（高信心者 15.5°），
    # 30° 未殺到任何一對；且 disk pair 僅佔配對約 0.5%，門檻高低對候選池影響 <0.1pp。
    disk_angle_th_deg: float = 30.0,
    chunk: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Conditional orientation gating with two regimes:
    - rod-like pairs: compare v3 (λ3 axis)
    - disk-like pairs: compare v1 (λ1 axis, the normal of the plane)
    Each regime requires the eigenvalue pair bracketing the compared axis to be
    separated, so that the axis itself is stable: rod needs r21 - r31 to be
    large, disk needs 1 - r21 to be large.
    Only enabled when BOTH sides are in the same regime.
    Others pass through without orientation filtering.

    Returns:
        ia3, ib3,
        angle_deg (NaN if not enabled),
        enabled_type (uint8): 0=not enabled, 1=rod, 2=disk
    """
    n = ia.shape[0]
    if n == 0:
        return ia, ib, np.empty((0,), np.float32), np.empty((0,), np.uint8)

    thr_cos_rod = float(np.cos(np.deg2rad(rod_angle_th_deg)))
    thr_cos_disk = float(np.cos(np.deg2rad(disk_angle_th_deg)))

    keep_ia, keep_ib, keep_ang, keep_type = [], [], [], []

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ia_s = ia[s:e]
        ib_s = ib[s:e]

        ra = ratios2d_a[ia_s]
        rb = ratios2d_b[ib_s]

        r21_a, r31_a = ra[:, 0], ra[:, 1]
        r21_b, r31_b = rb[:, 0], rb[:, 1]

        gap_a = r21_a - r31_a
        gap_b = r21_b - r31_b

        # per-item regime
        rod_a = (r31_a <= rod_r31_max) & (gap_a >= rod_gap_min)
        rod_b = (r31_b <= rod_r31_max) & (gap_b >= rod_gap_min)

        # 盤狀：λ1 與 λ2 分開，v1 (盤面法線) 才是良好定義的方向。
        # r31 >= 1 - r21，所以 r21 夠小時面內的近似各向同性是自動成立的。
        disk_a = (1.0 - r21_a) >= disk_gap_min
        disk_b = (1.0 - r21_b) >= disk_gap_min

        enable_rod = rod_a & rod_b
        enable_disk = disk_a & disk_b

        enabled = enable_rod | enable_disk
        keep = ~enabled  # default: keep non-enabled
        ang = np.full((e - s,), np.nan, dtype=np.float32)
        typ = np.zeros((e - s,), dtype=np.uint8)

        # --- rod: compare vector3 ---
        if np.any(enable_rod):
            va = eigvecs_a[ia_s[enable_rod], :, 2].astype(np.float32, copy=False)
            vb = eigvecs_b[ib_s[enable_rod], :, 2].astype(np.float32, copy=False)
            dot = np.abs(np.einsum("ij,ij->i", va, vb)).astype(np.float32)  # 取絕對值，避免eigen vector方向相反
            na = np.sqrt(np.einsum("ij,ij->i", va, va)).astype(np.float32)
            nb = np.sqrt(np.einsum("ij,ij->i", vb, vb)).astype(np.float32)
            cos = dot / (na * nb + 1e-12)
            k = cos >= thr_cos_rod

            cos_clip = np.clip(cos, -1.0, 1.0)
            ang_enable = np.rad2deg(np.arccos(cos_clip)).astype(np.float32)

            idx = np.flatnonzero(enable_rod)
            ang[idx] = ang_enable
            typ[idx] = 1
            keep[idx] = k

        # --- disk: compare vector1 ---
        if np.any(enable_disk):
            va = eigvecs_a[ia_s[enable_disk], :, 0].astype(np.float32, copy=False)
            vb = eigvecs_b[ib_s[enable_disk], :, 0].astype(np.float32, copy=False)
            dot = np.abs(np.einsum("ij,ij->i", va, vb)).astype(np.float32)
            na = np.sqrt(np.einsum("ij,ij->i", va, va)).astype(np.float32)
            nb = np.sqrt(np.einsum("ij,ij->i", vb, vb)).astype(np.float32)
            cos = dot / (na * nb + 1e-12)
            k = cos >= thr_cos_disk

            cos_clip = np.clip(cos, -1.0, 1.0)
            ang_enable = np.rad2deg(np.arccos(cos_clip)).astype(np.float32)

            idx = np.flatnonzero(enable_disk)
            ang[idx] = ang_enable
            typ[idx] = 2
            keep[idx] = k

        if np.any(keep):
            keep_ia.append(ia_s[keep])
            keep_ib.append(ib_s[keep])
            keep_ang.append(ang[keep])
            keep_type.append(typ[keep])

    if not keep_ia:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

    return (
        np.concatenate(keep_ia).astype(np.int32, copy=False),
        np.concatenate(keep_ib).astype(np.int32, copy=False),
        np.concatenate(keep_ang).astype(np.float32, copy=False),
        np.concatenate(keep_type).astype(np.uint8, copy=False),
    )


def run_matching(
    fc_dir: str | Path = "data/descriptors_FC/",
    em_dir: str | Path = "data/descriptors_EM/",
    out_dir: str | Path = "data/pairs_label/",
    centroid_th: float = 100.0,
    ratio_th: float = 0.4,
) -> Path:
    fc = load_source(fc_dir, "FC")
    em = load_source(em_dir, "EM")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) centroid candidate generation (using KDTree)
    ia, ib, _ = candidate_pairs_by_centroid_distance(
        fc.centroids, em.centroids, threshold=centroid_th
    )

    # 2) ratio2d filter
    ia2, ib2, _ = filter_pairs_by_ratio2d_distance(
        ia, ib, fc.ratios2d, em.ratios2d, threshold=ratio_th
    )

    ia3, ib3, _, _ = filter_pairs_by_orientation_rod_disk(
        ia2,
        ib2,
        fc.ratios2d,
        em.ratios2d,
        fc.eigvecs,
        em.eigvecs,
        rod_angle_th_deg=35.0,
        disk_angle_th_deg=30.0,
    )

    # Output CSV with only neuron IDs
    fc_ids = np.asarray(fc.neuron_ids[ia3]).astype(str)
    em_ids = np.asarray(em.neuron_ids[ib3]).astype(str)
    out_df = pd.DataFrame({"fc_id": fc_ids, "em_id": em_ids})
    out_df = out_df.drop_duplicates(["fc_id", "em_id"], keep="first")

    out_csv = out_dir / "pairs_FC_EM.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"FC: {fc.centroids.shape[0]}  EM: {em.centroids.shape[0]}")
    print(f"centroid_th={centroid_th}  ratio_th={ratio_th}")
    print(f"after centroid filter: {ia.shape[0]}")
    print(f"after ratio2d filter:  {ia2.shape[0]}")
    print(f"after orientation filter: {ia3.shape[0]}")
    print(f"saved: {out_csv}")
    return out_csv


# %%
def main():
    ap = argparse.ArgumentParser(description="Stage1: Candidate matching by centroid + (r21,r31) distance")
    ap.add_argument("--fc_dir", default='data/descriptors_FC/', help="Folder containing centroids_FC.npy and eigvals_ratio_FC.npy")
    ap.add_argument("--em_dir", default='data/descriptors_EM/', help="Folder containing centroids_EM.npy and eigvals_ratio_EM.npy")
    ap.add_argument("--centroid_th", type=float, default=100.0, help="Centroid distance threshold")
    ap.add_argument("--ratio_th", type=float, default=0.4, help="(r21,r31) 2D distance threshold")
    ap.add_argument("--out_dir", default='data/pairs_label/', help="Folder to write candidate pairs")
    args = ap.parse_args()

    run_matching(
        fc_dir=args.fc_dir,
        em_dir=args.em_dir,
        out_dir=args.out_dir,
        centroid_th=args.centroid_th,
        ratio_th=args.ratio_th,
    )


if __name__ == "__main__":
    main()
