# %%
# stage1_candidate_matching.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial import cKDTree  # type: ignore

import numpy as np


@dataclass(frozen=True)
class SourceDesc:
    source: str
    centroids: np.ndarray  # (N,3) float32
    ratios2d: np.ndarray   # (N,2) float32 -> [r21, r31]


def load_source(out_dir: str | Path, source: str) -> SourceDesc:
    out_dir = Path(out_dir)

    cent_path = out_dir / f"centroids_{source}.npy"
    ratio_path = out_dir / f"eigvals_ratio_{source}.npy"

    if not cent_path.exists():
        raise FileNotFoundError(f"Missing {cent_path}")
    if not ratio_path.exists():
        raise FileNotFoundError(f"Missing {ratio_path}")

    centroids = np.load(cent_path).astype(np.float32)
    ratios = np.load(ratio_path).astype(np.float32)

    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"{cent_path} invalid shape: {centroids.shape}")

    if ratios.ndim != 2:
        raise ValueError(f"{ratio_path} invalid shape: {ratios.shape}")

    # Accept (N,3)=[r11,r21,r31] OR (N,2)=[r21,r31]
    if ratios.shape[1] == 3:
        ratios2d = ratios[:, 1:3]
    elif ratios.shape[1] == 2:
        ratios2d = ratios
    else:
        raise ValueError(f"{ratio_path} expected (N,3) or (N,2), got {ratios.shape}")

    return SourceDesc(source=source, centroids=centroids, ratios2d=ratios2d)


def candidate_pairs_by_centroid_distance( cent_a: np.ndarray, cent_b: np.ndarray,
    threshold: float,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (ia, ib, dist_centroid) for pairs with centroid distance <= threshold.
    Uses cKDTree if scipy available; otherwise raises with hint.
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


def main():
    ap = argparse.ArgumentParser(description="Stage1: Candidate matching by centroid + (r21,r31) distance")
    ap.add_argument("--fc_dir", required=True, help="Folder containing centroids_FC.npy and eigvals_ratio_FC.npy")
    ap.add_argument("--em_dir", required=True, help="Folder containing centroids_EM.npy and eigvals_ratio_EM.npy")
    ap.add_argument("--centroid_th", type=float, default=100.0, help="Centroid distance threshold")
    ap.add_argument("--ratio_th", type=float, default=0.2, help="(r21,r31) 2D distance threshold")
    ap.add_argument("--out_dir", required=True, help="Folder to write candidate pairs")
    ap.add_argument("--no_kdtree", action="store_true", help="Disable KDTree (not supported currently)")
    args = ap.parse_args()

    fc = load_source(args.fc_dir, "FC")
    em = load_source(args.em_dir, "EM")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) centroid candidate generation
    ia, ib, d_cent = candidate_pairs_by_centroid_distance(
        fc.centroids, em.centroids, threshold=args.centroid_th, use_kdtree=not args.no_kdtree
    )

    # 2) ratio2d filter
    ia2, ib2, d_ratio = filter_pairs_by_ratio2d_distance(
        ia, ib, fc.ratios2d, em.ratios2d, threshold=args.ratio_th
    )

    # Keep centroid distance aligned to ia2/ib2 (recompute quickly; avoids carrying masks around)
    if ia2.size:
        diff = fc.centroids[ia2] - em.centroids[ib2]
        d_cent2 = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    else:
        d_cent2 = np.empty((0,), dtype=np.float32)

    # Save arrays (compact + easy downstream)
    np.save(out_dir / "pairs_FC_EM_ia.npy", ia2)
    np.save(out_dir / "pairs_FC_EM_ib.npy", ib2)
    np.save(out_dir / "pairs_FC_EM_centroid_dist.npy", d_cent2)
    np.save(out_dir / "pairs_FC_EM_ratio2d_dist.npy", d_ratio)

    # Save combined table-like array: [ia, ib, centroid_dist, ratio2d_dist]
    combined = np.empty((ia2.shape[0], 4), dtype=np.float32)
    combined[:, 0] = ia2.astype(np.float32)
    combined[:, 1] = ib2.astype(np.float32)
    combined[:, 2] = d_cent2
    combined[:, 3] = d_ratio
    np.save(out_dir / "pairs_FC_EM.npy", combined)

    print(f"FC: {fc.centroids.shape[0]}  EM: {em.centroids.shape[0]}")
    print(f"centroid_th={args.centroid_th}  ratio_th={args.ratio_th}")
    print(f"after centroid filter: {ia.shape[0]}")
    print(f"after ratio2d filter:  {ia2.shape[0]}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
