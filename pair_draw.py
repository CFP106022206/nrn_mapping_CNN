# %% stage2_pair_views.py (sharded npz, store only neuron IDs + uint8 views)

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


# -----------------------------
# SWC IO
# -----------------------------
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


# -----------------------------
# Tree + Strahler
# -----------------------------
def build_id_to_index(nid: np.ndarray) -> Dict[int, int]:
    return {int(x): i for i, x in enumerate(nid.tolist())}


def build_children(parent: np.ndarray, nid: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, int]]:
    n = nid.shape[0]
    id2idx = build_id_to_index(nid)

    children_lists: List[List[int]] = [[] for _ in range(n)]
    roots = []

    for i in range(n):
        pid = int(parent[i])
        if pid < 0 or pid == int(nid[i]) or pid not in id2idx:
            roots.append(i)
        else:
            pidx = id2idx[pid]
            children_lists[pidx].append(i)

    children = [np.asarray(xs, dtype=np.int32) if xs else np.empty((0,), dtype=np.int32) for xs in children_lists]
    return children, np.asarray(roots, dtype=np.int32), id2idx


def postorder(children: List[np.ndarray], roots: np.ndarray) -> np.ndarray:
    order: List[int] = []
    stack: List[Tuple[int, int]] = []
    for r in roots.tolist():
        stack.append((r, 0))
        while stack:
            node, ci = stack[-1]
            ch = children[node]
            if ci < ch.shape[0]:
                child = int(ch[ci])
                stack[-1] = (node, ci + 1)
                stack.append((child, 0))
            else:
                order.append(node)
                stack.pop()
    return np.asarray(order, dtype=np.int32)


def compute_strahler_node(children: List[np.ndarray], roots: np.ndarray) -> np.ndarray:
    n = len(children)
    s = np.ones((n,), dtype=np.int16)
    for u in postorder(children, roots):
        ch = children[int(u)]
        if ch.size == 0:
            s[int(u)] = 1
            continue
        cs = s[ch]
        m = int(cs.max())
        cnt = int(np.sum(cs == m))
        s[int(u)] = m + 1 if cnt >= 2 else m
    return s


# -----------------------------
# Geometry
# -----------------------------
def rotate_to_ia_axes(xyz: np.ndarray, centroid: np.ndarray, eigvecs_ia: np.ndarray) -> np.ndarray:
    x = (xyz - centroid[None, :]).astype(np.float32, copy=False)
    return (x @ eigvecs_ia).astype(np.float32, copy=False)


def pair_view_bbox_2d(a_uv: np.ndarray, b_uv: np.ndarray, pad: float = 1e-6) -> Tuple[float, float, float, float]:
    uv = np.concatenate([a_uv, b_uv], axis=0)
    umin = float(np.min(uv[:, 0])); umax = float(np.max(uv[:, 0]))
    vmin = float(np.min(uv[:, 1])); vmax = float(np.max(uv[:, 1]))

    if umax - umin < pad:
        mid = 0.5 * (umin + umax); umin, umax = mid - 0.5, mid + 0.5
    if vmax - vmin < pad:
        mid = 0.5 * (vmin + vmax); vmin, vmax = mid - 0.5, mid + 0.5
    return umin, umax, vmin, vmax


# -----------------------------
# Rasterize (Bresenham) + normalize
# -----------------------------
def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    pts: List[Tuple[int, int]] = []
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
    return np.asarray(pts, dtype=np.int32)


def project_and_rasterize(
    xyz_rot: np.ndarray,
    edges: np.ndarray,          # (E,2) child,parent indices
    edge_w: np.ndarray,         # (E,) float32
    grid: int,
    view: int,
    bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    umin, umax, vmin, vmax = bbox
    du = umax - umin
    dv = vmax - vmin

    if view == 0:
        uv = xyz_rot[:, [1, 2]]  # (y,z)
    elif view == 1:
        uv = xyz_rot[:, [0, 2]]  # (x,z)
    elif view == 2:
        uv = xyz_rot[:, [0, 1]]  # (x,y)
    else:
        raise ValueError("view must be 0,1,2")

    u = (uv[:, 0] - umin) / du
    v = (uv[:, 1] - vmin) / dv
    px = np.clip(np.floor(u * (grid - 1) + 0.5), 0, grid - 1).astype(np.int32)
    py = np.clip(np.floor(v * (grid - 1) + 0.5), 0, grid - 1).astype(np.int32)

    img = np.zeros((grid, grid), dtype=np.float32)
    for (c, p), w in zip(edges, edge_w):
        x0, y0 = int(px[c]), int(py[c])
        x1, y1 = int(px[p]), int(py[p])
        pts = bresenham_line(x0, y0, x1, y1)
        img[pts[:, 1], pts[:, 0]] = np.maximum(img[pts[:, 1], pts[:, 0]], float(w))
    return img


def normalize_views(views: np.ndarray, mode: str = "p99") -> np.ndarray:
    v = views.astype(np.float32, copy=False)
    nz = v[v > 0]
    if nz.size == 0:
        return v
    if mode == "max":
        scale = float(nz.max())
    elif mode == "p99":
        scale = float(np.percentile(nz, 99.0))
        if scale <= 0:
            scale = float(nz.max())
    else:
        raise ValueError("mode must be max|p99")
    return np.clip(v / (scale + 1e-12), 0.0, 1.0).astype(np.float32)


def views_to_uint8(views01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(views01 * 255.0), 0, 255).astype(np.uint8)


# -----------------------------
# Cache + edges
# -----------------------------
@dataclass
class NeuronCacheItem:
    swc: Swc
    id2idx: Dict[int, int]
    strahler: np.ndarray  # (N,) int16


def get_cached(
    cache: Dict[str, NeuronCacheItem],
    swc_path: Path,
) -> NeuronCacheItem:
    key = swc_path.as_posix()
    if key in cache:
        return cache[key]

    swc = load_swc_fast(swc_path)
    children, roots, id2idx = build_children(swc.parent, swc.nid)
    s = compute_strahler_node(children, roots)

    item = NeuronCacheItem(swc=swc, id2idx=id2idx, strahler=s)
    cache[key] = item
    return item


def build_edges_and_weights(item: NeuronCacheItem) -> Tuple[np.ndarray, np.ndarray]:
    swc = item.swc
    edges = []
    w = []
    for i in range(swc.nid.shape[0]):
        pid = int(swc.parent[i])
        if pid < 0 or pid == int(swc.nid[i]) or pid not in item.id2idx:
            continue
        pidx = item.id2idx[pid]
        edges.append((i, pidx))
        w.append(int(item.strahler[i]))  # Strahler(child)
    if not edges:
        return np.empty((0, 2), np.int32), np.empty((0,), np.float32)
    return np.asarray(edges, np.int32), np.asarray(w, np.float32)


def render_pair(
    ia: int,
    ib: int,
    *,
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    swc_a: Path,
    swc_b: Path,
    cent_a: np.ndarray,
    cent_b: np.ndarray,
    eigvecs_a: np.ndarray,
    cache: Dict[str, NeuronCacheItem],
    grid: int,
    norm: str,
) -> Tuple[str, str, np.ndarray, np.ndarray]:
    ida = str(ids_a[ia]); idb = str(ids_b[ib])
    pa = swc_a / f"{ida}.swc"
    pb = swc_b / f"{idb}.swc"

    item_a = get_cached(cache, pa)
    item_b = get_cached(cache, pb)

    R = eigvecs_a[ia].astype(np.float32, copy=False)  # (3,3) columns v1,v2,v3
    a_rot = rotate_to_ia_axes(item_a.swc.xyz, cent_a[ia], R)
    b_rot = rotate_to_ia_axes(item_b.swc.xyz, cent_b[ib], R)

    edges_a, w_a = build_edges_and_weights(item_a)
    edges_b, w_b = build_edges_and_weights(item_b)

    va = np.zeros((3, grid, grid), np.float32)
    vb = np.zeros((3, grid, grid), np.float32)

    for view in range(3):
        if view == 0:
            a_uv = a_rot[:, [1, 2]]; b_uv = b_rot[:, [1, 2]]
        elif view == 1:
            a_uv = a_rot[:, [0, 2]]; b_uv = b_rot[:, [0, 2]]
        else:
            a_uv = a_rot[:, [0, 1]]; b_uv = b_rot[:, [0, 1]]

        bbox = pair_view_bbox_2d(a_uv, b_uv)
        if edges_a.size:
            va[view] = project_and_rasterize(a_rot, edges_a, w_a, grid, view, bbox)
        if edges_b.size:
            vb[view] = project_and_rasterize(b_rot, edges_b, w_b, grid, view, bbox)

    va = views_to_uint8(normalize_views(va, mode=norm))
    vb = views_to_uint8(normalize_views(vb, mode=norm))
    return ida, idb, va, vb


def flush_shard(out_dir: Path, shard_id: int, ids_a: List[str], ids_b: List[str], va: List[np.ndarray], vb: List[np.ndarray]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairs_views_{shard_id:05d}.npz"
    np.savez_compressed(
        out_path,
        id_a=np.asarray(ids_a, dtype=object),
        id_b=np.asarray(ids_b, dtype=object),
        views_a=np.stack(va, axis=0).astype(np.uint8),
        views_b=np.stack(vb, axis=0).astype(np.uint8),
    )
    print(f"[save] {out_path.name}  pairs={len(ids_a)}")


def main():
    ap = argparse.ArgumentParser(description="Stage2: render pair-aligned 3-view projections into sharded NPZ (ID + uint8 views).")
    ap.add_argument("--pairs", required=True, help="pairs_FC_EM.npy (cols: ia, ib, ...)")
    ap.add_argument("--ids_a", required=True, help="neuron_ids_FC.npy")
    ap.add_argument("--ids_b", required=True, help="neuron_ids_EM.npy")
    ap.add_argument("--swc_a", required=True, help="SWC folder for source A (FC)")
    ap.add_argument("--swc_b", required=True, help="SWC folder for source B (EM)")
    ap.add_argument("--desc_a", required=True, help="descriptor folder for source A (needs centroids_*.npy, eigvecs_*.npy)")
    ap.add_argument("--desc_b", required=True, help="descriptor folder for source B (needs centroids_*.npy)")
    ap.add_argument("--source_a", default="FC")
    ap.add_argument("--source_b", default="EM")
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--normalize", choices=["max", "p99"], default="p99")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--max_pairs", type=int, default=0, help="debug limit")
    args = ap.parse_args()

    pairs = np.load(args.pairs).astype(np.float32, copy=False)
    ia = pairs[:, 0].astype(np.int32)
    ib = pairs[:, 1].astype(np.int32)
    if args.max_pairs and args.max_pairs > 0:
        ia = ia[: args.max_pairs]
        ib = ib[: args.max_pairs]

    ids_a = np.load(args.ids_a, allow_pickle=True)
    ids_b = np.load(args.ids_b, allow_pickle=True)

    desc_a = Path(args.desc_a)
    desc_b = Path(args.desc_b)
    cent_a = np.load(desc_a / f"centroids_{args.source_a}.npy").astype(np.float32, copy=False)
    eig_a = np.load(desc_a / f"eigvecs_{args.source_a}.npy").astype(np.float32, copy=False)

    # 檢查eigvec是否為正交矩陣，否則後續旋轉會有問題
    G = eig_a[0].T @ eig_a[0]
    if not np.allclose(G, np.eye(3), atol=1e-3):
        raise ValueError("Eigvecs not orthonormal; check eigvecs convention / ordering.")

    cent_b = np.load(desc_b / f"centroids_{args.source_b}.npy").astype(np.float32, copy=False)

    cache: Dict[str, NeuronCacheItem] = {}
    out_dir = Path(args.out_dir)

    buf_ida: List[str] = []
    buf_idb: List[str] = []
    buf_va: List[np.ndarray] = []
    buf_vb: List[np.ndarray] = []

    shard_id = 0
    total = len(ia)

    for k, (i, j) in enumerate(zip(ia.tolist(), ib.tolist()), start=1):
        try:
            ida, idb, va, vb = render_pair(
                i, j,
                ids_a=ids_a,
                ids_b=ids_b,
                swc_a=Path(args.swc_a),
                swc_b=Path(args.swc_b),
                cent_a=cent_a,
                cent_b=cent_b,
                eigvecs_a=eig_a,
                cache=cache,
                grid=args.grid,
                norm=args.normalize,
            )
            buf_ida.append(ida)
            buf_idb.append(idb)
            buf_va.append(va)
            buf_vb.append(vb)

        except Exception as e:
            print(f"[warn] pair {k}/{total} (ia={i}, ib={j}) failed: {repr(e)}")

        if len(buf_ida) >= args.shard_size:
            flush_shard(out_dir, shard_id, buf_ida, buf_idb, buf_va, buf_vb)
            shard_id += 1
            buf_ida, buf_idb, buf_va, buf_vb = [], [], [], []

        if k % 200 == 0:
            print(f"[{k}/{total}] done, cache={len(cache)} neurons")

    if buf_ida:
        flush_shard(out_dir, shard_id, buf_ida, buf_idb, buf_va, buf_vb)

    print(f"All done. Shards saved under: {out_dir}")


if __name__ == "__main__":
    main()
