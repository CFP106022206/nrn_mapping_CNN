# %% standard_draw.py - Render single neuron 3-view projections
# 使用標準腦座標畫圖，不需要讀取 descriptor 中的 centroid 和 eigvecs
# 將每一個神經元的 SWC 檔案單獨畫成一個三視圖
# 視角固定為標準腦座標的三個垂直方向（XYZ）
# 圖片大小不固定，但保證每個圖的 scale 一樣（可通過 scale_um_per_px 參數調整）

"""
使用示例：
    python standard_draw.py \
        --swc_dir ./data/SWC/FC \
        --neuron_list neuron_ids.txt \
        --scale_um_per_px 5.0 \
        --normalize p99 \
        --output_dir ./standard_views/ \
        --format npz

或使用 .csv 格式的神經元配對列表：
    python3 standard_draw.py \
        --swc_dir data/SWC/FC \
        --neuron_list data/pairs_label/D1-D6_total_conf.csv \
        --csv_id_col fc_id \
        --scale_um_per_px 5.0 \
        --output_dir data/standard_views/FC \
        --format npz \
        --skip_existing \
        --export_missing_list data/pairs_label/missing_fc_swc_ids.txt
    
    python3 standard_draw.py \
        --swc_dir data/SWC/EM \
        --neuron_list data/pairs_label/D1-D6_total_conf.csv \
        --csv_id_col em_id \
        --scale_um_per_px 5.0 \
        --output_dir data/standard_views/EM \
        --format npz \
        --skip_existing \
        --export_missing_list data/pairs_label/missing_em_swc_ids.txt
參數說明：
  --swc_dir        : SWC 檔案所在目錄
  --neuron_list    : 神經元 ID 列表
  --csv_id_col     : 當 neuron_list 是 .csv 時，指定哪一列是神經元 ID（例如 fc_id 或 em_id）。如果留空，會嘗試從 swc_dir 名稱推斷
  --scale_um_per_px: 微米/像素，用於控制圖像大小。值越小圖越大（默認 1.0）
  --normalize      : 歸一化方式："max" 或 "p99"（默認 p99）
  --output_dir     : 輸出目錄（默認 ./standard_views/）
  --format         : 輸出格式："npz" 或 "png"（默認 npz）
  --max_neurons    : 處理的最大神經元數量，用於調試（0=全部）
  --skip_existing  : 當輸出文件已存在時跳過渲染（只處理缺失的）
  --export_missing_list : 輸出缺失的神經元 ID 列表(在這裡指定輸出路徑)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
from PIL import Image

from swc_util import Swc, load_swc_fast


def _stable_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _read_ids_from_txt(path: Path) -> List[str]:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        ids = [line.strip() for line in f if line.strip()]
    return _stable_unique(ids)


def _read_ids_from_npy(path: Path) -> List[str]:
    arr = np.load(path, allow_pickle=True)
    # 支援 numpy array 或 python list
    if isinstance(arr, np.ndarray):
        ids = [str(x).strip() for x in arr.tolist()]
    else:
        ids = [str(x).strip() for x in list(arr)]
    ids = [x for x in ids if x]
    return _stable_unique(ids)


def _infer_csv_id_col(swc_dir: Path, header: List[str]) -> Optional[str]:
    cols = {c.strip() for c in header}
    name_u = swc_dir.name.upper()
    if 'FC' in name_u and 'fc_id' in cols:
        return 'fc_id'
    if 'EM' in name_u and 'em_id' in cols:
        return 'em_id'
    return None


def _read_ids_from_pairs_csv(path: Path, *, swc_dir: Path, id_col: Optional[str]) -> List[str]:
    """从 pairs CSV 读取 neuron ids，并做 stable unique。

    期望 CSV 至少包含：fc_id / em_id（或你指定的列）。
    """
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        header = [h.strip() for h in reader.fieldnames]

        chosen_col = (id_col or '').strip() or _infer_csv_id_col(swc_dir, header)
        if not chosen_col:
            # 如果 header 里只有一个 *_id，就用它；否则要求显式指定
            id_like = [c for c in header if c.lower().endswith('_id')]
            if len(id_like) == 1:
                chosen_col = id_like[0]
            else:
                raise ValueError(
                    f"CSV input requires --csv_id_col. Header={header}. "
                    f"Example: --csv_id_col fc_id (for FC) or --csv_id_col em_id (for EM)"
                )

        if chosen_col not in header:
            raise KeyError(f"Column '{chosen_col}' not found in CSV header: {header}")

        ids: List[str] = []
        for row in reader:
            v = row.get(chosen_col, '')
            if v is None:
                continue
            s = str(v).strip()
            if s:
                ids.append(s)

    return _stable_unique(ids)


def load_neuron_ids(
    neuron_list_path: Path,
    *,
    swc_dir: Path,
    csv_id_col: Optional[str],
) -> List[str]:
    """统一入口：从 txt / npy / pairs csv 读取 neuron id 名单（并去重）。"""
    suf = neuron_list_path.suffix.lower()
    if suf == '.npy':
        return _read_ids_from_npy(neuron_list_path)
    if suf == '.csv':
        return _read_ids_from_pairs_csv(neuron_list_path, swc_dir=swc_dir, id_col=csv_id_col)
    return _read_ids_from_txt(neuron_list_path)


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
def compute_bbox_3d(xyz: np.ndarray, margin: float = 0.08) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    計算單個神經元的 3D bounding box，使用標準腦座標（不旋轉）。
    
    Args:
        xyz: (N, 3) float32 座標
        margin: 邊界邊距比例
    
    Returns:
        ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    """
    xmin, ymin, zmin = xyz.min(axis=0)
    xmax, ymax, zmax = xyz.max(axis=0)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    side = max(dx, dy, dz, 1e-3)
    side *= (1.0 + margin)

    half = 0.5 * side

    return (
        (cx - half, cx + half),
        (cy - half, cy + half),
        (cz - half, cz + half),
    )


def unit_cube_bbox_from_range(xyz_min: np.ndarray, xyz_max: np.ndarray, margin: float = 0.08) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    从 min/max 坐标计算正方形 bbox
    """
    xmin, ymin, zmin = xyz_min
    xmax, ymax, zmax = xyz_max

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    side = max(dx, dy, dz, 1e-3)
    side *= (1.0 + margin)

    half = 0.5 * side

    return (
        (cx - half, cx + half),
        (cy - half, cy + half),
        (cz - half, cz + half),
    )


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
    xyz: np.ndarray,
    edges: np.ndarray,          # (E,2) child,parent indices
    edge_w: np.ndarray,         # (E,) float32
    scale_um_per_px: float,     # 微米/像素，用于计算grid大小
    view: int,
    bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    将 3D 坐标投影到 2D 并光栅化
    
    Args:
        xyz: (N, 3) float32 坐标
        edges: (E, 2) int32，每行是 (child_idx, parent_idx)
        edge_w: (E,) float32，边的权重
        scale_um_per_px: 微米/像素（大的值=小图，小值=大图）
        view: 0=YZ, 1=XZ, 2=XY
        bbox: (umin, umax, vmin, vmax)
    
    Returns:
        (grid, grid) float32 图像
    """
    umin, umax, vmin, vmax = bbox
    du = umax - umin
    dv = vmax - vmin

    # 根据 bbox 大小和 scale 参数计算 grid 大小
    grid = max(int(np.ceil(du / scale_um_per_px)), int(np.ceil(dv / scale_um_per_px)), 10)
    grid = min(grid, 1024)  # 限制最大size

    if view == 0:      # YZ
        uv = xyz[:, [1, 2]]  # (y,z)
    elif view == 1:    # XZ
        uv = xyz[:, [0, 2]]  # (x,z)
    elif view == 2:    # XY
        uv = xyz[:, [0, 1]]  # (x,y)
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


def render_single(
    nid: int | str,
    *,
    swc_path: Path,
    cache: Dict[str, NeuronCacheItem],
    scale_um_per_px: float = 1.0,  # 微米/像素
    norm: str = "p99",
) -> Tuple[str, np.ndarray, int]:
    """
    单个神经元的三视图渲染
    
    Args:
        nid: 神经元 ID
        swc_path: SWC 文件路径
        cache: 缓存
        scale_um_per_px: 微米/像素（用于控制图像大小）
        norm: 归一化模式 "max" 或 "p99"
    
    Returns:
        (nid_str, views_uint8, grid_size)
        views_uint8: (3, H, W) uint8 图像
    """
    nid_str = str(nid)
    swc_p = swc_path / f"{nid_str}.swc"

    if not swc_p.exists():
        raise FileNotFoundError(f"SWC file not found: {swc_p}")

    item = get_cached(cache, swc_p)
    xyz = item.swc.xyz

    edges, w = build_edges_and_weights(item)

    # 计算 bbox（使用标准腦座標，不旋轉）
    (xb, yb, zb) = compute_bbox_3d(xyz, margin=0.08)

    # 三视图：0=YZ, 1=XZ, 2=XY
    views = []
    grid_sizes = []

    for view in range(3):
        if view == 0:      # YZ
            bbox = (yb[0], yb[1], zb[0], zb[1])
        elif view == 1:    # XZ
            bbox = (xb[0], xb[1], zb[0], zb[1])
        else:              # XY
            bbox = (xb[0], xb[1], yb[0], yb[1])

        if edges.size:
            img = project_and_rasterize(xyz, edges, w, scale_um_per_px, view, bbox)
        else:
            img = np.zeros((10, 10), dtype=np.float32)
        
        # 记录grid大小
        grid_sizes.append(img.shape[0])
        
        # 单个图像的归一化和转换
        nz = img[img > 0]
        if nz.size > 0:
            if norm == "max":
                scale = float(nz.max())
            elif norm == "p99":
                scale = float(np.percentile(nz, 99.0))
                if scale <= 0:
                    scale = float(nz.max())
            else:
                scale = 1.0
            img_norm = np.clip(img / (scale + 1e-12), 0.0, 1.0)
        else:
            img_norm = np.zeros_like(img, dtype=np.float32)
        
        img_uint8 = np.clip(np.round(img_norm * 255.0), 0, 255).astype(np.uint8)
        views.append(img_uint8)

    # 所有view使用相同grid大小（取最大值）
    max_grid = max(grid_sizes)
    views_resized = []
    for v in views:
        if v.shape[0] != max_grid:
            # 使用 PIL 缩放
            img_pil = Image.fromarray(v, mode='L')
            img_pil_resized = img_pil.resize((max_grid, max_grid), Image.NEAREST)
            views_resized.append(np.asarray(img_pil_resized, dtype=np.uint8))
        else:
            views_resized.append(v)

    views_final = np.stack(views_resized, axis=0).astype(np.uint8)
    return nid_str, views_final, max_grid


def save_views(
    out_dir: Path,
    nid: str,
    views: np.ndarray,
    grid_size: int,
    format: str = 'npz',
    file_tag: str = "",
) -> None:
    """
    保存三视图
    
    Args:
        out_dir: 输出目录
        nid: 神经元 ID
        views: (3, H, W) uint8 数组
        grid_size: grid 大小
        format: 'npz' 或 'png'（分别保存三个png）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = str(file_tag).strip()
    if tag and not tag.startswith('_'):
        tag = '_' + tag

    if format.lower() == 'npz':
        out_path = out_dir / f"{nid}_views{tag}.npz"
        np.savez_compressed(
            out_path,
            nid=np.str_(nid),
            views=views.astype(np.uint8),
            grid_size=np.int32(grid_size),
        )
        print(f"[save] {out_path.name}")
    
    elif format.lower() == 'png':
        for i in range(3):
            out_path = out_dir / f"{nid}_view_{i}{tag}.png"
            im = Image.fromarray(views[i], mode='L')
            im.save(out_path)
        print(f"[save] {nid}_view_*{tag}.png (grid={grid_size})")


def outputs_exist(out_dir: Path, nid: str, format: str, file_tag: str = "") -> bool:
    fmt = format.lower()
    tag = str(file_tag).strip()
    if tag and not tag.startswith('_'):
        tag = '_' + tag
    if fmt == 'npz':
        return (out_dir / f"{nid}_views{tag}.npz").exists()
    if fmt == 'png':
        return all((out_dir / f"{nid}_view_{i}{tag}.png").exists() for i in range(3))
    raise ValueError(f"Unknown format: {format}")


def run_standard_draw(
    swc_dir: str | Path,
    neuron_list: str | Path,
    output_dir: str | Path = './standard_views/',
    *,
    csv_id_col: str | None = None,
    export_unique_list: str | Path | None = None,
    scale_um_per_px: float = 1.0,
    normalize: str = "p99",
    format: str = "npz",
    skip_existing: bool = False,
    export_missing_list: str | Path | None = None,
    max_neurons: int = 0,
) -> None:
    swc_dir = Path(swc_dir)

    neuron_list_path = Path(neuron_list)
    csv_col = csv_id_col.strip() if csv_id_col else None
    neuron_ids = load_neuron_ids(neuron_list_path, swc_dir=swc_dir, csv_id_col=csv_col)

    if export_unique_list:
        out_list = Path(export_unique_list)
        out_list.parent.mkdir(parents=True, exist_ok=True)
        out_list.write_text("\n".join(neuron_ids) + "\n", encoding='utf-8')
        print(f"[save] unique neuron list -> {out_list}  (n={len(neuron_ids)})")

    if max_neurons > 0:
        neuron_ids = neuron_ids[:max_neurons]

    cache: Dict[str, NeuronCacheItem] = {}
    out_dir = Path(output_dir)

    total = len(neuron_ids)
    success_count = 0
    skip_count = 0
    missing_count = 0
    fail_count = 0
    missing_ids: List[str] = []

    for k, nid in enumerate(neuron_ids, start=1):
        try:
            nid_str_pre = str(nid).strip()

            swc_file = swc_dir / f"{nid_str_pre}.swc"
            if not swc_file.exists():
                missing_count += 1
                missing_ids.append(nid_str_pre)
                continue

            if skip_existing and outputs_exist(out_dir, nid_str_pre, format, file_tag=""):
                skip_count += 1
                continue

            nid_str, views, grid_size = render_single(
                nid,
                swc_path=swc_dir,
                cache=cache,
                scale_um_per_px=scale_um_per_px,
                norm=normalize,
            )
            save_views(out_dir, nid_str, views, grid_size, format=format, file_tag="")
            success_count += 1

        except Exception as e:
            print(f"[warn] neuron {k}/{total} (id={nid}) failed: {repr(e)}")
            fail_count += 1

        if k % 50 == 0:
            print(
                f"[{k}/{total}] processed, success={success_count}, skipped={skip_count}, missing={missing_count}, "
                f"fail={fail_count}, cache_size={len(cache)}"
            )

    print(f"\nAll done. Output: {out_dir}")
    print(f"Success: {success_count}, Skipped: {skip_count}, Missing SWC: {missing_count}, Failed: {fail_count}")

    if missing_ids:
        missing_unique = _stable_unique(missing_ids)
        print(f"Missing unique neuron ids: {len(missing_unique)}")
        print("Example missing ids (top 20):", missing_unique[:20])

        if export_missing_list:
            out_missing = Path(export_missing_list)
            out_missing.parent.mkdir(parents=True, exist_ok=True)
            out_missing.write_text("\n".join(missing_unique) + "\n", encoding='utf-8')
            print(f"[save] missing swc list -> {out_missing}")


def main():
    ap = argparse.ArgumentParser(description="Render single neuron 3-view projections using standard brain coordinates (XYZ).")
    ap.add_argument("--swc_dir", required=True, help="SWC file directory")
    ap.add_argument(
        "--neuron_list",
        required=True,
        help="Neuron ID list: .txt(one per line) / .npy / pairs .csv (e.g. D1-D6_total_conf.csv)",
    )
    ap.add_argument(
        "--csv_id_col",
        default="",
        help="When --neuron_list is a .csv, choose which column to use (e.g. fc_id or em_id). If empty, try infer from --swc_dir name.",
    )
    ap.add_argument(
        "--export_unique_list",
        default="",
        help="Optional: export the unique neuron id list to this path (.txt).",
    )
    ap.add_argument("--scale_um_per_px", type=float, default=1.0, help="Micrometers per pixel (controls image size)")
    ap.add_argument("--normalize", choices=["max", "p99"], default="p99")
    ap.add_argument("--output_dir", default='./standard_views/')
    ap.add_argument("--format", choices=["npz", "png"], default="npz", help="Output format")
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip rendering when output file(s) already exist (only render missing).",
    )
    ap.add_argument(
        "--export_missing_list",
        default="",
        help="Optional: export missing SWC neuron ids to this path (.txt).",
    )
    ap.add_argument("--max_neurons", type=int, default=0, help="Debug limit (0=all)")
    args = ap.parse_args()

    run_standard_draw(
        swc_dir=args.swc_dir,
        neuron_list=args.neuron_list,
        output_dir=args.output_dir,
        csv_id_col=args.csv_id_col or None,
        export_unique_list=args.export_unique_list or None,
        scale_um_per_px=args.scale_um_per_px,
        normalize=args.normalize,
        format=args.format,
        skip_existing=args.skip_existing,
        export_missing_list=args.export_missing_list or None,
        max_neurons=args.max_neurons,
    )


# %%
# 快速測試函數
def test_single_neuron(
    swc_file: str | Path,
    scale_um_per_px: float = 1.0,
    norm: str = "p99",
    output_dir: str | Path = "./test_output/",
    show_plot: bool = True,
) -> None:
    """
    快速測試單個 SWC 文件
    
    Args:
        swc_file: SWC 文件完整路徑或相對路徑
        scale_um_per_px: 微米/像素
        norm: 歸一化模式 ("max" 或 "p99")
        output_dir: 輸出目錄
        show_plot: 是否顯示 matplotlib 圖表
    
    Example:
        # 測試方式1：使用檔案路徑
        test_single_neuron("./data/SWC/FC/12345.swc", scale_um_per_px=1.0)
        
        # 測試方式2：使用相對路徑
        test_single_neuron("data/SWC/EM/67890.swc", output_dir="./my_test/")
    """
    import matplotlib.pyplot as plt
    
    swc_file = Path(swc_file)
    output_dir = Path(output_dir)
    
    if not swc_file.exists():
        print(f"❌ SWC file not found: {swc_file}")
        return
    
    print(f"📦 Testing: {swc_file.name}")
    print(f"   Scale: {scale_um_per_px} μm/px, Norm: {norm}")
    
    try:
        cache: Dict[str, NeuronCacheItem] = {}
        
        # 從檔案名提取神經元 ID
        nid = swc_file.stem
        
        # 渲染三視圖
        nid_str, views, grid_size = render_single(
            nid,
            swc_path=swc_file.parent,
            cache=cache,
            scale_um_per_px=scale_um_per_px,
            norm=norm,
        )
        
        print(f"✅ Success!")
        print(f"   Neuron ID: {nid_str}")
        print(f"   Grid size: {grid_size}×{grid_size}")
        print(f"   Views shape: {views.shape}")
        print(f"   Cache size: {len(cache)}")
        
        # 保存結果
        save_views(output_dir, nid_str, views, grid_size, format="npz")
        print(f"   Saved to: {output_dir}/{nid_str}_views.npz")
        
        # 顯示圖表
        if show_plot:
            try:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                titles = ["YZ (view 0)", "XZ (view 1)", "XY (view 2)"]
                for i in range(3):
                    axes[i].imshow(views[i], cmap='magma')
                    axes[i].set_title(titles[i])
                    axes[i].axis('off')
                fig.suptitle(f"Neuron {nid_str} (grid={grid_size}×{grid_size})", fontsize=14)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"   ⚠️  Could not display plot: {e}")
        
    except Exception as e:
        print(f"❌ Error: {repr(e)}")
        import traceback
        traceback.print_exc()


# %%
if __name__ == "__main__":
    # 如需快速測試單個 SWC，可取消下面註解並填入正確路徑
    # test_single_neuron("./data/SWC/FC/12345.swc", scale_um_per_px=5.0)
    # test_single_neuron("./data/SWC/EM/67890.swc", scale_um_per_px=5.0, show_plot=False)
    main()

