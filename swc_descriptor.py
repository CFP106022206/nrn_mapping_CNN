# %%
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import numpy as np
import time

@dataclass(frozen=True)
class Swc:
    nid: np.ndarray      # (N,) int32
    ntype: np.ndarray    # (N,) int16
    xyz: np.ndarray      # (N,3) float32
    radius: np.ndarray   # (N,) float32
    parent: np.ndarray   # (N,) int32


def load_swc(path: str | Path) -> Swc:
    """Robust SWC loader: skips comment/header lines, accepts float-like ints, ignores extra cols."""
    path = Path(path)
    nid = []
    ntype = []
    xyz = []
    radius = []
    parent = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//") or line.startswith(";"):
                continue
            # skip non-comment header lines containing letters
            if any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in line):
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 7:
                raise ValueError(
                    f"{path.name}:{lineno} malformed SWC row: expected >=7 cols, got {len(parts)} -> {line!r}"
                )
            parts = parts[:7]

            try:
                i = int(float(parts[0]))
                t = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                r = float(parts[5])
                p = int(float(parts[6]))
            except Exception as e:
                raise ValueError(f"{path.name}:{lineno} parse error -> {line!r}") from e

            nid.append(i)
            ntype.append(t)
            xyz.append((x, y, z))
            radius.append(r)
            parent.append(p)

    nid = np.asarray(nid, dtype=np.int32)
    ntype = np.asarray(ntype, dtype=np.int16)
    xyz = np.asarray(xyz, dtype=np.float32)
    radius = np.asarray(radius, dtype=np.float32)
    parent = np.asarray(parent, dtype=np.int32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{path.name}: xyz shape invalid: {xyz.shape}")

    return Swc(nid=nid, ntype=ntype, xyz=xyz, radius=radius, parent=parent)

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
            # skip header lines containing letters
            if any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in line):
                continue
            buf.append(line.replace(",", " "))

    if not buf:
        raise ValueError(f"{path.name}: no numeric SWC rows found")

    # fromstring 比 loadtxt 更快
    data = np.fromstring(" ".join(buf), sep=" ", dtype=np.float64)
    if data.size % 7 != 0:
        # 有些 swc 行可能有额外列
        raise ValueError(f"{path.name}: parsed values not divisible by 7 (got {data.size})")

    arr = data.reshape(-1, 7)

    nid = arr[:, 0].astype(np.int32, copy=False)
    ntype = arr[:, 1].astype(np.int16, copy=False)
    xyz = arr[:, 2:5].astype(np.float32, copy=False)
    radius = arr[:, 5].astype(np.float32, copy=False)
    parent = arr[:, 6].astype(np.int32, copy=False)
    return Swc(nid=nid, ntype=ntype, xyz=xyz, radius=radius, parent=parent)

def _build_parent_index(nid: np.ndarray) -> dict[int, int]:
    """Map SWC node id -> row index."""
    # SWC ids are usually 1..N, but do not assume contiguous.
    return {int(nid[i]): i for i in range(nid.shape[0])}


def compute_segment_weighted_centroid_and_inertia_eigen(
    swc: Swc,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        centroid: (3,) float32
        eigvals: (3,) float32 sorted descending
        eigvecs: (3,3) float32 columns aligned with eigvals (same order)
    """
    xyz = swc.xyz.astype(np.float64, copy=False)
    parent = swc.parent

    id2idx = _build_parent_index(swc.nid)

    child_idx = []
    par_idx = []
    for i in range(parent.shape[0]):
        pid = int(parent[i])
        if pid == -1:
            continue
        pidx = id2idx.get(pid)
        if pidx is None:
            continue
        child_idx.append(i)
        par_idx.append(pidx)

    if not child_idx:
        raise ValueError("No valid parent-child segments found (all parent=-1 or missing parents).")

    child_idx = np.asarray(child_idx, dtype=np.int32)
    par_idx = np.asarray(par_idx, dtype=np.int32)

    a = xyz[child_idx]
    b = xyz[par_idx]
    d = a - b
    seg_len = np.linalg.norm(d, axis=1)

    mask = seg_len > 0
    if not np.any(mask):
        raise ValueError("All segments have zero length; cannot compute weighted centroid/inertia.")

    a = a[mask]
    b = b[mask]
    seg_len = seg_len[mask]

    mid = (a + b) * 0.5
    wsum = seg_len.sum()
    centroid = (mid * seg_len[:, None]).sum(axis=0) / wsum

    r = mid - centroid[None, :]
    r2 = np.einsum("ij,ij->i", r, r)

    s = (seg_len * r2).sum()
    I = np.eye(3) * s

    O = np.einsum("i,ik,il->kl", seg_len, r, r)
    I = I - O
    I = 0.5 * (I + I.T)  # 消除浮點數誤差，確保完全對稱

    # eigvals + eigvecs
    eigvals, eigvecs = np.linalg.eigh(I)  # eigvals ascending, eigvecs columns

    # sort descending and sync eigenvectors
    idx = np.argsort(eigvals)[::-1]       # descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]   # 同步排序特征向量

    return centroid.astype(np.float32), eigvals.astype(np.float32), eigvecs.astype(np.float32)

def normalized_inertia_ratios(eigvals: np.ndarray, eps: float = 1e-12,) -> np.ndarray:
    """
    eigvals: (3,) descending [λ1, λ2, λ3]
    returns: (3,) [1.0, λ2/λ1, λ3/λ1]
    """
    l1 = max(eigvals[0], eps)
    return np.array(
        [1.0, eigvals[1] / l1, eigvals[2] / l1],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class Descriptor:
    centroid: np.ndarray   # (3,) float32
    eigvals: np.ndarray    # (3,) float32 (descending)
    eigvecs: np.ndarray    # (3,3) float32 columns correspond to eigvals order
    ratios: np.ndarray       # (3,) = [1, λ2/λ1, λ3/λ1]
    
    def as_vector(self) -> np.ndarray:
        """
        [拋棄]
        Coarse-matching feature vector (scale-invariant).
        """
        return np.concatenate([self.centroid, self.ratios], dtype=np.float32)

def compute_descriptor(path: str | Path) -> Descriptor:

    swc = load_swc_fast(path)
    c, eigvals, eigvecs = compute_segment_weighted_centroid_and_inertia_eigen(swc)
    ratios = normalized_inertia_ratios(eigvals)  #  [1, λ2/λ1, λ3/λ1]
    return Descriptor(centroid=c, eigvals=eigvals, eigvecs=eigvecs, ratios=ratios,)

# %% 測試
if __name__ == "__main__":

    t0 = time.time()
    d = compute_descriptor("data/SWC/EM/203257652.swc")
    print(d.centroid, d.eigvals, d.ratios, d.eigvecs)

    print("Elapsed:", time.time() - t0)
# %%
