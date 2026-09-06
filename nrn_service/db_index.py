"""curated 資料庫的常駐索引。

每一側（FC / EM）在服務啟動時載入一次：
    - descriptor（centroid / inertia ratio / eigvec / neuron id）
    - centroid 的 KDTree（候選初篩用，建一次重複用）
    - neuron id -> row 的對照表
    - 可選的 sha256 索引（由 tools/build_curated_index.py 產生），
      用來判斷「上傳的檔案是不是就是資料庫裡那一顆」

descriptor 檔是位置對齊的 .npy，重跑 swc_descriptor_batch.py 之後列順序會變，
所以這裡一律用 neuron id 對照，不在任何地方保存 row index。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # type: ignore

from candidate_matching import SourceDesc, load_source

from .config import Paths, normalize_side


@dataclass(frozen=True)
class NeuronDescriptor:
    """單顆神經元的幾何特徵，格式與 curated 資料庫的一列相同。"""

    neuron_id: str
    centroid: np.ndarray   # (3,) float32
    ratios2d: np.ndarray   # (2,) float32 = [r21, r31]
    eigvecs: np.ndarray    # (3,3) float32，欄向量對應 λ1>=λ2>=λ3


class SideIndex:
    """單側資料庫的索引。"""

    def __init__(self, side: str, paths: Paths, *, build_tree: bool = True) -> None:
        self.side = normalize_side(side)
        self.desc: SourceDesc = load_source(paths.descriptor_dir(self.side), self.side)
        self.neuron_ids: np.ndarray = np.asarray(self.desc.neuron_ids).astype(str)
        self.id2row: dict[str, int] = {k: i for i, k in enumerate(self.neuron_ids)}
        self.tree: cKDTree | None = (
            cKDTree(self.desc.centroids.astype(np.float64, copy=False)) if build_tree else None
        )
        # sha256 索引是可選的；沒有就退回只靠檔名比對
        self.sha256_by_id: dict[str, str] = {}
        self.id_by_sha256: dict[str, str] = {}
        self._load_sha_index(paths)

    def _load_sha_index(self, paths: Paths) -> None:
        p = paths.index_dir / f"curated_index_{self.side}.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p)
        if not {"neuron_id", "sha256"}.issubset(df.columns):
            return
        for nid, sha in zip(df["neuron_id"].astype(str), df["sha256"].astype(str)):
            self.sha256_by_id[nid] = sha
            # 同樣內容的檔案理論上只會有一份；真的重複時保留第一個
            self.id_by_sha256.setdefault(sha, nid)

    def __len__(self) -> int:
        return len(self.neuron_ids)

    def __contains__(self, neuron_id: str) -> bool:
        return str(neuron_id) in self.id2row

    def get(self, neuron_id: str) -> NeuronDescriptor | None:
        i = self.id2row.get(str(neuron_id))
        if i is None:
            return None
        return NeuronDescriptor(
            neuron_id=str(neuron_id),
            centroid=self.desc.centroids[i],
            ratios2d=self.desc.ratios2d[i],
            eigvecs=self.desc.eigvecs[i],
        )

    def lookup_by_sha256(self, sha256: str) -> str | None:
        """內容雜湊命中 -> 回傳資料庫裡的 neuron id（可能與上傳檔名不同）。"""
        return self.id_by_sha256.get(str(sha256))

    def sha256_of(self, neuron_id: str) -> str | None:
        return self.sha256_by_id.get(str(neuron_id))

    @property
    def has_sha_index(self) -> bool:
        return bool(self.sha256_by_id)


class CuratedDB:
    """兩側索引的容器。服務執行期間只讀。"""

    def __init__(self, paths: Paths, *, sides: tuple[str, ...] = ("FC", "EM")) -> None:
        self.paths = paths
        self.sides: dict[str, SideIndex] = {s: SideIndex(s, paths) for s in sides}

    def __getitem__(self, side: str) -> SideIndex:
        return self.sides[normalize_side(side)]

    def find_by_sha256(self, sha256: str) -> tuple[str, str] | None:
        """在兩側找內容相同的神經元，回傳 (side, neuron_id)。"""
        for side, idx in self.sides.items():
            nid = idx.lookup_by_sha256(sha256)
            if nid is not None:
                return side, nid
        return None

    def summary(self) -> str:
        parts = []
        for side, idx in self.sides.items():
            parts.append(f"{side}={len(idx)}" + ("(+sha)" if idx.has_sha_index else ""))
        return " ".join(parts)
