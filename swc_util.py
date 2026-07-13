from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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