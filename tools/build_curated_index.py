"""建立 curated 資料庫的 sha256 索引。

    python3 tools/build_curated_index.py

用途：判斷使用者上傳的檔案是不是「其實就是資料庫裡那一顆，只是改了檔名」，
以及反過來偵測「檔名撞名但內容不同」的情況。

輸出：data/index/curated_index_{FC,EM}.parquet
      欄位 neuron_id / sha256 / n_bytes / has_views
SWC 檔有增刪之後要重跑。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrn_service.config import SIDES, ServiceConfig  # noqa: E402


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_side(side: str, cfg: ServiceConfig) -> Path:
    swc_dir = cfg.paths.swc_dir(side)
    views_dir = cfg.paths.views_dir(side)
    desc_ids = set(
        np.load(cfg.paths.descriptor_dir(side) / f"neuron_ids_{side}.npy", allow_pickle=True)
        .astype(str)
        .tolist()
    )

    rows = []
    paths = sorted(swc_dir.glob("*.swc"))
    for k, p in enumerate(paths, start=1):
        rows.append(
            {
                "neuron_id": p.stem,
                "sha256": sha256_file(p),
                "n_bytes": p.stat().st_size,
                "has_descriptor": p.stem in desc_ids,
                "has_views": (views_dir / f"{p.stem}_views.npz").exists(),
            }
        )
        if k % 5000 == 0 or k == len(paths):
            print(f"  [{k}/{len(paths)}] hashed", flush=True)

    df = pd.DataFrame(rows)
    cfg.paths.index_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.index_dir / f"curated_index_{side}.parquet"
    df.to_parquet(out, index=False)

    dup = int(df["sha256"].duplicated().sum())
    print(
        f"[index] {side}: {len(df)} neurons -> {out}"
        f"  (內容重複 {dup}, 缺 descriptor {int((~df['has_descriptor']).sum())},"
        f" 缺 views {int((~df['has_views']).sum())})"
    )
    return out


def main() -> None:
    cfg = ServiceConfig()
    ap = argparse.ArgumentParser(description="Build sha256 index for the curated SWC database.")
    ap.add_argument("--side", choices=list(SIDES), default=None)
    args = ap.parse_args()
    for side in ([args.side] if args.side else list(SIDES)):
        print(f"[index] {side} ...", flush=True)
        build_side(side, cfg)


if __name__ == "__main__":
    main()
