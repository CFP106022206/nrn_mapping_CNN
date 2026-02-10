# %%
from __future__ import annotations

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd # need pyarrow for parquet support in pandas


from swc_descriptor import compute_descriptor  # 核心函式


def iter_swc_files(root: Path, recursive: bool = True):
    if root.is_file() and root.suffix.lower() == ".swc":
        yield root
        return
    # recursive: 子目錄的遞回搜索模式，應對文件夾內有多層結構的情況
    pattern = "**/*.swc" if recursive else "*.swc"
    for p in sorted(root.glob(pattern)):
        yield p


def batch_run(input_dir: str | Path, out_dir: str | Path, source: str='FC', recursive: bool = True,
              fail_fast: bool = False):
    
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    eigvecs_list = []

    files = list(iter_swc_files(input_dir, recursive=recursive))
    files = sorted(files, key=lambda p: p.as_posix())

    if not files:
        raise FileNotFoundError(f"No .swc found under {input_dir}")

    for k, p in enumerate(files, start=1):
        try:
            d = compute_descriptor(p)
            rows.append(
                {
                    "neuron_id": p.stem,  # 只保留檔案名
                    "cx": float(d.centroid[0]),
                    "cy": float(d.centroid[1]),
                    "cz": float(d.centroid[2]),

                    # inertia eigenvalues ratio(descending)
                    "r11": float(d.ratios[0]),
                    "r21": float(d.ratios[1]),
                    "r31": float(d.ratios[2]),
                }
            )
            eigvecs_list.append(d.eigvecs.astype(np.float32))

        except Exception as e:
            errors.append({"path": str(p), "error": repr(e)})
            if fail_fast:
                raise
        if k % 200 == 0 or k == len(files):
            print(f"[{k}/{len(files)}] done")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("All files failed, no output generated.")

    parquet_path = out_dir / f"descriptors_{source}.parquet"
    df.to_parquet(parquet_path, index=False)

    eigvecs_all = np.stack(eigvecs_list, axis=0)   # (N,3,3)
    np.save(out_dir / f"eigvecs_{source}.npy", eigvecs_all)

    np.save(out_dir / f"centroids_{source}.npy", df[["cx", "cy", "cz"]].to_numpy(np.float32))
    np.save(out_dir / f"eigvals_ratio_{source}.npy", df[["r11", "r21", "r31"]].to_numpy(np.float32))
    np.save(out_dir / f"neuron_ids_{source}.npy", df["neuron_id"].to_numpy())

    if errors:
        err_df = pd.DataFrame(errors)
        err_path = out_dir / f"errors_{source}.parquet"
        err_df.to_parquet(err_path, index=False)
        print(f"{len(errors)} failed files, saved to {err_path}")

    print(f" Saved: {parquet_path}")
    return parquet_path

# %%
if __name__ == "__main__":

    # 默認運行參數
    source = "EM"
    input_path = "./data/SWC/"+source
    output_path = "./data/descriptors_"+source



    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=input_path, help="Folder containing .swc (or a single .swc)")
    ap.add_argument("--out", default=output_path, help="Output folder")
    ap.add_argument("--source", default=source, help="Dataset label, e.g. FC/EM")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    batch_run(
        input_dir=args.input,
        out_dir=args.out,
        source=args.source,
        recursive=not args.no_recursive,
        fail_fast=args.fail_fast,
    )

# %%
