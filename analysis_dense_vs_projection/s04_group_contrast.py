"""步驟 5 - 把每個描述子依「分得多開」排名。

每個特徵都會回報各組的中位數 [四分位距]、Mann-Whitney U 檢定、Cliff's delta、
該特徵單獨使用時的 ROC AUC, 以及讓 Youden's J 最大的單一切點 -- 也就是一條
可以直接寫進論文的納入條件。

形態指標一律在「同一個資料庫內部」比較。FlyCircuit (光學顯微鏡) 與 hemibrain
(電顯) 的重建在 cable 長度與節點密度上差了一個數量級以上, 把 FC 與 EM 混在
一起比, 量到的會是成像方式而不是細胞類型。

同時出現在兩個 sub dataset 的神經會被排除 (`exclusive == True`)。

輸出: results/contrast_neuropil_FC.csv
      results/contrast_morphology_{FC,EM}.csv
      results/contrast_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K

SKIP = {"neuron_id", "group", "source", "exclusive", "n_pairs",
        "top_region", "second_region", "n_groups"}


def contrast(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = df[df.exclusive].copy() if "exclusive" in df else df.copy()
    feats = [c for c in df.columns
             if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
    rows = [r for r in (K.describe_split(df, f) for f in feats) if r]
    out = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    out.insert(0, "panel", tag)
    out.to_csv(C.OUT / f"contrast_{tag}.csv", index=False)
    return out


def show(out: pd.DataFrame, n: int = 12) -> None:
    cols = ["feature", "auc", "cliffs_delta", "mannwhitney_p",
            "projection_median", "dense_median", "cut_direction",
            "cut_threshold", "cut_balanced_acc"]
    d = out[cols].head(n).copy()
    for c in ("auc", "cliffs_delta", "projection_median", "dense_median",
              "cut_threshold", "cut_balanced_acc"):
        d[c] = d[c].astype(float).round(3)
    d["mannwhitney_p"] = d["mannwhitney_p"].map(lambda v: f"{v:.2e}")
    print(d.to_string(index=False))


def main() -> pd.DataFrame:
    npil = pd.read_csv(C.OUT / "neuropil_metrics.csv")
    morph = pd.read_csv(C.OUT / "morphology_metrics.csv")

    parts = []
    print("\n=== neuropil occupancy (FlyCircuit) ===")
    a = contrast(npil, "neuropil_FC"); show(a); parts.append(a)

    for src in ("FC", "EM"):
        print(f"\n=== skeleton geometry ({src}) ===")
        b = contrast(morph[morph.source == src], f"morphology_{src}")
        show(b); parts.append(b)

    conf = pd.read_csv(C.OUT / "confusability.csv") if (C.OUT / "confusability.csv").exists() else None
    if conf is not None:
        print("\n=== candidate confusability (FlyCircuit queries) ===")
        conf["exclusive"] = True
        c = contrast(conf, "confusability_FC"); show(c); parts.append(c)

    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(C.OUT / "contrast_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    main()
