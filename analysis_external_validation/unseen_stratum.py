"""在「兩側神經都沒進訓練集」的乾淨分層上比較 annotator 與 fine-tune。

十折測試集的配對有 91–97 % 涉及訓練時見過的神經（`leakage_overlap.csv`），
身分先驗本身就值 AUC 0.85（`leakage_shortcut.csv`）。唯一不受此影響的是
「兩側都沒看過」那一層，但它很小，所以一律附 bootstrap CI。

分層定義與 `analysis_model_results/s13_leakage_check.py` 一致：
測試配對的 fc_id / em_id 是否出現在同一折的 train split（任一側、任一配對皆算看過）。

執行： python3 analysis_external_validation/unseen_stratum.py
輸出： results/unseen_stratum.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = ROOT / "results"

N_FOLDS = 10
SEED = 7
N_BOOT = 5000

MODELS = {
    "annotator": "Annotator_D1-D6",
    "finetune_180K": "FineTune180K_miniLR_D1-D6",
    "finetune_miniLR": "FineTune_miniLR_D1-D6",
    "finetune_e7": "FineTune_miniLR_e7_D1-D6",
}


def boot_auc(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """對 AUC 做 bootstrap，回傳 95 % CI。"""
    n = len(y)
    out = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 每折的「看過」集合
    seen = {}
    for i in range(N_FOLDS):
        tr = pd.read_csv(PROJECT / "train_test_split" / f"train_split_{i}_D1-D6.csv")
        seen[i] = (set(tr.fc_id.astype(str)), set(tr.em_id.astype(str)))

    frames = {}
    for name, stem in MODELS.items():
        parts = []
        for i in range(N_FOLDS):
            f = PROJECT / "result" / f"test_label_{stem}_{i}.csv"
            if not f.exists():
                break
            t = pd.read_csv(f)
            fc_seen, em_seen = seen[i]
            t["fold"] = i
            t["fc_seen"] = t.fc_id.astype(str).isin(fc_seen)
            t["em_seen"] = t.em_id.astype(str).isin(em_seen)
            parts.append(t)
        if parts:
            frames[name] = pd.concat(parts, ignore_index=True)

    ref = next(iter(frames.values()))
    strata = {
        "兩側都沒看過": ~ref.fc_seen & ~ref.em_seen,
        "只有 FC 看過": ref.fc_seen & ~ref.em_seen,
        "只有 EM 看過": ~ref.fc_seen & ref.em_seen,
        "兩側都看過": ref.fc_seen & ref.em_seen,
        "全部": pd.Series(True, index=ref.index),
    }

    rng = np.random.default_rng(SEED)
    rows = []
    for sname, mask in strata.items():
        for mname, d in frames.items():
            s = d[mask.to_numpy()]
            y = (s.label >= 0.5).to_numpy().astype(int)
            p = s.model_pred.to_numpy()
            if len(np.unique(y)) < 2:
                rows.append({"stratum": sname, "model": mname, "n": len(s), "auc": None})
                continue
            lo, hi = boot_auc(y, p, rng)
            rows.append(
                {
                    "stratum": sname,
                    "model": mname,
                    "n": len(s),
                    "pos_rate": round(y.mean(), 3),
                    "auc": round(roc_auc_score(y, p), 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "acc@0.5": round(((p >= 0.5).astype(int) == y).mean(), 4),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "unseen_stratum.csv", index=False)
    print(out.to_string(index=False))

    # 乾淨分層上的配對比較（同一批配對，兩個模型）
    mask = (~ref.fc_seen & ~ref.em_seen).to_numpy()
    print(f"\n=== 「兩側都沒看過」分層：n = {mask.sum()} ===")
    a = frames["annotator"][mask]
    ya = (a.label >= 0.5).to_numpy().astype(int)
    for mname in [m for m in frames if m != "annotator"]:
        b = frames[mname][mask]
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, mask.sum(), mask.sum())
            if len(np.unique(ya[idx])) < 2:
                continue
            diffs.append(
                roc_auc_score(ya[idx], b.model_pred.to_numpy()[idx])
                - roc_auc_score(ya[idx], a.model_pred.to_numpy()[idx])
            )
        d = np.array(diffs)
        print(
            f"  {mname:18s} − annotator: ΔAUC {d.mean():+.4f} "
            f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
            f"P(Δ>0) = {(d > 0).mean():.3f}"
        )


if __name__ == "__main__":
    main()
