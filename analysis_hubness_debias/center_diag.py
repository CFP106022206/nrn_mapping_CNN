"""樣本外 per-EM 置中：扣掉 FC-independent 的成分後還剩多少？

為什麼需要
----------
`em_prior_*` 問的是「完全忽略 FC 能做多好」，是一個**平行**的虛無模型。
置中問的是互補的另一面：把每顆 EM 的全域偏置**從模型自己的分數裡減掉**，
剩下的 FC-conditional 成分還能不能排對。

先驗高不必然代表模型沒學到形態——同型 EM 本來就該對整族 FC 都評分高。
但若置中後增益整個消失，那就只剩先驗。

⚠️ 置中會傷害**所有**模型（`debias.py` 的 11 種設定沒有一組讓池內 AUC 上升），
所以絕對值偏悲觀；有效的是**同一變換下的模型間比較**。

偏置一律樣本外估：一半 FC 估、另一半評估，避免用同一批 FC 自己減自己。

執行：
    python3 analysis_hubness_debias/center_diag.py results/probe_scores_annotator_scratch.parquet
    python3 analysis_hubness_debias/center_diag.py ../result/evalscores_<stem>.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def metrics(d: pd.DataFrame, key: str) -> dict[str, float]:
    s = d.sort_values(key, ascending=False)
    g = s.groupby("fc_id", observed=True)
    t = g.head(1)
    t = t[t.type_label.notna()]
    s = s.copy()
    s["_r"] = g.cumcount() + 1
    corr = s[s.type_label == 1.0].groupby("fc_id", observed=True)._r.min()
    nan = float("nan")
    return {
        "rank1": 100 * float((t.type_label == 1.0).mean()) if len(t) else nan,
        "MRR": float((1 / corr).mean()) if len(corr) else nan,
        "p@5": 100 * float((corr <= 5).mean()) if len(corr) else nan,
        "n_fc": int(corr.index.nunique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():          # 先試 cwd，再試腳本目錄
        p = (ROOT / args.path).resolve()
    d = pd.read_parquet(p)
    d["fc_id"] = d.fc_id.astype(str)
    cols = [c for c in d.columns if c not in ("fc_id", "em_id", "type_label")]
    print(f"[center] {p.name}：{len(d):,} 對、FC {d.fc_id.nunique()}、分數欄 {cols}\n")

    # 樣本外切分：偏置在 est 上估，指標在 val 上算（兩個分數欄共用同一切分）
    rng = np.random.default_rng(args.seed)
    fcs = d.fc_id.unique()
    perm = rng.permutation(fcs)
    est, val = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
    dv = d[d.fc_id.isin(val)].copy()

    print(f"{'欄位':<12}{'':<4}{'rank-1':>9}{'MRR':>9}{'p@5':>9}{'n_fc':>7}")
    out = []
    for c in cols:
        bias = d[d.fc_id.isin(est)].groupby("em_id", observed=True)[c].mean()
        dv[f"{c}_ctr"] = dv[c] - dv.em_id.map(bias).fillna(bias.mean())
        for tag, k in (("原始", c), ("置中", f"{c}_ctr")):
            m = metrics(dv, k)
            print(f"{c:<12}{tag:<4}{m['rank1']:>9.2f}{m['MRR']:>9.4f}"
                  f"{m['p@5']:>9.2f}{m['n_fc']:>7d}")
            out.append({"col": c, "variant": tag, **m})
        print()

    res = pd.DataFrame(out)
    o = ROOT / "results" / f"center_{p.stem}.csv"
    res.to_csv(o, index=False)
    print(f"-> {o.relative_to(ROOT.parent)}")
    print("\n判讀：置中後若某欄仍明顯高於基線欄的置中值，"
          "\n      它的 FC-conditional 成分就是真的；若兩者拉平，增益只剩先驗。")


if __name__ == "__main__":
    main()
