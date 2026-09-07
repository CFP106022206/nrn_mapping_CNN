"""步驟 6 - 把區分兩組的兩條軸線拆開。

步驟 5 顯示單一描述子裡最強的都是「尺寸」類 (cable 長度、分支點數、佔用體積),
而 neuropil 統計量描述的是 cable「去了哪裡」。這兩者有可能只是同一件事說兩遍:
比較大的 arbor 本來就比較容易把整團都塞在一個 neuropil 裡。本步驟檢驗在尺寸
固定之後 neuropil 訊號是否還在, 並補上「空間重疊」主張所需要的族群層級統計量。

(1) 尺寸軸與 neuropil 分散軸之間的相關性。
(2) 尺寸配對後的對照: 每顆 projection 神經 1:1 配上 cable 長度最接近的 dense
    神經 (|d log L| <= log 1.25), 再在配對後的子集上重測 neuropil 特徵。
輸出: results/size_vs_spread_correlation.csv
      results/contrast_neuropil_FC_sizematched.csv
      results/size_matching_sensitivity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K

SPREAD_FEATURES = ["region_top2", "side_top2", "region_balance21", "side_balance21",
                   "region_top1", "region_top3_sum", "region_n_above_20pct",
                   "side_n_above_20pct", "other_fraction", "region_neff_simpson"]
SIZE_FEATURE = "cable_length_um"
MATCH_TOL = np.log(1.25)          # cable 長度容許 +-25 %
MATCH_TOLS = [1.25, 1.5, 2.0]     # 配對視窗的敏感度掃描


def merged_fc() -> pd.DataFrame:
    npil = pd.read_csv(C.OUT / "neuropil_metrics.csv")
    morph = pd.read_csv(C.OUT / "morphology_metrics.csv")
    morph = morph[morph.source == "FC"]
    df = npil.merge(morph.drop(columns=["group", "exclusive"]), on="neuron_id", how="inner")
    return df[df.exclusive].reset_index(drop=True)


# ------------------------------------------------------------------- (1) ----
def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in SPREAD_FEATURES:
        for scope, sub in [("pooled", df),
                           (C.GROUP_DENSE, df[df.group == C.GROUP_DENSE]),
                           (C.GROUP_PROJ, df[df.group == C.GROUP_PROJ])]:
            r, p = stats.spearmanr(np.log(sub[SIZE_FEATURE]), sub[f])
            rows.append({"feature": f, "scope": scope, "spearman_rho": r, "p": p, "n": len(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "size_vs_spread_correlation.csv", index=False)
    print("\n(1) log(cable length) vs neuropil spread")
    print(out.pivot(index="feature", columns="scope", values="spearman_rho").round(3).to_string())
    return out


# ------------------------------------------------------------------- (2) ----
def _match(df: pd.DataFrame, tol: float):
    proj = df[df.group == C.GROUP_PROJ]
    dense = df[df.group == C.GROUP_DENSE]
    lp, ld = np.log(proj[SIZE_FEATURE].to_numpy()), np.log(dense[SIZE_FEATURE].to_numpy())
    used, kp, kd = set(), [], []
    for i in np.argsort(lp):
        d = np.abs(ld - lp[i])
        if used:
            d[list(used)] = np.inf
        j = int(d.argmin())
        if d[j] <= np.log(tol):
            used.add(j); kp.append(proj.index[i]); kd.append(dense.index[j])
    return df.loc[kp + kd].copy(), len(kp)


def matching_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """在不同的尺寸配對視窗下, neuropil 訊號是否還存在?"""
    rows = []
    for tol in MATCH_TOLS:
        m, n = _match(df, tol)
        for f in SPREAD_FEATURES:
            r = K.describe_split(m, f)
            if r:
                rows.append({"tol": tol, "n_matched_pairs": n, "feature": f,
                             "auc": r["auc"], "cliffs_delta": r["cliffs_delta"],
                             "p": r["mannwhitney_p"]})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "size_matching_sensitivity.csv", index=False)
    print("\n(2b) AUC vs size-matching window")
    piv = out.pivot(index="feature", columns="tol", values="auc").round(3)
    piv.columns = [f"+-{int((t-1)*100)}%(n={out.loc[out.tol==t,'n_matched_pairs'].iloc[0]})"
                   for t in piv.columns]
    print(piv.to_string())
    return out


def size_matched(df: pd.DataFrame) -> pd.DataFrame:
    proj = df[df.group == C.GROUP_PROJ].copy()
    dense = df[df.group == C.GROUP_DENSE].copy()
    lp = np.log(proj[SIZE_FEATURE].to_numpy())
    ld = np.log(dense[SIZE_FEATURE].to_numpy())

    used, keep_p, keep_d = set(), [], []
    for i in np.argsort(lp):                       # 貪婪最近鄰 1:1 配對
        d = np.abs(ld - lp[i])
        d[list(used)] = np.inf
        j = int(d.argmin())
        if d[j] <= MATCH_TOL:
            used.add(j)
            keep_p.append(proj.index[i])
            keep_d.append(dense.index[j])
    matched = df.loc[keep_p + keep_d].copy()
    print(f"\n(2) size-matched subset: {len(keep_p)} pairs "
          f"(median cable {matched.groupby('group')[SIZE_FEATURE].median().round(0).to_dict()})")

    rows = [r for r in (K.describe_split(matched, f) for f in SPREAD_FEATURES) if r]
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out.insert(0, "panel", "neuropil_FC_sizematched")
    out.to_csv(C.OUT / "contrast_neuropil_FC_sizematched.csv", index=False)

    full = pd.read_csv(C.OUT / "contrast_neuropil_FC.csv").set_index("feature")["auc"]
    cmp = out.set_index("feature")[["auc", "cliffs_delta", "mannwhitney_p",
                                    "projection_median", "dense_median"]].copy()
    cmp["auc_full_cohort"] = full.reindex(cmp.index)
    print(cmp.round(3).to_string())
    matched.to_csv(C.OUT / "size_matched_cohort.csv", index=False)
    return out


if __name__ == "__main__":
    df = merged_fc()
    correlations(df)
    size_matched(df)
    matching_sensitivity(df)
