"""步驟 12 - MorphoMatcher (CNN) 與 NBLAST 在同一組資料性質上並排: CNN 受不受海綿效應影響。

問題
----
s10 證明 NBLAST 在 D2 的假陽性主要來自 hemibrain 側的「海綿效應」: target 越密, query
的點越容易找到近鄰, 分數被結構性推高。MorphoMatcher 的輸入是三視圖, 每顆神經以自己的
立方包圍盒正規化 (standard_draw.py 的 compute_bbox_3d), 看不到位置與絕對大小, 也沒有
最近鄰計分。它的誤判若仍與 EM 密度相關, 機制必然不同; 若不相關, 就要看它的誤判跟哪一條
資料性質 (大小 / 佔位集中度) 有關。

兩種方法一律用同一組配對、同一份標籤 (results/nblast_official.csv 的 conf, 專家信心
>= 0.5 為正例), 描述子與「達真配對水準」的定義與 s10 相同:
  (1) 可分離度 (AUC)
  (2) 分數 vs 資料性質的 Spearman 相關 (真 / 假配對分開; EM / FC 兩側; 配對層級的空間重疊)
  (3) 依 EM 凸包密度四等分, 假配對「達真配對水準」(>= 真配對分數 25% 分位) 的比例
  (4) 偏相關: 密度與大小分開 (互相控制), EM / FC 兩側互相控制, 佔位集中度控制大小
  (5) 誤判輪廓: 在 ROC 最靠近 (0,1) 的操作點, FP vs TN 與 FN vs TP 的描述子 AUC
  (6) hub EM: 高分假配對是否集中在同幾顆密集 EM
  (7) fold 穩健性: CNN 分數來自 10 個 fold 模型, 另以 fold 內百分位 (CNN_foldrank) 重算

CNN 分數: result/test_label_FineTune_miniLR_D1-D6_{0..9}.csv (10-fold 交叉驗證, 每組配對
只由沒訓練過它的 fold 模型打分)。NBLAST 分數: results/nblast_official.csv。

輸出: results/cnn_sponge_auc.csv          兩種方法的可分離度
      results/cnn_sponge_correlations.csv  分數 vs 資料性質的相關
      results/cnn_sponge_density_bins.csv  依 EM 密度分箱的「達真配對水準」比例
      results/cnn_sponge_partial.csv       偏相關
      results/cnn_error_profile.csv        誤判輪廓
      results/cnn_hub_em.csv               高分假配對最多的 EM
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K

GROUP_MAP = {"D1_projection": C.GROUP_PROJ, "D2_dense": C.GROUP_DENSE}
CNN_PREFIX = C.ROOT / "result" / "test_label_FineTune_miniLR_D1-D6_"
N_FOLDS = 10
PRIMARY_FEAT = "cable_per_hull_um2"  # 與 s10 相同的主要密度指標
MORPH_FEATS = ["cable_per_hull_um2", "fill_ratio", "revisit_r16um",
               "hull_volume_um3", "cable_length_um"]
NEUROPIL_FEATS = ["sidetot_top2", "regiontot_top2"]   # 只有 FC 端
METHODS = ["NBLAST", "CNN", "CNN_foldrank"]
FEATURES = ([f"em_{f}" for f in MORPH_FEATS] + [f"fc_{f}" for f in MORPH_FEATS]
            + [f"fc_{f}" for f in NEUROPIL_FEATS] + ["frac_nn_gt10"])


def load() -> pd.DataFrame:
    d = pd.read_csv(C.OUT / "nblast_official.csv")
    for c in ("fc_id", "em_id"):
        d[c] = d[c].astype(str).str.strip()
    d["group"] = d["group"].map(GROUP_MAP)
    d["label"] = (d["conf"] >= C.POS_CONF).astype(int)

    parts = []
    for i in range(N_FOLDS):
        t = pd.read_csv(f"{CNN_PREFIX}{i}.csv")[["fc_id", "em_id", "model_pred"]]
        t["fold"] = i
        parts.append(t)
    cnn = pd.concat(parts, ignore_index=True)
    for c in ("fc_id", "em_id"):
        cnn[c] = cnn[c].astype(str).str.strip()
    # 不同 fold 的模型分數尺度未必一致, 另存 fold 內百分位作為穩健性檢驗
    cnn["CNN_foldrank"] = cnn.groupby("fold")["model_pred"].rank(pct=True)
    d = d.merge(cnn, on=["fc_id", "em_id"], how="inner", validate="many_to_one")
    d = d.rename(columns={"nblast_official": "NBLAST", "model_pred": "CNN"})

    m = pd.read_csv(C.OUT / "morphology_metrics.csv")
    m["neuron_id"] = m["neuron_id"].astype(str)
    keep = ["neuron_id"] + MORPH_FEATS
    em = m[m.source == "EM"].drop_duplicates("neuron_id")[keep]
    fc = m[m.source == "FC"].drop_duplicates("neuron_id")[keep]
    d = d.merge(em.add_prefix("em_"), left_on="em_id", right_on="em_neuron_id")
    d = d.merge(fc.add_prefix("fc_"), left_on="fc_id", right_on="fc_neuron_id")

    n = pd.read_csv(C.OUT / "neuropil_metrics.csv")
    n["neuron_id"] = n["neuron_id"].astype(str)
    n = n.drop_duplicates("neuron_id")[["neuron_id"] + NEUROPIL_FEATS]
    n = n.rename(columns={"neuron_id": "fc_id"}).rename(columns={f: f"fc_{f}" for f in NEUROPIL_FEATS})
    d = d.merge(n, on="fc_id", how="left")

    o = pd.read_csv(C.OUT / "sponge_orphan_points.csv")
    for c in ("fc_id", "em_id"):
        o[c] = o[c].astype(str).str.strip()
    o = o.drop_duplicates(["fc_id", "em_id"])[["fc_id", "em_id", "frac_nn_gt10"]]
    return d.merge(o, on=["fc_id", "em_id"], how="left")


def _partial_spearman(x, y, z):
    """控制 z (可多欄) 之後 x 與 y 的 Spearman 偏相關; 單一 z 時與 s10 的實作相同。"""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.asarray(z, float)
    Z = Z[:, None] if Z.ndim == 1 else Z
    B = np.c_[np.ones(len(rx)), np.column_stack([stats.rankdata(c) for c in Z.T])]

    def resid(a):
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]

    return stats.pearsonr(resid(rx), resid(ry))


# ------------------------------------------------------------------- (1) ----
def separability(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in C.GROUP_ORDER:
        s = d[d.group == g]
        for meth in METHODS:
            rows.append({"group": g, "method": meth, "n_pos": int(s.label.sum()),
                         "n_neg": int((s.label == 0).sum()),
                         "auc": roc_auc_score(s.label, s[meth])})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "cnn_sponge_auc.csv", index=False)
    print("=== (1) 可分離度 AUC (專家信心 >= 0.5 為正例) ===")
    for g in C.GROUP_ORDER:
        t = out[out.group == g]
        print(f"  {g:11s} (真 {t.n_pos.iloc[0]} / 假 {t.n_neg.iloc[0]})  "
              + "   ".join(f"{r.method} {r.auc:.3f}" for r in t.itertuples()))
    return out


# ------------------------------------------------------------------- (2) ----
def correlations(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in C.GROUP_ORDER:
        for lab in (0, 1):
            s = d[(d.group == g) & (d.label == lab)]
            for col in FEATURES:
                v = s[[col] + METHODS].dropna()
                if len(v) < 20:
                    continue
                for meth in METHODS:
                    rho, p = stats.spearmanr(v[col], v[meth])
                    rows.append({"group": g, "label": lab, "feature": col, "method": meth,
                                 "n": len(v), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "cnn_sponge_correlations.csv", index=False)

    print("\n=== (2) 分數 vs 資料性質 (Spearman; * p<0.001) ===")
    for g in C.GROUP_ORDER:
        for lab in (0, 1):
            t = out[(out.group == g) & (out.label == lab)]
            if t.empty:
                continue
            print(f"\n  {g} / {'真配對' if lab else '假配對'}  (n={int(t.n.max())})")
            print(f"    {'':26s}" + "".join(f"{m:>14s}" for m in METHODS))
            for col in FEATURES:
                r = t[t.feature == col].set_index("method")
                if r.empty:
                    continue
                cells = "".join(f"{r.loc[m, 'rho']:+13.3f}{'*' if r.loc[m, 'p'] < 1e-3 else ' '}"
                                for m in METHODS)
                print(f"    {col:26s}{cells}")
    return out


# ------------------------------------------------------------------- (3) ----
def density_bins(d: pd.DataFrame, feature: str = f"em_{PRIMARY_FEAT}",
                 n_bins: int = 4) -> pd.DataFrame:
    rows = []
    print(f"\n=== (3) 假配對依 {feature} 四等分: 達真配對水準 (>= 真配對 25% 分位) 的比例 ===")
    for g in C.GROUP_ORDER:
        s = d[d.group == g]
        neg, pos = s[s.label == 0].copy(), s[s.label == 1]
        if len(neg) < 4 * n_bins or len(pos) < 10:
            continue
        neg["bin"] = pd.qcut(neg[feature], n_bins, labels=False, duplicates="drop")
        q25 = {m: pos[m].quantile(0.25) for m in METHODS}
        print(f"\n  {g}  (真配對 25% 分位: "
              + ", ".join(f"{m} {q25[m]:.3f}" for m in METHODS) + ")")
        print(f"  {'箱':>3} {'EM 密度區間':>21} {'n':>4}" + "".join(f"{m:>14s}" for m in METHODS))
        for b, t in neg.groupby("bin"):
            fr = {m: float((t[m] >= q25[m]).mean()) for m in METHODS}
            for m in METHODS:
                rows.append({"group": g, "feature": feature, "bin": int(b) + 1,
                             "lo": t[feature].min(), "hi": t[feature].max(), "n": len(t),
                             "method": m, "frac_reaching_true": fr[m]})
            print(f"  {int(b)+1:>3} {t[feature].min():9.3g}-{t[feature].max():<11.3g} {len(t):>4}"
                  + "".join(f"{fr[m]*100:13.1f}%" for m in METHODS))
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "cnn_sponge_density_bins.csv", index=False)
    return out


# ------------------------------------------------------------------- (4) ----
PARTIAL_TESTS = [
    # (名稱, x, 控制變數)
    ("EM 密度", f"em_{PRIMARY_FEAT}", None),
    ("EM 密度 | EM cable", f"em_{PRIMARY_FEAT}", ["em_cable_length_um"]),
    ("EM cable", "em_cable_length_um", None),
    ("EM cable | EM 密度", "em_cable_length_um", [f"em_{PRIMARY_FEAT}"]),
    ("EM 密度 | FC 密度", f"em_{PRIMARY_FEAT}", [f"fc_{PRIMARY_FEAT}"]),
    ("FC 密度 | EM 密度", f"fc_{PRIMARY_FEAT}", [f"em_{PRIMARY_FEAT}"]),
    ("FC cable", "fc_cable_length_um", None),
    ("FC 第二 compartment 佔比", "fc_sidetot_top2", None),
    ("FC 第二 compartment | FC cable", "fc_sidetot_top2", ["fc_cable_length_um"]),
    ("孤兒率 (空間不重疊)", "frac_nn_gt10", None),
    ("孤兒率 | EM 密度", "frac_nn_gt10", [f"em_{PRIMARY_FEAT}"]),
]


def partials(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in C.GROUP_ORDER:
        for lab in (0, 1):
            s = d[(d.group == g) & (d.label == lab)]
            for name, x, z in PARTIAL_TESTS:
                cols = [x] + (z or []) + METHODS
                v = s[cols].dropna()
                if len(v) < 20:
                    continue
                for meth in METHODS:
                    if z:
                        rho, p = _partial_spearman(v[meth].values, v[x].values, v[z].values)
                    else:
                        rho, p = stats.spearmanr(v[x], v[meth])
                    rows.append({"group": g, "label": lab, "test": name, "method": meth,
                                 "n": len(v), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "cnn_sponge_partial.csv", index=False)

    print("\n=== (4) 偏相關: 密度 vs 大小、EM vs FC、佔位集中度、空間重疊 (* p<0.001) ===")
    for g in C.GROUP_ORDER:
        for lab in (0, 1):
            t = out[(out.group == g) & (out.label == lab)]
            if t.empty:
                continue
            print(f"\n  {g} / {'真配對' if lab else '假配對'}  (n={int(t.n.max())})")
            print(f"    {'':32s}" + "".join(f"{m:>14s}" for m in METHODS))
            for name, _, _ in PARTIAL_TESTS:
                r = t[t.test == name].set_index("method")
                if r.empty:
                    continue
                print(f"    {name:32s}" + "".join(
                    f"{r.loc[m, 'rho']:+13.3f}{'*' if r.loc[m, 'p'] < 1e-3 else ' '}" for m in METHODS))
    return out


# ------------------------------------------------------------------- (5) ----
def _operating_point(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """ROC 上最靠近 (0,1) 的操作點, 與 result_analysis_make_figure.py 相同的正規化與選點;
    比較用 >= (與 sklearn roc_curve 的門檻定義一致)。"""
    s = (score - score.min()) / (score.max() - score.min())
    fpr, tpr, thr = roc_curve(y, s)
    fin = np.isfinite(thr)
    i = int(np.argmin(np.sqrt(fpr[fin] ** 2 + (tpr[fin] - 1.0) ** 2)))
    return s >= thr[fin][i]


def error_profile(d: pd.DataFrame) -> pd.DataFrame:
    rows, preds = [], {}
    for g in C.GROUP_ORDER:
        s = d[d.group == g].reset_index(drop=True)
        y = s.label.values.astype(bool)
        for meth in ("NBLAST", "CNN"):
            pr = _operating_point(s.label.values, s[meth].values)
            preds[(g, meth)] = pr
            fp, tn, fn, tp = s[pr & ~y], s[~pr & ~y], s[~pr & y], s[pr & y]
            for col in FEATURES:
                rows.append({"group": g, "method": meth, "feature": col,
                             "n_fp": len(fp), "n_tn": len(tn), "n_fn": len(fn), "n_tp": len(tp),
                             "auc_fp_vs_tn": K.auc_from_u(fp[col], tn[col]),
                             "auc_fn_vs_tp": K.auc_from_u(fn[col], tp[col])})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "cnn_error_profile.csv", index=False)

    print("\n=== (5) 誤判輪廓: P(誤判的描述子 > 判對的)  0.5 = 無關 ===")
    for g in C.GROUP_ORDER:
        t = out[out.group == g]
        nb, cn = t[t.method == "NBLAST"].iloc[0], t[t.method == "CNN"].iloc[0]
        y = d[d.group == g].label.values.astype(bool)
        both_fp = int((preds[(g, "NBLAST")] & preds[(g, "CNN")] & ~y).sum())
        print(f"\n  {g}:  NBLAST FP {nb.n_fp} / FN {nb.n_fn}   CNN FP {cn.n_fp} / FN {cn.n_fn}"
              f"   (兩者共同的 FP {both_fp})")
        print(f"    {'':26s}{'NBLAST FP>TN':>14s}{'CNN FP>TN':>12s}{'NBLAST FN>TP':>14s}{'CNN FN>TP':>12s}")
        for col in FEATURES:
            a = t[(t.feature == col) & (t.method == "NBLAST")].iloc[0]
            b = t[(t.feature == col) & (t.method == "CNN")].iloc[0]
            print(f"    {col:26s}{a.auc_fp_vs_tn:14.3f}{b.auc_fp_vs_tn:12.3f}"
                  f"{a.auc_fn_vs_tp:14.3f}{b.auc_fn_vs_tp:12.3f}")
    return out


# ------------------------------------------------------------------- (6) ----
def hub_em(d: pd.DataFrame) -> pd.DataFrame:
    s = d[d.group == C.GROUP_DENSE]
    neg = s[s.label == 0]
    rows = []
    print("\n=== (6) hub EM: 高分假配對 (>= 真配對 25% 分位) 最多的 hemibrain 神經, D2 ===")
    for meth in ("NBLAST", "CNN"):
        q25 = s.loc[s.label == 1, meth].quantile(0.25)
        per_em = (neg.assign(high=neg[meth] >= q25)
                  .groupby("em_id").agg(n_false=("high", "size"), n_high_false=("high", "sum"),
                                        em_density=(f"em_{PRIMARY_FEAT}", "first"),
                                        em_cable=("em_cable_length_um", "first"))
                  .reset_index().sort_values(["n_high_false", "em_density"], ascending=False))
        per_em["method"] = meth
        rows.append(per_em)
        many = per_em[per_em.n_false >= 2]
        rho, p = stats.spearmanr(many.n_high_false, many.em_density)
        print(f"\n  {meth}: 高分假配對 {int(per_em.n_high_false.sum())} 組, 分布在 "
              f"{int((per_em.n_high_false > 0).sum())} 顆 EM; 吸引力 vs EM 密度 rho {rho:+.3f} "
              f"(p {p:.3g}, 至少 2 組假配對的 EM n={len(many)})")
        print(per_em.head(6)[["em_id", "n_false", "n_high_false", "em_density", "em_cable"]]
              .round(4).to_string(index=False))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.OUT / "cnn_hub_em.csv", index=False)
    return out


def main() -> None:
    d = load()
    print(f"配對 {len(d)} 組 (D1 {int((d.group == C.GROUP_PROJ).sum())}, "
          f"D2 {int((d.group == C.GROUP_DENSE).sum())})\n")
    separability(d)
    correlations(d)
    density_bins(d)
    partials(d)
    error_profile(d)
    hub_em(d)


if __name__ == "__main__":
    main()
