"""步驟 7 - 把描述子變成一條明確、可重現的納入條件。

論文需要的是讀者可以重新套用的規則, 而不只是一串「有顯著差異」的清單。
這裡擬合並交叉驗證兩類規則 (10-fold 分層, 重複 10 次):

  規則 A (只用 SWC)       -- 兩個資料庫都適用, 只吃骨架資訊
  規則 B (SWC + neuropil) -- 僅限 FlyCircuit, 額外加入 compartment 佔位編碼

每組特徵都會回報 logistic regression (輸入標準化) 與深度 2 的決策樹; 決策樹會
印成明確的 if/else, 方便直接引用到 methods 章節。

輸出: results/selection_rule_cv.csv
      results/selection_rule_tree.txt
      results/selection_rule_coefficients.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C

FEATURE_SETS = {
    "A_swc_only": ["cable_length_um", "n_branch_points", "arbor_separation", "span_um"],
    "A_swc_minimal": ["cable_length_um", "arbor_separation"],
    "B_swc_neuropil": ["cable_length_um", "sidetot_top2", "arbor_separation"],
    "B_neuropil_only": ["sidetot_top2", "regiontot_balance21", "sidetot_n_above_20pct"],
    "B_minimal": ["cable_length_um", "sidetot_top2"],
}


def load() -> pd.DataFrame:
    npil = pd.read_csv(C.OUT / "neuropil_metrics.csv")
    morph = pd.read_csv(C.OUT / "morphology_metrics.csv")
    morph = morph[morph.source == "FC"]
    df = npil.merge(morph.drop(columns=["group", "exclusive"]), on="neuron_id", how="inner")
    return df[df.exclusive].reset_index(drop=True)


def conjunctive_rule(df: pd.DataFrame, f1: str, f2: str, d1: str = ">=", d2: str = "<=",
                     folds: int = 10, repeats: int = 10) -> dict:
    """最佳的兩項 AND 規則, 例如 (cable >= t1) AND (second-share <= t2)。

    門檻的網格搜尋在每個訓練 fold 內部進行, 只在留出的 fold 上計分, 因此回報的
    balanced accuracy 不是擬合準確率。
    """
    y = (df.group == C.GROUP_DENSE).astype(int).to_numpy()
    X1, X2 = df[f1].to_numpy(float), df[f2].to_numpy(float)

    def fire(t1, t2, i=slice(None)):
        a = X1[i] >= t1 if d1 == ">=" else X1[i] <= t1
        b = X2[i] <= t2 if d2 == "<=" else X2[i] >= t2
        return a & b

    def search(idx):
        g1 = np.quantile(X1[idx], np.linspace(0.05, 0.95, 40))
        g2 = np.quantile(X2[idx], np.linspace(0.05, 0.95, 40))
        best = (-1, None, None)
        for t1 in g1:
            for t2 in g2:
                sc = balanced_accuracy_score(y[idx], fire(t1, t2, idx).astype(int))
                if sc > best[0]:
                    best = (sc, t1, t2)
        return best

    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                 random_state=C.RANDOM_STATE)
    scores = []
    for tr, te in cv.split(X1, y):
        _, t1, t2 = search(tr)
        scores.append(balanced_accuracy_score(y[te], fire(t1, t2, te).astype(int)))
    full, t1, t2 = search(np.arange(len(y)))
    pred = fire(t1, t2).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "rule": f"({f1} {d1} {t1:.4g}) AND ({f2} {d2} {t2:.4g})",
        "feature_1": f1, "direction_1": d1, "threshold_1": float(t1),
        "feature_2": f2, "direction_2": d2, "threshold_2": float(t2),
        "balanced_acc_insample": full,
        "balanced_acc_cv_mean": float(np.mean(scores)),
        "balanced_acc_cv_sd": float(np.std(scores)),
        "precision_dense": tp / (tp + fp) if tp + fp else np.nan,
        "recall_dense": tp / (tp + fn) if tp + fn else np.nan,
        "n": int(len(y)),
    }


def single_threshold_rule(df: pd.DataFrame, f: str, d: str = ">=",
                          folds: int = 10, repeats: int = 10) -> dict:
    """單一特徵切點, 用同樣方式交叉驗證, 作為要被超越的基準線。"""
    y = (df.group == C.GROUP_DENSE).astype(int).to_numpy()
    X = df[f].to_numpy(float)
    fire = (lambda t, i=slice(None): X[i] >= t) if d == ">=" else (lambda t, i=slice(None): X[i] <= t)

    def search(idx):
        best = (-1, None)
        for t in np.quantile(X[idx], np.linspace(0.02, 0.98, 200)):
            sc = balanced_accuracy_score(y[idx], fire(t, idx).astype(int))
            if sc > best[0]:
                best = (sc, t)
        return best

    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                 random_state=C.RANDOM_STATE)
    scores = [balanced_accuracy_score(y[te], fire(search(tr)[1], te).astype(int))
              for tr, te in cv.split(X, y)]
    full, t = search(np.arange(len(y)))
    return {"rule": f"{f} {d} {t:.4g}", "feature_1": f, "direction_1": d,
            "threshold_1": float(t), "feature_2": "", "direction_2": "", "threshold_2": np.nan,
            "balanced_acc_insample": full, "balanced_acc_cv_mean": float(np.mean(scores)),
            "balanced_acc_cv_sd": float(np.std(scores)), "n": int(len(y))}


def main() -> pd.DataFrame:
    df = load()
    y = (df.group == C.GROUP_DENSE).astype(int).to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=C.RANDOM_STATE)

    rows, coefs, tree_txt = [], [], []
    for name, feats in FEATURE_SETS.items():
        X = df[feats].to_numpy(float)
        models = {
            "logistic": make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000, C=1.0)),
            "tree_d2": DecisionTreeClassifier(max_depth=2, min_samples_leaf=15,
                                              random_state=C.RANDOM_STATE),
        }
        for mname, model in models.items():
            sc = cross_validate(model, X, y, cv=cv,
                                scoring=("balanced_accuracy", "roc_auc"), n_jobs=-1)
            rows.append({
                "feature_set": name, "model": mname, "n": len(y),
                "features": ", ".join(feats),
                "balanced_acc_mean": sc["test_balanced_accuracy"].mean(),
                "balanced_acc_sd": sc["test_balanced_accuracy"].std(),
                "auc_mean": sc["test_roc_auc"].mean(),
                "auc_sd": sc["test_roc_auc"].std(),
            })
        # 用全部資料重新擬合一次, 得到可引用的規則
        t = models["tree_d2"].fit(X, y)
        tree_txt.append(f"### {name}\n" + export_text(t, feature_names=feats, decimals=2)
                        + f"  (class 1 = {C.GROUP_DENSE}; in-sample balanced acc "
                          f"{balanced_accuracy_score(y, t.predict(X)):.3f})\n")
        lr = models["logistic"].fit(X, y)
        for f, c in zip(feats, lr[-1].coef_[0]):
            coefs.append({"feature_set": name, "feature": f, "std_coef": c})

    out = pd.DataFrame(rows).sort_values("balanced_acc_mean", ascending=False)
    out.to_csv(C.OUT / "selection_rule_cv.csv", index=False)
    pd.DataFrame(coefs).to_csv(C.OUT / "selection_rule_coefficients.csv", index=False)
    (C.OUT / "selection_rule_tree.txt").write_text("\n".join(tree_txt))

    print(out.round(3).drop(columns=["features"]).to_string(index=False))
    print("\n" + "\n".join(tree_txt))

    print("=== explicit thresholds (cross-validated) ===")
    rules = [
        single_threshold_rule(df, "cable_length_um", ">="),
        single_threshold_rule(df, "n_branch_points", ">="),
        single_threshold_rule(df, "sidetot_top2", "<="),
        conjunctive_rule(df, "cable_length_um", "sidetot_top2", ">=", "<="),
        conjunctive_rule(df, "cable_length_um", "regiontot_top2", ">=", "<="),
        conjunctive_rule(df, "n_branch_points", "sidetot_top2", ">=", "<="),
        conjunctive_rule(df, "cable_length_um", "arbor_separation", ">=", "<="),
    ]
    rdf = pd.DataFrame(rules).sort_values("balanced_acc_cv_mean", ascending=False)
    rdf.to_csv(C.OUT / "selection_rule_thresholds.csv", index=False)
    print(rdf[["rule", "balanced_acc_insample", "balanced_acc_cv_mean",
               "balanced_acc_cv_sd"]].round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
