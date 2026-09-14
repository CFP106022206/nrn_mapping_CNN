"""步驟 4 - 用三個角度量化「這組有多難配對」。

(a) 專家信心: labeled_info/{D2,D6,D5}_conf.csv 記錄了人類標註者對每組候選配對
    在 [0,1] 之間的信心值。如果 dense 類神經真的比較難分辨, 標註者本身就應該
    比較猶豫 -- 打 1.0 的比例較低, 分布更集中在模稜兩可的中間帶。

(b) NBLAST 可分離度: labeled_info/nblast_{D2+D6,D5}_50as1.csv 存有 NBLAST 分數
    與二元化的 ground truth。單看 NBLAST 的 AUC, 就能知道傳統形態配對在各組
    手上有多少可用訊號。

NBLAST 分數取自 results/nblast_official.csv (run_nblast_official.py 以 navis +
官方 smat.fcwb 重算)。專案內原有的分數版本彼此不一致, 已淘汰。

輸出: results/confidence_by_group.csv
      results/nblast_separability.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K

SIGMA_UM = 8.0        # NBLAST 式核函數的空間容忍度
N_POINTS = 500        # 重採樣加抽樣後每顆神經保留的點數
N_BOOT = 20           # 為了讓候選池大小一致而做的抽樣次數


# ------------------------------------------------------------------- (a) ----
def expert_confidence() -> pd.DataFrame:
    src = {C.GROUP_DENSE: ["D2_conf.csv", "D6_conf.csv"],
           C.GROUP_PROJ: ["D5_conf.csv"]}
    rows = []
    for grp, files in src.items():
        d = pd.concat([pd.read_csv(C.ROOT / "labeled_info" / f) for f in files],
                      ignore_index=True)
        d["group"] = grp
        rows.append(d[["fc_id", "em_id", "label", "group"]])
    conf = pd.concat(rows, ignore_index=True)
    conf.to_csv(C.OUT / "expert_confidence_pairs.csv", index=False)

    pos = conf[conf.label > 0]          # 標註者接受的候選配對
    summ = []
    for grp, g in conf.groupby("group"):
        p = g[g.label > 0]["label"]
        summ.append({
            "group": grp, "n_pairs": len(g),
            "n_accepted": int((g.label > 0).sum()),
            "accept_rate": float((g.label > 0).mean()),
            "mean_conf_accepted": float(p.mean()),
            "median_conf_accepted": float(p.median()),
            "frac_conf_eq_1": float((p == 1.0).mean()),
            "frac_conf_le_0p7": float((p <= 0.7).mean()),
        })
    out = pd.DataFrame(summ)
    a = pos[pos.group == C.GROUP_DENSE]["label"].to_numpy()
    b = pos[pos.group == C.GROUP_PROJ]["label"].to_numpy()
    out.attrs["p"] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    out["mannwhitney_p_vs_other"] = out.attrs["p"]
    out["cliffs_delta_dense_minus_proj"] = K.cliffs_delta(a, b)
    out.to_csv(C.OUT / "confidence_by_group.csv", index=False)
    print("\n(a) expert confidence\n", out.to_string(index=False))
    return out


# ------------------------------------------------------------------- (b) ----
def nblast_separability() -> pd.DataFrame:
    f = C.OUT / "nblast_official.csv"
    if not f.exists():
        raise SystemExit("缺 results/nblast_official.csv, 請先執行:\n"
                         "  conda run -n nblast python run_nblast_official.py")
    d0 = pd.read_csv(f)
    d0["group"] = d0["group"].map({"D1_projection": C.GROUP_PROJ,
                                   "D2_dense": C.GROUP_DENSE})
    d0["label"] = (d0["conf"] >= C.POS_CONF).astype(int)
    rows = []
    for grp, d in d0.groupby("group"):
        pos = d.loc[d.label == 1, "nblast_official"].to_numpy(float)
        neg = d.loc[d.label == 0, "nblast_official"].to_numpy(float)
        cut = K.best_threshold(pos, neg)
        rows.append({
            "group": grp, "source": "nblast_official.csv",
            "n_pos": pos.size, "n_neg": neg.size,
            "nblast_auc": K.auc_from_u(pos, neg),
            "pos_median": np.median(pos), "neg_median": np.median(neg),
            "median_gap": np.median(pos) - np.median(neg),
            "best_balanced_acc": cut.get("balanced_acc", np.nan),
            "mannwhitney_p": stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue,
        })
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "nblast_separability.csv", index=False)
    print("\n(b) NBLAST separability\n", out.to_string(index=False))
    return out


if __name__ == "__main__":
    expert_confidence()
    nblast_separability()
