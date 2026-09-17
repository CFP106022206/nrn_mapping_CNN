"""專家標註與型別判定是否衝突？守門 AUC 卡在 0.81–0.86 是不是因為這個？

問題
----
排序微調的訓練訊號是型別層級的：同型 EM 要排在「確定不同型」的 EM 前面。
專家標註是個體層級的。若兩者衝突，排序項與專家 BCE 會把 head 往相反方向拉，
守門（專家測試集 AUC）就可能永遠追不上 annotator。

三個檢查
--------
1. 交叉表：專家信心度 × 型別判定（1.0 同型 / 0.5 同亞族無法斷定 / 0.0 確定不同型 / 無法判定）。
   * **硬衝突**：專家判為正例（>= 0.5），型別卻是 0.0。兩個訊號直接相反。
   * **軟張力**：專家判為負例（< 0.5），型別卻是 1.0（同型但不是那一顆）。
     這不矛盾——型別比個體寬鬆——但 InfoNCE 會把這種 EM 當正例往上推，
     BCE 卻往下壓。
2. 同一顆 FC 內的排序：同型配對的專家分數是否一律 >= 確定不同型配對的專家分數。
3. 拆解守門 AUC：用各輪存下的 `result/test_label_*.csv`，
   依型別判定把專家測試集切開，分別算 annotator 與 RankTune 的 AUC。
   若 RankTune 只在「同型配對之間分辨對錯個體」那一層掉分，
   原因就是型別訊號太粗，不是標註衝突。

執行：python3 analysis_hubness_debias/expert_type_consistency.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT))
from debias import build_judge, label_pairs  # noqa: E402

RESULTS = ROOT / "results"
TL_NAME = {1.0: "同型(1.0)", 0.5: "同亞族(0.5)", 0.0: "不同型(0.0)"}


def tl_cat(v) -> str:
    return TL_NAME.get(v, "無法判定") if v is not None and not pd.isna(v) else "無法判定"


def load_split(split: str, fc, judge) -> tuple[pd.DataFrame, int]:
    d = pd.read_csv(PROJ / "train_test_split" / f"{split}_split_0_D1-D6.csv")
    d["fc_id"] = d.fc_id.astype(str)
    d["em_id"] = d.em_id.astype("int64")
    n_all = len(d)
    lab = label_pairs(d, fc, judge)
    d = d.merge(lab[["fc_id", "em_id", "np_type", "type_label"]],
                on=["fc_id", "em_id"], how="left")
    d["tl"] = d.type_label.map(tl_cat)
    d["exp_pos"] = d.label >= 0.5
    return d, n_all


def auc_ci(y, s, n_boot=2000, seed=3) -> tuple[float, float, float]:
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main() -> None:
    fc, judge = build_judge()
    order = ["同型(1.0)", "同亞族(0.5)", "不同型(0.0)", "無法判定"]

    splits = {}
    for split in ("train", "test"):
        d, n_all = load_split(split, fc, judge)
        splits[split] = d
        print(f"\n######## {split}_split_0：{n_all} 對 ########")
        print("型別判定覆蓋：", d.tl.value_counts().reindex(order, fill_value=0).to_dict())

        # --- 檢查 1：交叉表 ---
        ct = pd.crosstab(d.label, d.tl).reindex(columns=order, fill_value=0)
        print("\n[檢查 1] 專家信心度 × 型別判定")
        print(ct.to_string())
        hard = d[d.exp_pos & (d.type_label == 0.0)]
        soft = d[~d.exp_pos & (d.type_label == 1.0)]
        n_pos_typed = int((d.exp_pos & d.type_label.isin([0.0, 1.0])).sum())
        n_neg_typed = int((~d.exp_pos & d.type_label.isin([0.0, 1.0])).sum())
        print(f"\n  硬衝突（專家正例但型別 0.0）：{len(hard)} 對"
              f"（佔可判定正例 {len(hard)}/{n_pos_typed}）")
        print(f"  軟張力（專家負例但型別 1.0）：{len(soft)} 對"
              f"（佔可判定負例 {len(soft)}/{n_neg_typed}）")
        if len(hard):
            vfb = dict(zip(fc.fc_id, fc.vfb_type))
            h = hard.assign(vfb_type=hard.fc_id.map(vfb))
            print("  硬衝突明細：")
            print(h[["fc_id", "em_id", "label", "vfb_type", "np_type"]]
                  .to_string(index=False))

        # --- 檢查 2：同一顆 FC 內的排序 ---
        n_fc = n_cmp = n_bad = n_tie = 0
        bad_rows = []
        for fcid, g in d.groupby("fc_id"):
            a = g[g.type_label == 1.0]
            b = g[g.type_label == 0.0]
            if not len(a) or not len(b):
                continue
            n_fc += 1
            for (_, ra), (_, rb) in product(a.iterrows(), b.iterrows()):
                n_cmp += 1
                if ra.label < rb.label:
                    n_bad += 1
                    bad_rows.append((fcid, ra.em_id, ra.label, rb.em_id, rb.label))
                elif ra.label == rb.label:
                    n_tie += 1
        print(f"\n[檢查 2] 同一顆 FC 內，同型 vs 不同型的專家分數")
        print(f"  同時有兩種配對的 FC：{n_fc} 顆，比較 {n_cmp} 組")
        if n_cmp:
            print(f"  同型分數 > 不同型：{n_cmp - n_bad - n_tie}  "
                  f"相等：{n_tie}  **同型 < 不同型（違反）：{n_bad}**")
        for r in bad_rows[:10]:
            print(f"    {r[0]}：同型 em {r[1]} 專家 {r[2]}  <  不同型 em {r[3]} 專家 {r[4]}")

    # --- 檢查 3：拆解守門 AUC ---
    te = splits["test"]
    runs = sorted((PROJ / "result").glob("test_label_RankTune*.csv"))
    print("\n\n######## [檢查 3] 拆解守門 AUC（專家測試集 122 對）########")
    print("每一層：n（正/負）  annotator AUC [95% CI]  ->  RankTune AUC [95% CI]")
    layers = {
        "全部": lambda x: np.ones(len(x), bool),
        "同型(1.0)：同型之間分辨對錯個體": lambda x: (x.type_label == 1.0).to_numpy(),
        "非同型（0.0/0.5/無法判定）": lambda x: (x.type_label != 1.0).to_numpy(),
    }
    out = []
    for f in runs:
        r = pd.read_csv(f)
        r["fc_id"] = r.fc_id.astype(str)
        r["em_id"] = r.em_id.astype("int64")
        m = r.merge(te[["fc_id", "em_id", "type_label"]], on=["fc_id", "em_id"], how="left")
        y = (m.label >= 0.5).astype(int).to_numpy()
        tag = f.stem.replace("test_label_RankTune_annotator_", "")
        print(f"\n  {tag}")
        for name, sel in layers.items():
            k = sel(m)
            npos, nneg = int(y[k].sum()), int((1 - y[k]).sum())
            if npos == 0 or nneg == 0:
                print(f"    {name:<34s} n={k.sum():3d}（{npos}/{nneg}）  單一類別，無法算 AUC")
                continue
            a, alo, ahi = auc_ci(y[k], m.base.to_numpy()[k])
            b, blo, bhi = auc_ci(y[k], m.ranktune.to_numpy()[k])
            print(f"    {name:<34s} n={k.sum():3d}（{npos}/{nneg}）  "
                  f"{a:.3f} [{alo:.3f},{ahi:.3f}]  ->  {b:.3f} [{blo:.3f},{bhi:.3f}]")
            out.append({"run": tag, "layer": name, "n": int(k.sum()), "n_pos": npos,
                        "n_neg": nneg, "auc_annotator": a, "auc_ranktune": b,
                        "ranktune_lo": blo, "ranktune_hi": bhi})
        # 全部那一層做配對 bootstrap，看差距本身是否顯著
        rng = np.random.default_rng(5)
        diffs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) == 2:
                diffs.append(roc_auc_score(y[i], m.base.to_numpy()[i])
                             - roc_auc_score(y[i], m.ranktune.to_numpy()[i]))
        diffs = np.array(diffs)
        print(f"    annotator − RankTune（配對 bootstrap）：{diffs.mean():+.3f} "
              f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]  "
              f"P(annotator 較好)={(diffs > 0).mean():.3f}")

    pd.DataFrame(out).to_csv(RESULTS / "expert_type_consistency_guard.csv", index=False)
    for split, d in splits.items():
        d.to_csv(RESULTS / f"expert_type_consistency_{split}.csv", index=False)
    print("\n-> results/expert_type_consistency_{guard,train,test}.csv")


if __name__ == "__main__":
    main()
