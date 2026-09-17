"""探針的增益是形態，還是 EM 側的身分捷徑？

階段 2 的探針在 held-out 的 638 顆 FC 上把 rank-1 從 31 % 拉到 52–67 %。
跳幅這麼大，必須先排除一個替代解釋：訓練負例集中在少數 EM 上，
模型可能只是記住「這些 body ID 是壞的」。

這正是 `RANKING_FINETUNE_PLAN.md` §3.3 警告的 EM 側捷徑，
也是 `analysis_model_results/FINDINGS.md` §9 的 `spillover_FC_negonly` 的鏡像。
麻煩之處在於**它看起來會像成功**，因為被壓低的正是海綿 EM。

三個檢查
--------
1. 依「這顆 FC 的正解 EM 有沒有在訓練出現過」把 eval FC 分層，各報 rank-1。
2. 探針的 rank-1 選擇裡，有多少落在訓練時見過的 EM 上（與基線比）。
3. **最鋒利**：把候選池限縮到訓練時完全沒見過的 EM，重新排序後再比。
   若探針在全新候選集上仍勝出，學到的就是形態。

執行：python3 analysis_hubness_debias/check_em_leakage.py --tag annotator_existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


def rank1_hit(d: pd.DataFrame, key: str) -> tuple[float, int]:
    """回傳 (rank-1 同型率 %, 可判定的 FC 數)。"""
    t = d.sort_values(key, ascending=False).groupby("fc_id", observed=True).head(1)
    t = t[t.type_label.notna()]
    if not len(t):
        return float("nan"), 0
    return 100 * float((t.type_label == 1.0).mean()), len(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="annotator_existing")
    args = ap.parse_args()

    ev = pd.read_parquet(RESULTS / f"probe_scores_{args.tag}.parquet")
    samp = pd.read_parquet(RESULTS / f"probe_trainsample_{args.tag}.parquet")
    ev["fc_id"] = ev.fc_id.astype(str)

    seen_all = set(samp.em_id.astype("int64"))
    seen_pos = set(samp[samp.is_pos == 1].em_id.astype("int64"))
    seen_neg = set(samp[samp.is_pos == 0].em_id.astype("int64"))
    print(f"[leak] 訓練見過的 EM：{len(seen_all)}（正例 {len(seen_pos)} / 負例 {len(seen_neg)}）")
    print(f"[leak] eval 池中的 EM：{ev.em_id.nunique()}，"
          f"其中見過 {ev.em_id.isin(seen_all).groupby(ev.em_id).first().sum()}")
    print()

    out = []

    # --- 檢查 1：依「正解 EM 有沒有見過」分層 ---
    corr = ev[ev.type_label == 1.0]
    fc_any_seen = corr.groupby("fc_id", observed=True).em_id.apply(
        lambda s: bool(set(s.astype("int64")) & seen_all)
    )
    print("=== 檢查 1：依『這顆 FC 的正解 EM 有沒有在訓練出現過』分層 ===")
    for label, mask in (("正解全未見過", ~fc_any_seen), ("正解有見過", fc_any_seen)):
        fcs = set(fc_any_seen[mask].index)
        d = ev[ev.fc_id.isin(fcs)]
        if not len(d):
            print(f"  {label}: 無 FC")
            continue
        b, nb = rank1_hit(d, "score")
        p, npb = rank1_hit(d, "probe")
        print(f"  {label:12s} FC={d.fc_id.nunique():4d}  基線 {b:5.1f}%  探針 {p:5.1f}%  Δ {p-b:+5.1f}pp")
        out.append({"check": "1_正解是否見過", "stratum": label, "n_fc": d.fc_id.nunique(),
                    "baseline_pct": round(b, 2), "probe_pct": round(p, 2), "delta_pp": round(p - b, 2)})

    # --- 檢查 2：rank-1 選擇落在見過的 EM 上的比例 ---
    print("\n=== 檢查 2：rank-1 的選擇有多少落在訓練見過的 EM 上 ===")
    for key, name in (("score", "基線"), ("probe", "探針")):
        t = ev.sort_values(key, ascending=False).groupby("fc_id", observed=True).head(1)
        f_all = 100 * t.em_id.isin(seen_all).mean()
        f_pos = 100 * t.em_id.isin(seen_pos).mean()
        print(f"  {name}: 見過 {f_all:5.1f}%（其中曾當正例 {f_pos:5.1f}%）")
        out.append({"check": "2_rank1落在見過的EM", "stratum": name, "n_fc": len(t),
                    "baseline_pct": round(f_all, 2), "probe_pct": round(f_pos, 2), "delta_pp": None})

    # --- 檢查 3：候選池限縮到完全沒見過的 EM ---
    print("\n=== 檢查 3：把候選池限縮到訓練時完全沒見過的 EM（最鋒利）===")
    unseen = ev[~ev.em_id.isin(seen_all)]
    # 限縮後仍要有正解可找，否則不可比
    win = unseen[unseen.type_label == 1.0].fc_id.unique()
    d = unseen[unseen.fc_id.isin(win)]
    b, nb = rank1_hit(d, "score")
    p, npb = rank1_hit(d, "probe")
    print(f"  可用 FC {d.fc_id.nunique()}（池中位 {int(d.groupby('fc_id', observed=True).size().median())} 個候選）")
    print(f"  基線 {b:5.1f}%   探針 {p:5.1f}%   Δ {p-b:+5.1f}pp")
    out.append({"check": "3_候選池限縮為未見過的EM", "stratum": "全未見候選",
                "n_fc": d.fc_id.nunique(), "baseline_pct": round(b, 2),
                "probe_pct": round(p, 2), "delta_pp": round(p - b, 2)})

    # --- 檢查 4：把「訓練正例」整批移出候選池（最能區分記憶 vs 學會型別）---
    # 檢查 2 無法區分兩種解釋：訓練正例本來就是同型 EM 的集合，
    # 學會型別的模型也會選到它們。把它們全部拿掉，剩下的同型 EM
    # 若仍被排到第一，就只能是型別知識而非記住的清單。
    print("\n=== 檢查 4：移除所有『訓練正例』EM 後重排（區分記憶 vs 型別知識）===")
    d = ev[~ev.em_id.isin(seen_pos)]
    win = d[d.type_label == 1.0].fc_id.unique()
    d = d[d.fc_id.isin(win)]
    b, _ = rank1_hit(d, "score")
    p, _ = rank1_hit(d, "probe")
    n_fc = d.fc_id.nunique()
    print(f"  可用 FC {n_fc}（池中位 {int(d.groupby('fc_id', observed=True).size().median())} 個候選）")
    print(f"  基線 {b:5.1f}%   探針 {p:5.1f}%   Δ {p-b:+5.1f}pp")
    out.append({"check": "4_移除訓練正例EM", "stratum": "剩餘候選",
                "n_fc": n_fc, "baseline_pct": round(b, 2),
                "probe_pct": round(p, 2), "delta_pp": round(p - b, 2)})

    # --- 檢查 3、4 的 bootstrap CI（以 FC 為單位重抽）---
    print("\n=== Δ 的 95 % CI（cluster bootstrap by FC，2000 次）===")
    rng = np.random.default_rng(7)
    for name, sub in (("檢查3_未見候選", unseen[unseen.fc_id.isin(
                          unseen[unseen.type_label == 1.0].fc_id.unique())]),
                      ("檢查4_移除訓練正例", d)):
        per = {}
        for fc, g in sub.groupby("fc_id", observed=True):
            gb = g.sort_values("score", ascending=False).head(1)
            gp = g.sort_values("probe", ascending=False).head(1)
            if gb.type_label.isna().all() or gp.type_label.isna().all():
                continue
            per[fc] = (float(gb.type_label.iloc[0] == 1.0),
                       float(gp.type_label.iloc[0] == 1.0))
        keys = list(per)
        if len(keys) < 10:
            print(f"  {name}: FC 太少（{len(keys)}），略過")
            continue
        ds = []
        for _ in range(2000):
            pick = rng.choice(keys, len(keys), replace=True)
            arr = np.array([per[k] for k in pick])
            ds.append(100 * (arr[:, 1].mean() - arr[:, 0].mean()))
        ds = np.array(ds)
        print(f"  {name}: n={len(keys):4d}  Δ {ds.mean():+5.1f}pp  "
              f"CI [{np.percentile(ds,2.5):+5.1f}, {np.percentile(ds,97.5):+5.1f}]  "
              f"P(Δ>0)={(ds>0).mean():.3f}")
        out.append({"check": f"CI_{name}", "stratum": "bootstrap", "n_fc": len(keys),
                    "baseline_pct": None, "probe_pct": None,
                    "delta_pp": round(float(ds.mean()), 2)})

    res = pd.DataFrame(out)
    res.to_csv(RESULTS / f"em_leakage_{args.tag}.csv", index=False)
    print(f"\n-> results/em_leakage_{args.tag}.csv")
    print("\n判讀：檢查 3 的 Δ 若仍明顯為正，增益來自形態而非身分捷徑；"
          "\n      若檢查 1 的增益集中在「正解有見過」那一層，就是捷徑。")


if __name__ == "__main__":
    main()
