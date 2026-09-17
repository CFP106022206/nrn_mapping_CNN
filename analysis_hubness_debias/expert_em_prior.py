"""EM 側的先驗是不是被專家標註的分布帶進模型的？

階段 3 留下的線索
----------------
`--head-init existing` 與 `--head-init scratch` 唯一的差別是 head 的初始權重，
EM 先驗卻差 9.5 pp（20.9 % vs 11.4 %）。這指向一個可能：
**EM 側的傾向是 annotator head 從專家標註學來的，不是排序損失新產生的。**
若成立，調損失（BCE 權重、cross 負例、temperature）都是在下游擦地板。

怎麼問
------
`Annotator_D1-D6_0` 的訓練集是 `train_split_0_D1-D6.csv`（1 097 對、543 顆 EM），
而 `pool_scores_annotator.parquet` 是同一個模型掃 2 735 顆 FC 的全池分數。
FC 側幾乎不重疊（2 735 顆裡只有 6 顆被專家標註過），所以兩者之間
能傳遞的只有 **EM 側**的資訊——這正好把 EM 側的影響隔離出來。

把每顆 EM 依「在專家訓練集裡以什麼身分出現過」分組，比較它們在全池的表現：

  pos_only / neg_only / both / unseen

判讀
----
* 只有 `pos_only` 被抬高 -> 可能是先驗，也可能那些 EM 本來就是好配對，不可分辨。
* `neg_only` 也被抬高    -> 與標籤無關、只要「出現過」就被抬高，
                            那就是身分記憶（identity shortcut），不是形態。
* 出現次數愈多分數愈高   -> 記憶的劑量反應，比組間比較更難用「品質」解釋。

執行：python3 analysis_hubness_debias/expert_em_prior.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
RESULTS = ROOT / "results"
FOLD = 0


def main() -> None:
    exp = pd.read_csv(PROJ / "train_test_split" / f"train_split_{FOLD}_D1-D6.csv")
    exp["em_id"] = exp.em_id.astype("int64")
    # 專案約定：信心度 >= 0.5 視為正例（見 git log 2659a1e 之前的那次修正）
    exp["is_pos"] = exp.label >= 0.5

    pool = pd.read_parquet(ROOT / "scores" / "pool_scores_annotator.parquet")
    pool["em_id"] = pool.em_id.astype("int64")

    print(f"[expert] 專家訓練集 {len(exp)} 對：正 {int(exp.is_pos.sum())} / "
          f"負 {int((~exp.is_pos).sum())}，涵蓋 {exp.em_id.nunique()} 顆 EM、"
          f"{exp.fc_id.nunique()} 顆 FC")
    print(f"[expert] 全池 {len(pool):,} 對、{pool.fc_id.nunique()} 顆 FC、"
          f"{pool.em_id.nunique()} 顆 EM")
    ov = set(pool.fc_id.astype(str)) & set(exp.fc_id.astype(str))
    print(f"[expert] FC 側重疊：{len(ov)} 顆 —— 兩邊能傳遞的只有 EM 側資訊\n")

    # --- 每顆 EM 的全池表現 ---
    per_em = pool.groupby("em_id", observed=True).score.agg(["mean", "max", "size"])
    per_em.columns = ["mean_score", "max_score", "n_chance"]

    win = (pool.sort_values("score", ascending=False)
               .groupby("fc_id", observed=True).head(1)
               .em_id.value_counts())
    per_em["n_rank1"] = win.reindex(per_em.index).fillna(0).astype(int)

    top5 = (pool.sort_values("score", ascending=False)
                .groupby("fc_id", observed=True).head(5)
                .em_id.value_counts())
    per_em["n_top5"] = top5.reindex(per_em.index).fillna(0).astype(int)

    # --- 分組 ---
    pos = set(exp.loc[exp.is_pos, "em_id"])
    neg = set(exp.loc[~exp.is_pos, "em_id"])
    def grp(e: int) -> str:
        if e in pos and e in neg:
            return "both"
        if e in pos:
            return "pos_only"
        if e in neg:
            return "neg_only"
        return "unseen"
    per_em["group"] = [grp(e) for e in per_em.index]

    print("=== 依「在專家訓練集裡以什麼身分出現過」分組 ===")
    rows = []
    for g in ("pos_only", "neg_only", "both", "unseen"):
        d = per_em[per_em.group == g]
        if not len(d):
            continue
        r = {"group": g, "n_em": len(d),
             "mean_score": d.mean_score.mean(),
             "median_mean_score": d.mean_score.median(),
             "rank1_per_em": d.n_rank1.mean(),
             "top5_per_em": d.n_top5.mean(),
             "n_chance": d.n_chance.mean()}
        rows.append(r)
        print(f"  {g:9s} EM={len(d):5d}  平均分 {r['mean_score']:.4f}"
              f"（中位 {r['median_mean_score']:.4f}）"
              f"  rank-1/EM {r['rank1_per_em']:6.3f}"
              f"  top5/EM {r['top5_per_em']:6.3f}"
              f"  出場數 {r['n_chance']:7.0f}")
    res = pd.DataFrame(rows)

    # --- 全池 rank-1 的佔比 ---
    seen = pos | neg
    tot_r1 = int(per_em.n_rank1.sum())
    seen_r1 = int(per_em.loc[per_em.index.isin(seen), "n_rank1"].sum())
    share_em = 100 * len(seen & set(per_em.index)) / len(per_em)
    print(f"\n=== 專家見過的 EM 在全池 rank-1 裡的佔比 ===")
    print(f"  它們只佔候選池的 {share_em:.2f} % 的 EM，"
          f"卻拿下 {100*seen_r1/tot_r1:.2f} % 的 rank-1（{seen_r1}/{tot_r1}）")
    print(f"  富集倍數 {(100*seen_r1/tot_r1)/share_em:.1f}x")

    # --- 劑量反應：出現次數 vs 分數 ---
    print("\n=== 劑量反應：在專家訓練集出現幾次 vs 全池平均分 ===")
    cnt = exp.em_id.value_counts()
    d = per_em[per_em.index.isin(seen)].copy()
    d["n_exp"] = cnt.reindex(d.index).fillna(0).astype(int)
    base = per_em[per_em.group == "unseen"].mean_score.mean()
    print(f"  （未見過的 EM 平均分 {base:.4f} 作為基準）")
    for lo, hi, lbl in ((1, 1, "1 次"), (2, 2, "2 次"), (3, 4, "3–4 次"), (5, 99, "5 次以上")):
        sub = d[(d.n_exp >= lo) & (d.n_exp <= hi)]
        if not len(sub):
            continue
        print(f"  {lbl:9s} EM={len(sub):4d}  平均分 {sub.mean_score.mean():.4f}"
              f"  (Δ vs 未見 {sub.mean_score.mean()-base:+.4f})"
              f"  rank-1/EM {sub.n_rank1.mean():.3f}")

    # --- 同樣的劑量反應，但只看負例（品質解釋在這裡不成立）---
    print("\n=== 只看「僅以負例出現」的 EM（若仍被抬高，就不是品質） ===")
    dn = per_em[per_em.group == "neg_only"].copy()
    dn["n_exp"] = cnt.reindex(dn.index).fillna(0).astype(int)
    for lo, hi, lbl in ((1, 1, "1 次"), (2, 99, "2 次以上")):
        sub = dn[(dn.n_exp >= lo) & (dn.n_exp <= hi)]
        if not len(sub):
            continue
        print(f"  {lbl:9s} EM={len(sub):4d}  平均分 {sub.mean_score.mean():.4f}"
              f"  (Δ vs 未見 {sub.mean_score.mean()-base:+.4f})")

    # --- bootstrap CI：pos_only / neg_only 相對 unseen 的 Δ 平均分 ---
    print("\n=== Δ 平均分的 95 % CI（以 EM 為單位重抽，2 000 次）===")
    rng = np.random.default_rng(11)
    u = per_em[per_em.group == "unseen"].mean_score.to_numpy()
    for g in ("pos_only", "neg_only", "both"):
        a = per_em[per_em.group == g].mean_score.to_numpy()
        if len(a) < 10:
            continue
        ds = np.array([rng.choice(a, len(a), True).mean()
                       - rng.choice(u, min(len(u), 5000), True).mean()
                       for _ in range(2000)])
        print(f"  {g:9s} Δ {ds.mean():+.4f}  "
              f"CI [{np.percentile(ds,2.5):+.4f}, {np.percentile(ds,97.5):+.4f}]  "
              f"P(Δ>0)={(ds>0).mean():.3f}")

    RESULTS.mkdir(exist_ok=True)
    per_em.to_csv(RESULTS / "expert_em_prior_per_em.csv")
    res.to_csv(RESULTS / "expert_em_prior_groups.csv", index=False)
    print(f"\n-> results/expert_em_prior_{{per_em,groups}}.csv")


if __name__ == "__main__":
    main()
