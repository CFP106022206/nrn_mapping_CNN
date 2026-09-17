"""per-EM 去偏：不重訓，只改排序的鍵。

`FINDINGS.md` §10 的機制是 **per-EM 偏移**——某些 EM 不分 FC 一律被拉高
（海綿效應）。全域平移不可能改變 rank-1（池內排序對單調變換不變），
但減去一個**每顆 EM 各自不同**的基線會重排。

這支程式比較數種去偏變換，在外部型別判定下量 rank-1 / MRR / 正解 rank。

估計式
------
從每顆 EM 在所有 FC 池中的分數估一個基線 `r_e`，然後：

    none      s
    center    s − α·mean_e
    topk      s − α·(e 的 top-k 分數均值)        ← CSLS 的核心項
    zscore    (s − mean_e) / std_e
    ranknorm  s 在 e 自己分數分布中的百分位

⚠️ CSLS 原式是 `2s − r_e − r_f`。`r_f`（查詢側的 local scaling）在**單一 FC 的池內是常數**，
不影響該池的排序，所以對 rank-1 而言 CSLS ≡ `s − r_e/2`，
也就是 `topk` 搭配 α = 0.5。這裡直接掃描 α。

基線的估計集
-----------
`--baseline-from all`   用全部 FC 的池估（轉導式，但不使用任何標籤，檢索場景標準做法）
`--baseline-from disjoint`  只用不在評估集裡的 FC 估（對照組，排除轉導疑慮）

執行：
    python3 analysis_hubness_debias/debias.py --model finetune
    python3 analysis_hubness_debias/debias.py --model annotator --baseline-from disjoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "analysis_external_validation"))

from type_match import expectation, grade  # noqa: E402

SCORES = ROOT / "scores"
RESULTS = ROOT / "results"
EXT = PROJECT / "analysis_external_validation"

TOPK = 10          # topk 估計式用的 k
MIN_COUNT = 20     # EM 出現在少於這麼多個池時，基線退回全域均值


# ------------------------------------------------------------------ 判定

def build_judge() -> tuple[pd.DataFrame, dict]:
    """回傳 (fc 型別表, (vfb_type, np_type) -> label 的查表)。"""
    fc = pd.read_csv(EXT / "cache" / "fc_types_vfb.csv")
    fc["exp"] = fc.vfb_type.map(expectation)
    fc = fc[fc.exp.notna()].copy()
    fc["_v"] = fc.exp.map(lambda e: e.vague)
    fc = fc.sort_values("_v").drop_duplicates("fc_id").drop(columns="_v")

    em = pd.read_csv(EXT / "cache" / "em_types_all_db.csv")
    em["np_type"] = em.np_type.fillna("")
    em_type = dict(zip(em.em_id.astype("int64"), em.np_type))

    # Expectation 是 dataclass、不可雜湊，用 label 去重
    exps = {e.label: e for e in fc.exp}
    types = set(em_type.values())
    verdict: dict[tuple[str, str], float | None] = {}
    for lbl, e in exps.items():
        for t in types:
            verdict[(lbl, t)] = grade(e, t)[1]
    return fc, (em_type, verdict)


def label_pairs(df: pd.DataFrame, fc: pd.DataFrame, judge) -> pd.DataFrame:
    em_type, verdict = judge
    lab = dict(zip(fc.fc_id, fc.exp.map(lambda e: e.label)))
    df = df[df.fc_id.isin(lab)].copy()
    df["np_type"] = df.em_id.map(em_type).fillna("")
    keys = list(zip(df.fc_id.map(lab), df.np_type))
    df["type_label"] = [verdict.get(k) for k in keys]
    return df


# ------------------------------------------------------------------ 去偏

def em_stats(df: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    g = df.groupby("em_id", observed=True).score
    out = pd.DataFrame({"n": g.size(), "mean": g.mean(), "std": g.std().fillna(0.0)})
    out["topk"] = (
        df.sort_values("score", ascending=False)
        .groupby("em_id", observed=True)
        .score.apply(lambda s: s.head(topk).mean())
    )
    return out


def build_ecdf(base_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """每顆 EM 在**基線集**中的分數排序陣列，供 ranknorm 用 searchsorted 查百分位。"""
    return {
        int(em): np.sort(g.to_numpy())
        for em, g in base_df.groupby("em_id", observed=True).score
    }


def apply_transform(df: pd.DataFrame, st: pd.DataFrame, method: str, alpha: float,
                    global_mean: float, ecdf: dict[int, np.ndarray] | None = None,
                    global_sorted: np.ndarray | None = None) -> pd.Series:
    if method == "none":
        return df.score
    n = df.em_id.map(st["n"]).fillna(0)
    ok = n >= MIN_COUNT
    if method in ("center", "topk"):
        col = "mean" if method == "center" else "topk"
        r = df.em_id.map(st[col]).fillna(global_mean).where(ok, global_mean)
        return df.score - alpha * r
    if method == "zscore":
        mu = df.em_id.map(st["mean"]).fillna(global_mean).where(ok, global_mean)
        sd = df.em_id.map(st["std"]).fillna(1.0).where(ok, 1.0).clip(lower=1e-3)
        return (df.score - mu) / sd
    if method == "ranknorm":
        # 百分位一律對照**基線集**的分布，disjoint 模式才不會偷看評估集
        assert ecdf is not None and global_sorted is not None
        out = np.empty(len(df), dtype="float64")
        ems = df.em_id.to_numpy()
        scs = df.score.to_numpy()
        for i, (e, s) in enumerate(zip(ems, scs)):
            arr = ecdf.get(int(e))
            if arr is None or len(arr) < MIN_COUNT:
                arr = global_sorted
            out[i] = np.searchsorted(arr, s, side="right") / len(arr)
        return pd.Series(out, index=df.index)
    raise ValueError(method)


# ------------------------------------------------------------------ 評估

def pool_auc(g: pd.DataFrame, key: str) -> float | None:
    """單一 FC 的池內 AUC（型別相符 vs 確定不同型），用 Mann-Whitney 的 rank 公式。

    這是唯一**不條件在模型自己 top-5 上**的排序品質量測，
    所以它能把「候選進不進得了頂端」與「進去之後怎麼排」分開看。
    """
    j = g[g.type_label.isin([0.0, 1.0])]
    y = (j.type_label == 1.0).to_numpy()
    if y.all() or not y.any():
        return None
    r = j[key].rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(df: pd.DataFrame, key: str) -> dict:
    """df 需含 fc_id / type_label / <key>。只算 winnable 的 FC。"""
    d = df.sort_values(key, ascending=False)
    g = d.groupby("fc_id", observed=True)

    top1 = g.head(1)
    judged = top1[top1.type_label.notna()]
    hit = (judged.type_label == 1.0).mean() * 100

    # 第一個型別相符者的 rank（1-based）
    d = d.copy()
    d["_r"] = g.cumcount() + 1
    corr = d[d.type_label == 1.0].groupby("fc_id", observed=True)._r.min()

    aucs = [a for a in (pool_auc(x, key) for _, x in d.groupby("fc_id", observed=True))
            if a is not None]
    return {
        "n_fc": d.fc_id.nunique(),
        "rank1_judged": len(judged),
        "rank1_hit_pct": round(hit, 2),
        "MRR": round((1.0 / corr).mean(), 4),
        "median_rank_of_correct": int(corr.median()),
        "p@5_pct": round(100 * (corr <= 5).mean(), 2),
        "pool_auc": round(float(np.mean(aucs)), 4) if aucs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["finetune", "annotator"], required=True)
    ap.add_argument("--baseline-from", choices=["all", "disjoint"], default="all")
    ap.add_argument("--eval-frac", type=float, default=0.5,
                    help="disjoint 模式下，多少比例的 FC 當評估集")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    src = SCORES / f"pool_scores_{args.model}.parquet"
    if not src.exists():
        sys.exit(f"找不到 {src}，請先跑 dump_pool_scores.py --model {args.model}")

    df = pd.read_parquet(src)
    df["fc_id"] = df.fc_id.astype(str)
    print(f"[debias] {args.model}: {len(df):,} 對 / FC {df.fc_id.nunique()} / EM {df.em_id.nunique()}")

    fc, judge = build_judge()
    df = label_pairs(df, fc, judge)

    # winnable：池中確實有型別相符的 EM，否則任何方法都不可能答對。
    # 只有「評估」限定在 winnable；估基線時 non-winnable 的池照用——
    # 它們天然不在評估集裡，是零循環的免費樣本。
    winnable = set(df[df.type_label == 1.0].fc_id.unique())
    print(f"[debias] winnable FC = {len(winnable)}，"
          f"另有 {df.fc_id.nunique() - len(winnable)} 顆 non-winnable 可供估基線")

    if args.baseline_from == "all":
        base_df, eval_fc = df, winnable
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(sorted(winnable))
        n_eval = int(len(perm) * args.eval_frac)
        eval_fc = set(perm[:n_eval])
        base_df = df[~df.fc_id.isin(eval_fc)]   # 含全部 non-winnable
        print(f"[debias] 基線由 {base_df.fc_id.nunique()} 顆與評估集不相交的 FC 估出"
              f"（{len(base_df):,} 對），評估 {len(eval_fc)} 顆")
    df = df[df.fc_id.isin(winnable)]

    st = em_stats(base_df)
    gm = float(base_df.score.mean())
    ecdf = build_ecdf(base_df)
    gsorted = np.sort(base_df.score.to_numpy())
    ev = df[df.fc_id.isin(eval_fc)].copy()

    rows = []
    grid = [("none", 0.0), ("zscore", 0.0), ("ranknorm", 0.0)]
    grid += [("center", a) for a in (0.25, 0.5, 0.75, 1.0)]
    grid += [("topk", a) for a in (0.25, 0.5, 0.75, 1.0)]
    for method, alpha in grid:
        ev["_k"] = apply_transform(ev, st, method, alpha, gm, ecdf, gsorted)
        r = evaluate(ev, "_k")
        r.update({"method": method, "alpha": alpha if method in ("center", "topk") else None})
        rows.append(r)

    out = pd.DataFrame(rows)[
        ["method", "alpha", "n_fc", "rank1_judged", "rank1_hit_pct",
         "MRR", "median_rank_of_correct", "p@5_pct", "pool_auc"]
    ]
    base = out[out.method == "none"].rank1_hit_pct.iloc[0]
    out["delta_pp"] = (out.rank1_hit_pct - base).round(2)
    out = out.sort_values("rank1_hit_pct", ascending=False)

    tag = f"{args.model}_{args.baseline_from}"
    out.to_csv(RESULTS / f"debias_{tag}.csv", index=False)
    print(f"\n=== {args.model} / 基線來源={args.baseline_from} ===")
    print(out.to_string(index=False))
    print(f"\n-> results/debias_{tag}.csv")


if __name__ == "__main__":
    main()
