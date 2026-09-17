"""用外部資料庫的策展型別，比較 finetune 與 annotator 兩個模型的全庫掃描結果。

作法
----
1. 取出 VFB 有給策展型別、且兩份掃描都涵蓋的 FlyCircuit 神經。
2. 只保留型別能明確對應到 hemibrain 的族群（LC 視覺投射、Kenyon cell、嗅覺投射神經）。
3. 對兩個模型各自的 top5 候選，查 neuPrint 的 hemibrain 型別。
4. 依 type_match 的規則判定每筆配對，統計兩個模型的命中率。

輸出（results/）
---------------
- `pair_labels.csv`   每筆配對一列，含判定與依據。
- `model_summary.csv` 兩個模型的比較。
- `by_family.csv`     分族群的比較。

執行： python3 analysis_external_validation/run_type_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from type_match import expectation, grade  # noqa: E402
from type_reference import fetch_em_types, fetch_fc_types, load_or_fetch  # noqa: E402

PROJECT = ROOT.parent
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"

SCANS = {
    "finetune": PROJECT / "result" / "fc_all_top5.csv",
    "annotator": PROJECT / "result" / "fc_all_top5_annotator_notrunc.csv",
}


def family_of(vfb_type: str) -> str:
    if vfb_type.startswith("lobula columnar"):
        return "LC"
    if "Kenyon" in vfb_type:
        return "KC"
    return "ALPN"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # --- 掃描結果 -----------------------------------------------------
    scans = {}
    for name, path in SCANS.items():
        df = pd.read_csv(path).rename(columns={"similarity_score": "score"})
        scans[name] = df
    covered = set.intersection(*(set(d.source_id) for d in scans.values()))

    # --- FC 端型別 ----------------------------------------------------
    fc_types = load_or_fetch(CACHE / "fc_types_vfb.csv", fetch_fc_types)
    fc_types = fc_types[fc_types.fc_id.isin(covered)].copy()

    # 一顆 FC 可能有多個 VFB 標註，只留能對應到 hemibrain 的那一個
    fc_types["exp"] = fc_types.vfb_type.map(expectation)
    fc_types = fc_types[fc_types.exp.notna()]
    # 同一顆若仍有多筆，優先取非 vague 的
    fc_types["_v"] = fc_types.exp.map(lambda e: e.vague)
    fc_types = fc_types.sort_values("_v").drop_duplicates("fc_id").drop(columns="_v")
    fc_types["family"] = fc_types.vfb_type.map(family_of)
    print(f"可評估的 FC 神經：{len(fc_types)}")
    print(fc_types.family.value_counts().to_string())

    keep = set(fc_types.fc_id)

    # --- 配對表 -------------------------------------------------------
    frames = []
    for name, df in scans.items():
        s = df[df.source_id.isin(keep)][["source_id", "target_id", "score", "rank"]]
        s = s.rename(columns={f: f"{f}_{name}" for f in ("score", "rank")})
        frames.append(s.set_index(["source_id", "target_id"]))
    pairs = pd.concat(frames, axis=1).reset_index()
    pairs = pairs.rename(columns={"source_id": "fc_id", "target_id": "em_id"})
    print(f"待判定配對：{len(pairs)}")

    # --- EM 端型別 ----------------------------------------------------
    em_types = load_or_fetch(
        CACHE / "em_types_neuprint.csv",
        fetch_em_types,
        body_ids=sorted(set(pairs.em_id)),
    )
    em_types["em_id"] = em_types.em_id.astype("int64")
    for c in ("np_type", "np_instance", "np_status"):
        em_types[c] = em_types[c].fillna("")

    d = pairs.merge(fc_types[["fc_id", "vfb_type", "exp", "family"]], on="fc_id")
    d = d.merge(em_types, on="em_id", how="left")
    for c in ("np_type", "np_instance", "np_status"):
        d[c] = d[c].fillna("")

    graded = [grade(r.exp, r.np_type, r.np_instance) for r in d.itertuples()]
    d["relation"] = [g[0] for g in graded]
    d["type_label"] = [g[1] for g in graded]
    d["basis"] = [g[2] for g in graded]

    # --- 可贏性：正解有沒有進候選池 -----------------------------------
    ceil_path = CACHE / "fc_pool_ceiling.csv"
    if ceil_path.exists():
        ceil = pd.read_csv(ceil_path)[
            ["fc_id", "pool_size", "n_correct_in_pool", "chance_pct"]
        ]
        d = d.merge(ceil, on="fc_id", how="left")
    else:
        print("!! 找不到 cache/fc_pool_ceiling.csv，先跑 build_pool_ceiling.py "
              "才會有隨機基準與可贏性欄位")
        d["pool_size"] = d["n_correct_in_pool"] = d["chance_pct"] = pd.NA
    d["winnable"] = d.n_correct_in_pool.fillna(0) > 0

    # --- 輸出配對表 ---------------------------------------------------
    out = d[
        [
            "fc_id", "em_id", "family", "vfb_type", "np_type", "np_instance", "np_status",
            "rank_finetune", "score_finetune", "rank_annotator", "score_annotator",
            "relation", "type_label", "basis",
            "pool_size", "n_correct_in_pool", "chance_pct", "winnable",
        ]
    ].sort_values(["fc_id", "type_label"], ascending=[True, False])
    out.to_csv(RESULTS / "pair_labels.csv", index=False)
    print(f"\n-> results/pair_labels.csv（{len(out)} 列）")

    def stats(sub: pd.DataFrame, name: str, extra: dict) -> dict:
        rc = f"rank_{name}"
        m = sub[sub[rc].notna()]
        judged = m[m.type_label.notna()]
        t1 = m[(m[rc] == 1) & m.type_label.notna()]
        # 每顆 FC 的 top-5 裡有沒有任何一個型別相符的候選。
        # 這與 pair_same_type_pct（每「對」的同型率）是不同的量，
        # 先前兩者共用 hit_top5_pct 這個名字，會誤導。
        per_fc = m.groupby("fc_id").type_label.apply(lambda s: (s == 1.0).any())
        row = dict(extra)
        row.update(
            {
                "model": name,
                "n_fc": m.fc_id.nunique(),
                "top5_pairs": len(m),
                "judgeable": len(judged),
                "same_type_1.0": int((judged.type_label == 1.0).sum()),
                "uncertain_0.5": int((judged.type_label == 0.5).sum()),
                "diff_type_0.0": int((judged.type_label == 0.0).sum()),
                # 每「對」的同型率
                "pair_same_type_pct": round(100 * (judged.type_label == 1.0).mean(), 1) if len(judged) else None,
                # 每「顆 FC」的 top-5 是否含正解
                "fc_top5_has_correct_pct": round(100 * per_fc.mean(), 1) if len(per_fc) else None,
                "rank1_n": len(t1),
                "rank1_hit_pct": round(100 * (t1.type_label == 1.0).mean(), 1) if len(t1) else None,
                "rank1_wrong_pct": round(100 * (t1.type_label == 0.0).mean(), 1) if len(t1) else None,
                "chance_pct": round(m.chance_pct.median(), 2) if m.chance_pct.notna().any() else None,
            }
        )
        return row

    # --- 模型比較（全體 / 只看正解有進池的） ---------------------------
    rows = []
    for scope, sub in (("all", d), ("winnable", d[d.winnable])):
        for name in SCANS:
            rows.append(stats(sub, name, {"scope": scope}))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "model_summary.csv", index=False)
    print("\n=== 模型比較 ===")
    print(summary.to_string(index=False))

    # --- 分族群（只看可贏的） -----------------------------------------
    rows = []
    for fam in sorted(d.family.unique()):
        sub = d[(d.family == fam) & d.winnable]
        if not len(sub):
            continue
        for name in SCANS:
            rows.append(stats(sub, name, {"family": fam}))
    by_fam = pd.DataFrame(rows)
    by_fam.to_csv(RESULTS / "by_family.csv", index=False)
    print("\n=== 分族群（只算正解有進候選池的 FC）===")
    print(by_fam.to_string(index=False))

    # --- 配對顯著性檢定（同一批 FC，兩個模型的 rank-1 命中）-------------
    from scipy.stats import binomtest

    w = d[d.winnable]
    hits = {}
    for name in SCANS:
        s = w[(w[f"rank_{name}"] == 1) & w.type_label.notna()]
        hits[name] = s.set_index("fc_id").type_label.eq(1.0)
    paired = pd.concat([hits["finetune"].rename("ft"), hits["annotator"].rename("an")], axis=1).dropna()
    fam_of = w.drop_duplicates("fc_id").set_index("fc_id").family

    rows = []
    for scope, sub in [("all", paired)] + [
        (f, paired[paired.index.map(fam_of) == f]) for f in sorted(fam_of.unique())
    ]:
        if not len(sub):
            continue
        b = int((sub.ft & ~sub.an).sum())
        c = int((~sub.ft & sub.an).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        rows.append(
            {
                "scope": scope,
                "n_fc": len(sub),
                "finetune_only_correct": b,
                "annotator_only_correct": c,
                "both_correct": int((sub.ft & sub.an).sum()),
                "p_value": f"{p:.3g}",
                "better": "annotator" if c > b else "finetune" if b > c else "tie",
                "significant_p05": bool(p < 0.05),
            }
        )
    mcnemar = pd.DataFrame(rows)
    mcnemar.to_csv(RESULTS / "mcnemar.csv", index=False)
    print("\n=== 配對檢定（McNemar，rank-1 命中）===")
    print(mcnemar.to_string(index=False))

    n_pos = int((out.type_label == 1.0).sum())
    n_neg = int((out.type_label == 0.0).sum())
    print(f"\n名單：確定同型 {n_pos} 筆、確定不同型 {n_neg} 筆、不確定 "
          f"{int((out.type_label == 0.5).sum())} 筆、無法判斷 {int(out.type_label.isna().sum())} 筆")


if __name__ == "__main__":
    main()
