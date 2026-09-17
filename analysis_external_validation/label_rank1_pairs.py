"""把一份全庫掃描的 rank-1 配對，用外部策展型別標上「是否同型」。

單一標註的判定規則與 `run_type_check.py` 相同（`type_match.grade`），對象換成
任意一份掃描的 rank-1，且不限於兩個模型都涵蓋的 FC；判定不了的配對也保留，
並在 `relation` / `basis` 說明原因。

與 `run_type_check.py` 的差異：一顆 FC 有多個可對應的 VFB 標註時（例如同時標
「multiglomerular PN」與「ALl1 lineage」），那邊只任取一個，這裡則全部都判。
這些標註描述的是同一顆神經、同時成立，所以任一個判「確定不同」就是不同；
若有標註判相符、另一個判確定不同，代表 VFB 自相矛盾，標為 Uncertain。

`same_type` 欄：

    True       type_label = 1.0（型別相符）
    False      type_label = 0.0（兩邊都有策展型別且確定不同）
    Uncertain  type_label = 0.5（形態近親無法斷定），或 VFB 標註自相矛盾
    Unknown    任一端沒有可用型別，無從判斷（type_label 空白）

執行： python3 analysis_external_validation/label_rank1_pairs.py [掃描檔]
預設掃描檔為 result/fc_find_em_0915.csv，輸出 results/<檔名>_rank1_type.csv。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_type_check import family_of  # noqa: E402
from type_match import expectation, grade  # noqa: E402

PROJECT = ROOT.parent
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"

DEFAULT_SCAN = PROJECT / "result" / "fc_find_em_0915.csv"
EM_FIELDS = ("np_type", "np_instance", "np_status")


def load_fc_types(fc_ids: set[str]) -> pd.DataFrame:
    """每顆 FC 一列：`exps` 為所有能對應 hemibrain 的期望值（可能為空）。"""
    fc = pd.read_csv(CACHE / "fc_types_vfb.csv")
    fc = fc[fc.fc_id.isin(fc_ids)].copy()
    fc["exp"] = fc.vfb_type.map(expectation)
    fc["mapped"] = fc.exp.notna()
    fc["_v"] = fc.exp.map(lambda e: e.vague if e else True)
    fc = fc.sort_values(["fc_id", "_v", "vfb_type"])

    rows = []
    for fc_id, g in fc.groupby("fc_id", sort=False):
        ok = g[g.mapped]
        # 對不上的 FC 保留全部標註供人工檢視；對得上的只列參與判定的標註
        shown = ok if len(ok) else g
        rows.append(
            {
                "fc_id": fc_id,
                "vfb_type": " | ".join(shown.vfb_type),
                "exps": list(ok.exp),
                "family": family_of(ok.vfb_type.iloc[0]) if len(ok) else "",
            }
        )
    return pd.DataFrame(rows)


def load_em_types() -> pd.DataFrame:
    frames = [pd.read_csv(CACHE / f) for f in ("em_types_all_db.csv", "em_types_neuprint.csv")]
    em = pd.concat(frames, ignore_index=True).drop_duplicates("em_id")
    em["em_id"] = em.em_id.astype("int64")
    for c in EM_FIELDS:
        em[c] = em[c].fillna("")
    return em


def judge(r) -> tuple[str, float | None, str]:
    if not r.vfb_type:
        return "fc_untyped", None, "VFB 未給 FlyCircuit 端策展型別，無法判斷"
    if not r.exps:
        return (
            "fc_type_unmapped",
            None,
            f"VFB「{r.vfb_type}」不在可對應 hemibrain 的族群（LC / KC / ALPN），無法判斷",
        )
    if not r.em_found:
        return "em_not_found", None, f"hemibrain v1.2.1 查無 bodyId {r.em_id}，無法判斷"

    grades = [grade(e, r.np_type, r.np_instance) for e in r.exps]
    by_label = {}
    for g in grades:
        by_label.setdefault(g[1], g)
    if 1.0 in by_label and 0.0 in by_label:
        return (
            "vfb_conflict",
            None,
            f"VFB 標註互相矛盾：{by_label[1.0][2]}；但{by_label[0.0][2]}",
        )
    for label in (0.0, 1.0, 0.5):
        if label in by_label:
            return by_label[label]
    return grades[0]


def main() -> None:
    scan_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCAN
    scan = pd.read_csv(scan_path).rename(columns={"similarity_score": "score"})
    d = scan[scan["rank"] == 1][["source_id", "target_id", "score", "rank"]]
    d = d.rename(columns={"source_id": "fc_id", "target_id": "em_id"})
    d["em_id"] = d.em_id.astype("int64")
    print(f"rank-1 配對：{len(d)}")

    d = d.merge(load_fc_types(set(d.fc_id)), on="fc_id", how="left")
    d["vfb_type"] = d.vfb_type.fillna("")
    d["family"] = d.family.fillna("")
    d["exps"] = [e if isinstance(e, list) else [] for e in d.exps]

    em = load_em_types()
    d["em_found"] = d.em_id.isin(set(em.em_id))
    d = d.merge(em, on="em_id", how="left")
    for c in EM_FIELDS:
        d[c] = d[c].fillna("")

    graded = [judge(r) for r in d.itertuples()]
    d["relation"] = [g[0] for g in graded]
    d["type_label"] = [g[1] for g in graded]
    d["basis"] = [g[2] for g in graded]
    d["same_type"] = d.type_label.map({1.0: "True", 0.0: "False", 0.5: "Uncertain"})
    d.loc[d.relation == "vfb_conflict", "same_type"] = "Uncertain"
    d["same_type"] = d.same_type.fillna("Unknown")

    # 候選池資訊只對 build_pool_ceiling 涵蓋的 FC 有值，其餘留空而不是判成不可贏
    ceil = pd.read_csv(CACHE / "fc_pool_ceiling.csv")[
        ["fc_id", "pool_size", "n_correct_in_pool", "chance_pct"]
    ]
    d = d.merge(ceil, on="fc_id", how="left")
    d["winnable"] = (d.n_correct_in_pool > 0).where(d.n_correct_in_pool.notna())

    out = d[
        [
            "fc_id", "em_id", "score", "rank",
            "family", "vfb_type", "np_type", "np_instance", "np_status",
            "relation", "type_label", "same_type", "basis",
            "pool_size", "n_correct_in_pool", "chance_pct", "winnable",
        ]
    ].sort_values("fc_id")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{scan_path.stem}_rank1_type.csv"
    out.to_csv(out_path, index=False)
    print(f"-> {out_path.relative_to(PROJECT)}（{len(out)} 列）\n")

    print("=== same_type × relation ===")
    print(pd.crosstab(out.relation, out.same_type, margins=True).to_string())
    judged = out[out.family != ""]
    print("\n=== 可對應族群內（same_type × family）===")
    print(pd.crosstab(judged.family, judged.same_type, margins=True).to_string())


if __name__ == "__main__":
    main()
