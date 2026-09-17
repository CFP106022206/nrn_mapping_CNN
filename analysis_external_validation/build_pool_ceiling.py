"""重建 stage-1 候選池，算出每顆 FC 的「可贏性」與隨機命中基準。

命中率單看數字沒有意義：如果某顆 FC 的正確型別根本沒進候選池，兩個模型都不可能答對。
這支程式用 `candidate_matching.run_matching` 重建完全相同的池（質心 100 µm、
ratio 0.4、rod/disk 方向 gate），再對每顆 FC 算：

- `pool_size`           候選數，應與掃描報告的 n_candidates 一致
- `n_correct_in_pool`   池中型別正確的 EM 數
- `chance_pct`          從池裡隨機抽一顆就答對的機率

輸出：cache/fc_pool_ceiling.csv

執行： python3 analysis_external_validation/build_pool_ceiling.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from type_match import expectation, grade  # noqa: E402

CACHE = ROOT / "cache"


def main() -> None:
    import candidate_matching as cm

    fc_ref = pd.read_csv(CACHE / "fc_types_vfb.csv")
    fc_ref["exp"] = fc_ref.vfb_type.map(expectation)
    fc_ref = fc_ref[fc_ref.exp.notna()]
    fc_ref["_v"] = fc_ref.exp.map(lambda e: e.vague)
    fc_ref = fc_ref.sort_values("_v").drop_duplicates("fc_id").drop(columns="_v")

    with tempfile.TemporaryDirectory() as tmp:
        out = cm.run_matching(out_dir=tmp)
        pool = pd.read_csv(out)

    pool = pool[pool.fc_id.isin(set(fc_ref.fc_id))]

    em = pd.read_csv(CACHE / "em_types_all_db.csv")
    em["np_type"] = em.np_type.fillna("")
    tmap = dict(zip(em.em_id.astype("int64"), em.np_type))
    pool["np_type"] = pool.em_id.astype("int64").map(tmap).fillna("")

    exp_by_fc = dict(zip(fc_ref.fc_id, fc_ref.exp))
    # 同一個 VFB 型別的判定結果可共用，避免重複比對
    verdict: dict[tuple[str, str], bool] = {}

    rows = []
    for fc_id, g in pool.groupby("fc_id", sort=False):
        e = exp_by_fc[fc_id]
        n_ok = 0
        for t in g.np_type:
            if not t:
                continue
            key = (e.label, t)
            if key not in verdict:
                verdict[key] = grade(e, t)[1] == 1.0
            n_ok += verdict[key]
        rows.append(
            {
                "fc_id": fc_id,
                "vfb_type": e.label,
                "pool_size": len(g),
                "n_correct_in_pool": n_ok,
                "chance_pct": round(100 * n_ok / len(g), 4),
            }
        )

    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE / "fc_pool_ceiling.csv", index=False)
    print(f"-> cache/fc_pool_ceiling.csv（{len(df)} 顆 FC）")
    print(f"   池中有正解的 FC：{int((df.n_correct_in_pool > 0).sum())} "
          f"({100 * (df.n_correct_in_pool > 0).mean():.1f}%)")


if __name__ == "__main__":
    main()
