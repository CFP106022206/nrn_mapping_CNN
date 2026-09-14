"""步驟 1 - 建立兩個 sub dataset 的神經名單。

輸入: labeled_info/D2+D6_ID.csv  (dense 組, 論文的 sub dataset D2)
      labeled_info/D5_conf.csv   (projection 組, 論文的 sub dataset D1)
輸出: results/neuron_roster.csv  -- 每列一顆 (neuron_id, source, group)

這兩個標註檔是「配對」清單; 一顆神經的形態不會因為它參與了幾組配對而改變,
所以這裡把神經去重, 並把它參與的配對數另存成一個欄位。同時出現在兩組的神經
會被標記出來, 讓組間比較可以只用互斥的名單。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K


def main() -> pd.DataFrame:
    dense = pd.read_csv(C.DENSE_CSV, index_col=0)[["fc_id", "em_id"]]
    proj = pd.read_csv(C.PROJ_CSV)[["fc_id", "em_id"]]
    dense["group"], proj["group"] = C.GROUP_DENSE, C.GROUP_PROJ

    pairs = pd.concat([dense, proj], ignore_index=True)
    pairs["fc_id"] = pairs["fc_id"].astype(str)
    pairs["em_id"] = pairs["em_id"].astype(str)
    pairs.to_csv(C.OUT / "pairs.csv", index=False)

    rows = []
    for source, col in (("FC", "fc_id"), ("EM", "em_id")):
        g = pairs.groupby([col, "group"]).size().rename("n_pairs").reset_index()
        g = g.rename(columns={col: "neuron_id"})
        g["source"] = source
        rows.append(g)
    roster = pd.concat(rows, ignore_index=True)

    # 同時列在兩組裡的神經不能當作任一類的代表樣本
    dup = roster.groupby(["neuron_id", "source"]).size().rename("n_groups").reset_index()
    roster = roster.merge(dup, on=["neuron_id", "source"])
    roster["exclusive"] = roster["n_groups"] == 1

    roster["swc"] = [K.swc_path(n, s).exists()
                     for n, s in zip(roster.neuron_id, roster.source)]
    roster = roster.sort_values(["group", "source", "neuron_id"]).reset_index(drop=True)
    roster.to_csv(C.OUT / "neuron_roster.csv", index=False)

    print(f"pairs                : {len(pairs)}")
    print(roster.groupby(["group", "source"])
                .agg(n=("neuron_id", "size"),
                     exclusive=("exclusive", "sum"),
                     swc_found=("swc", "sum")))
    return roster


if __name__ == "__main__":
    main()
