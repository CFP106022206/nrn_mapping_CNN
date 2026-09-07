"""步驟 9 - 直接檢驗當初的挑選標準是否在資料上成立。

問題:
  (1) D5 是不是主要橫跨「兩個特定腦區」? D2+D6 是不是主要集中在「單一腦區」?
  (2) 如果是, 佔位比例的門檻該訂在哪裡?

佔比一律同時給兩種分母 (已驗證 neuropil 總和 + other == volume):
  share_*        分母 = 58 個 neuropil 的總和 (排除 other)
  share_*_total  分母 = volume (含 other) -- 「佔總體積百分之多少」用這個
訂門檻時以 _total 版本為準, 因為 other 佔比本身兩組就有差 (0.225 vs 0.161),
排除 other 會把 other 多的那一組灌大。

輸出: results/region_identity.csv       每顆 FC 神經的前兩名腦區與佔比
      results/region_top1_counts.csv    各組主腦區的出現次數
      results/region_pair_counts.csv    各組「前兩名腦區」配對的出現次數
      results/region_threshold_scan.csv 各候選門檻的覆蓋率與誤收率
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K


def build() -> pd.DataFrame:
    roster = pd.read_csv(C.OUT / "neuron_roster.csv")
    fc = roster[(roster.source == "FC") & roster.exclusive].copy()
    fc["neuron_id"] = fc["neuron_id"].astype(str)

    code = pd.read_csv(C.NEUROPIL_CSV)
    code["neuron"] = code["neuron"].astype(str)
    npil_cols = [c for c in code.columns if c not in (["neuron"] + C.META_COLS)]
    code = code.set_index("neuron")
    fc = fc[fc.neuron_id.isin(code.index)].reset_index(drop=True)
    sub = code.loc[fc.neuron_id.to_numpy()]

    V = sub[npil_cols].to_numpy(float)
    volume = sub["volume"].to_numpy(float)
    other = sub["other"].to_numpy(float)
    regions = sorted({c.rsplit("_", 1)[0] for c in npil_cols})
    R = np.column_stack([V[:, [i for i, c in enumerate(npil_cols)
                               if c.rsplit("_", 1)[0] == r]].sum(axis=1) for r in regions])
    tot = R.sum(axis=1)

    order = np.argsort(R, axis=1)[:, ::-1]
    idx = np.arange(len(R))
    r1, r2 = order[:, 0], order[:, 1]
    v1, v2 = R[idx, r1], R[idx, r2]

    # side 層級 (左右分開) 的前兩名, 用來區分「跨腦區」與「跨左右半腦」
    so = np.argsort(V, axis=1)[:, ::-1]
    s1, s2 = so[:, 0], so[:, 1]
    sv1, sv2 = V[idx, s1], V[idx, s2]
    same_region_lr = np.array([npil_cols[a].rsplit("_", 1)[0] == npil_cols[b].rsplit("_", 1)[0]
                               for a, b in zip(s1, s2)])

    df = pd.DataFrame({
        "neuron_id": fc.neuron_id, "group": fc.group,
        "region_1": np.array(regions)[r1], "region_2": np.array(regions)[r2],
        "vox_1": v1, "vox_2": v2, "vox_neuropil": tot,
        "vox_total": volume, "vox_other": other, "other_frac": other / volume,
        # 分母 = 具名 neuropil 總和 (排除 other)
        "share_1": v1 / tot, "share_2": v2 / tot, "share_12": (v1 + v2) / tot,
        # 分母 = 總體積 (含 other) -- 對應「佔總體積百分之多少」
        "share_1_total": v1 / volume, "share_2_total": v2 / volume,
        "share_12_total": (v1 + v2) / volume,
        "side_1": np.array(npil_cols)[s1], "side_2": np.array(npil_cols)[s2],
        "side_share_1": sv1 / tot, "side_share_2": sv2 / tot,
        "top2_sides_same_region": same_region_lr,
    })
    df["region_pair"] = [" + ".join(sorted([a, b])) for a, b in zip(df.region_1, df.region_2)]
    df.to_csv(C.OUT / "region_identity.csv", index=False)
    return df


def identity_tables(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("(1) 主腦區 (region_1) 的分布")
    t1 = (df.groupby(["group", "region_1"]).size().rename("n")
          .reset_index().sort_values(["group", "n"], ascending=[True, False]))
    t1["pct"] = t1.n / t1.groupby("group").n.transform("sum") * 100
    t1.to_csv(C.OUT / "region_top1_counts.csv", index=False)
    for g in C.GROUP_ORDER:
        s = t1[t1.group == g]
        print(f"\n  {g}  (共 {s.n.sum()} 顆, {len(s)} 個不同主腦區)")
        print(s.head(8)[["region_1", "n", "pct"]].to_string(index=False,
              float_format=lambda v: f"{v:.1f}"))

    print("\n" + "=" * 78)
    print("(2) 前兩名腦區的配對")
    t2 = (df.groupby(["group", "region_pair"]).size().rename("n")
          .reset_index().sort_values(["group", "n"], ascending=[True, False]))
    t2["pct"] = t2.n / t2.groupby("group").n.transform("sum") * 100
    t2.to_csv(C.OUT / "region_pair_counts.csv", index=False)
    for g in C.GROUP_ORDER:
        s = t2[t2.group == g]
        print(f"\n  {g}  (共 {len(s)} 種配對)")
        print(s.head(8)[["region_pair", "n", "pct"]].to_string(index=False,
              float_format=lambda v: f"{v:.1f}"))

    print("\n" + "=" * 78)
    print("(3) side 層級的前兩名是否落在同一腦區的左右兩側")
    print(df.groupby("group")["top2_sides_same_region"]
            .agg(n="size", same_region_lr="sum")
            .assign(pct=lambda d: d.same_region_lr / d.n * 100).round(1).to_string())


def threshold_scan(df: pd.DataFrame) -> pd.DataFrame:
    """列出候選門檻: 覆蓋率 (該組有多少比例通過) 與誤收率 (另一組被誤收多少)。"""
    dense = df[df.group == C.GROUP_DENSE]
    proj = df[df.group == C.GROUP_PROJ]
    rows = []

    for col in ("share_1", "share_1_total"):
        for t in np.arange(0.30, 0.86, 0.05):
            rows.append({"rule": f"{col} >= {t:.2f}", "target": C.GROUP_DENSE,
                         "coverage_of_target": (dense[col] >= t).mean(),
                         "false_intake_from_other": (proj[col] >= t).mean()})
    for col in ("share_2", "share_2_total"):
        for t in np.arange(0.10, 0.46, 0.02):
            rows.append({"rule": f"{col} >= {t:.2f}", "target": C.GROUP_PROJ,
                         "coverage_of_target": (proj[col] >= t).mean(),
                         "false_intake_from_other": (dense[col] >= t).mean()})
            rows.append({"rule": f"{col} <= {t:.2f}", "target": C.GROUP_DENSE,
                         "coverage_of_target": (dense[col] <= t).mean(),
                         "false_intake_from_other": (proj[col] <= t).mean()})
    for col in ("share_12", "share_12_total"):
        for t in np.arange(0.50, 1.01, 0.05):
            rows.append({"rule": f"{col} >= {t:.2f}", "target": C.GROUP_PROJ,
                         "coverage_of_target": (proj[col] >= t).mean(),
                         "false_intake_from_other": (dense[col] >= t).mean()})
    for t in (10000, 20000, 30000, 40000, 60000, 80000):
        rows.append({"rule": f"vox_1 >= {t}", "target": C.GROUP_DENSE,
                     "coverage_of_target": (dense.vox_1 >= t).mean(),
                     "false_intake_from_other": (proj.vox_1 >= t).mean()})
        rows.append({"rule": f"vox_1 + vox_2 >= {t}", "target": C.GROUP_PROJ,
                     "coverage_of_target": (proj.vox_1 + proj.vox_2 >= t).mean(),
                     "false_intake_from_other": (dense.vox_1 + dense.vox_2 >= t).mean()})

    out = pd.DataFrame(rows)
    out["margin"] = out.coverage_of_target - out.false_intake_from_other
    out.to_csv(C.OUT / "region_threshold_scan.csv", index=False)

    print("\n" + "=" * 78)
    print("(4) 各佔位量的分位數 (供訂門檻)")
    for f in ("share_1", "share_1_total", "share_2", "share_2_total",
              "share_12", "share_12_total", "other_frac", "vox_1", "vox_total"):
        print(f"\n  {f}")
        print(df.groupby("group")[f].describe(
            percentiles=[.05, .10, .25, .5, .75, .90, .95])
            .drop(columns=["count", "mean", "std"]).round(3).to_string())

    print("\n" + "=" * 78)
    print("(5) 門檻掃描 (coverage = 目標組通過率, false = 另一組被誤收率)")
    for tgt in C.GROUP_ORDER:
        s = out[out.target == tgt].sort_values("margin", ascending=False).head(8)
        print(f"\n  收 {tgt} 的規則, 依 margin 排序")
        print(s[["rule", "coverage_of_target", "false_intake_from_other", "margin"]]
              .round(3).to_string(index=False))
    return out


def composite_rules(df: pd.DataFrame) -> pd.DataFrame:
    """把「特定腦區」與「佔位量」合起來, 這才是當初挑選流程的樣子。"""
    D5_REGIONS = {"mb_4", "dfp_5"}
    in_d5_regions = df.apply(lambda r: {r.region_1, r.region_2} == D5_REGIONS, axis=1)

    cand = {
        "D5-a  前兩名腦區 = {mb_4, dfp_5}": in_d5_regions,
        "D5-b  同上 且 share_2_total >= 0.20": in_d5_regions & (df.share_2_total >= 0.20),
        "D5-c  同上 且 share_2_total >= 0.25": in_d5_regions & (df.share_2_total >= 0.25),
        "D5-d  同上 且 share_12_total >= 0.65": in_d5_regions & (df.share_12_total >= 0.65),
        "D5-e  主腦區屬 {mb_4,dfp_5} 且 share_2_total>=0.20": df.region_1.isin(D5_REGIONS) & (df.share_2_total >= 0.20),
        "D2-a  vox_1 >= 20000": df.vox_1 >= 20000,
        "D2-b  vox_1 >= 30000": df.vox_1 >= 30000,
        "D2-c  vox_1 >= 20000 且 share_2_total <= 0.25": (df.vox_1 >= 20000) & (df.share_2_total <= 0.25),
        "D2-d  vox_1 >= 20000 且 前兩名腦區 != {mb_4,dfp_5}": (df.vox_1 >= 20000) & ~in_d5_regions,
        "D2-e  vox_1 >= 15000 且 share_2_total <= 0.25 且 非D5腦區": (df.vox_1 >= 15000) & (df.share_2_total <= 0.25) & ~in_d5_regions,
    }
    rows = []
    for name, m in cand.items():
        tgt = C.GROUP_PROJ if name.startswith("D5") else C.GROUP_DENSE
        oth = C.GROUP_DENSE if tgt == C.GROUP_PROJ else C.GROUP_PROJ
        t = df.group == tgt
        o = df.group == oth
        tp, fp = int((m & t).sum()), int((m & o).sum())
        rows.append({"rule": name, "target": tgt,
                     "coverage": tp / t.sum(), "false_intake": fp / o.sum(),
                     "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "n_selected": int(m.sum()), "n_target": int(t.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "region_composite_rules.csv", index=False)
    print("\n" + "=" * 78)
    print("(6) 複合規則 (貼近實際挑選流程)")
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    d = build()
    identity_tables(d)
    threshold_scan(d)
    composite_rules(d)
