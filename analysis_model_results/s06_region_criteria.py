"""步驟 9 - 直接檢驗當初的挑選標準是否在資料上成立。

問題:
  (1) D5 是不是主要橫跨「兩個特定腦區」? D2+D6 是不是主要集中在「單一腦區」?
  (2) 如果是, 佔位比例的門檻該訂在哪裡?

佔比的分母一律是 volume (= 58 個 neuropil + other, 已驗證完全相等), 即完整的
tracing 點數, 欄位以 _total 結尾標明。不提供排除 other 的版本: other 佔比本身兩組
就有差 (0.225 vs 0.161), 排除它會把 other 多的那一組灌大。

輸出: results/region_identity.csv       每顆 FC 神經的前兩名腦區與佔比
      results/region_top1_counts.csv    各組主腦區的出現次數
      results/region_pair_counts.csv    各組「前兩名腦區」配對的出現次數
      results/region_threshold_scan.csv 各候選門檻的覆蓋率與誤收率
      results/region_composite_rules.csv 貼近實際挑選流程的複合規則
      results/region_wording_support.csv 論文用詞的數據依據 (MB/DFP、嗅覺腦區、同側性)
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
        # 分母 = volume (完整 tracing 點數, 含 other)
        "share_1_total": v1 / volume, "share_2_total": v2 / volume,
        "share_12_total": (v1 + v2) / volume,
        "side_1": np.array(npil_cols)[s1], "side_2": np.array(npil_cols)[s2],
        "side_share_1_total": sv1 / volume, "side_share_2_total": sv2 / volume,
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

    for col in ("share_1_total",):
        for t in np.arange(0.30, 0.86, 0.05):
            rows.append({"rule": f"{col} >= {t:.2f}", "target": C.GROUP_DENSE,
                         "coverage_of_target": (dense[col] >= t).mean(),
                         "false_intake_from_other": (proj[col] >= t).mean()})
    for col in ("share_2_total",):
        for t in np.arange(0.10, 0.46, 0.02):
            rows.append({"rule": f"{col} >= {t:.2f}", "target": C.GROUP_PROJ,
                         "coverage_of_target": (proj[col] >= t).mean(),
                         "false_intake_from_other": (dense[col] >= t).mean()})
            rows.append({"rule": f"{col} <= {t:.2f}", "target": C.GROUP_DENSE,
                         "coverage_of_target": (dense[col] <= t).mean(),
                         "false_intake_from_other": (proj[col] <= t).mean()})
    for col in ("share_12_total",):
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
    for f in ("share_1_total", "share_2_total", "share_12_total",
              "other_frac", "vox_1", "vox_total"):
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


# 論文用詞的依據 ---------------------------------------------------------------
# 經典嗅覺 projection neuron 的路徑是 AL -> MB calyx -> LH。FlyCircuit 把 calyx
# 獨立編碼為 cal_18, 因此 mb_4 是 calyx 以外的 MB。
D1_REGIONS = ("mb_4", "dfp_5")
OLFACTORY_REGIONS = ("al_3", "cal_18", "lh_25")
SHARE_THRESHOLDS = (0.01, 0.05, 0.10, 0.15, 0.20)


def region_shares_total() -> pd.DataFrame:
    """每顆 FC 神經在各腦區 (左右合併) 佔總重建量的比例。分母 = volume (含 other)。

    neuron1x1Coding 的數值與 cable 長度成正比, 所以這個比例可讀成「該神經有多少
    比例的神經突落在此腦區」。它量的是神經突在哪, 不是突觸, 也不是 soma 位置。
    """
    roster = pd.read_csv(C.OUT / "neuron_roster.csv")
    fc = roster[(roster.source == "FC") & roster.exclusive].copy()
    fc["neuron_id"] = fc["neuron_id"].astype(str)
    code = pd.read_csv(C.NEUROPIL_CSV)
    code["neuron"] = code["neuron"].astype(str)
    npil_cols = [c for c in code.columns if c not in (["neuron"] + C.META_COLS)]
    code = code.set_index("neuron")
    fc = fc[fc.neuron_id.isin(code.index)].reset_index(drop=True)
    sub = code.loc[fc.neuron_id.to_numpy()]
    regions = sorted({c.rsplit("_", 1)[0] for c in npil_cols})
    R = pd.DataFrame({r: sub[[c for c in npil_cols if c.rsplit("_", 1)[0] == r]]
                      .sum(axis=1).to_numpy() for r in regions})
    share = R.div(sub["volume"].to_numpy(), axis=0)
    share.insert(0, "group", fc.group.to_numpy())
    share.insert(0, "neuron_id", fc.neuron_id.to_numpy())
    return share


def wording_support(df: pd.DataFrame) -> pd.DataFrame:
    """論文用詞的數據依據。

    (a) 主腦區 (佔比最大的腦區) 是 MB 或 DFP -- 論文採用的 D1 描述
        並檢查以 DFP 為主的神經在兩組間差在哪 (D2 也有 34% 以 DFP 為主)
    (b) MB / DFP 佔比 >= 門檻 (任一 / 皆達標), 掃門檻看敏感度, 作為參考
    (c) D1 裡 MB 或 DFP 佔比 < 5% 的例外
    (d) 嗅覺迴路腦區 (AL / calyx / LH): 檢驗 D1 是否為嗅覺 projection neuron
    (e) 同側性: 前兩名 side 層級 compartment 是否在同一半腦
    """
    share = region_shares_total()
    regions = [c for c in share.columns if c not in ("neuron_id", "group")]
    is_g = {g: share.group == g for g in C.GROUP_ORDER}
    rows = []
    print("\n" + "=" * 78)
    print("(7) 論文用詞的數據依據  (分母 = volume, 含 other)")
    print(f"    n: D1 {int(is_g[C.GROUP_PROJ].sum())}, D2 {int(is_g[C.GROUP_DENSE].sum())}"
          "  (FC, 僅限單一組別且在 neuron1x1Coding 內)")

    print("\n  (a) 主腦區 (佔比最大的腦區)")
    share["region_1"] = share.neuron_id.map(df.set_index("neuron_id").region_1)
    for name, label, hit in (("top1_mb_or_dfp", "MB 或 DFP", share.region_1.isin(D1_REGIONS)),
                             ("top1_mb", "MB", share.region_1 == "mb_4"),
                             ("top1_dfp", "DFP", share.region_1 == "dfp_5")):
        v = {g: hit[is_g[g]] for g in C.GROUP_ORDER}
        rows.append({"check": name, "frac_projection": float(v[C.GROUP_PROJ].mean()),
                     "frac_dense": float(v[C.GROUP_DENSE].mean())})
        print(f"      主腦區 = {label:9s} D1 {int(v[C.GROUP_PROJ].sum()):3d}"
              f" ({v[C.GROUP_PROJ].mean() * 100:5.1f}%)   D2 {int(v[C.GROUP_DENSE].sum()):3d}"
              f" ({v[C.GROUP_DENSE].mean() * 100:5.1f}%)")
    print("      主腦區 = DFP 的神經, MB 佔比如何 (DFP 本身不分兩組, 分兩組的是 MB):")
    for g in C.GROUP_ORDER:
        s_ = share[is_g[g] & (share.region_1 == "dfp_5")]
        rows.append({"check": "top1_dfp_mb_share", "group": g, "n": len(s_),
                     "frac_mb_ge_0.05": float((s_.mb_4 >= 0.05).mean()),
                     "median": float(s_.mb_4.median())})
        print(f"        {g:11s} n={len(s_):3d}  MB >= 5% {(s_.mb_4 >= 0.05).mean() * 100:5.1f}%"
              f"  MB 中位 {s_.mb_4.median() * 100:5.1f}%")

    print("\n  (b) MB / DFP 佔總重建量 >= 門檻的比例  (或 = 任一達標; 且 = 兩者皆達標)")
    print(f"      {'門檻':>6} {'D1 或':>8} {'D2 或':>8} {'D1 且':>8} {'D2 且':>8}")
    for t in SHARE_THRESHOLDS:
        mb, dfp = share.mb_4 >= t, share.dfp_5 >= t
        for name, hit in (("mb_or_dfp", mb | dfp), ("mb_and_dfp", mb & dfp)):
            rows.append({"check": name, "region": "mb_4|dfp_5" if name == "mb_or_dfp"
                         else "mb_4&dfp_5", "threshold": t,
                         "frac_projection": float(hit[is_g[C.GROUP_PROJ]].mean()),
                         "frac_dense": float(hit[is_g[C.GROUP_DENSE]].mean())})
        o, a = rows[-2], rows[-1]
        print(f"      {t * 100:5.0f}% {o['frac_projection'] * 100:7.1f}% {o['frac_dense'] * 100:7.1f}%"
              f" {a['frac_projection'] * 100:7.1f}% {a['frac_dense'] * 100:7.1f}%")

    print("\n  (c) D1 中 MB 或 DFP 任一 < 5% 的神經")
    ex = share[is_g[C.GROUP_PROJ] & ((share.mb_4 < 0.05) | (share.dfp_5 < 0.05))]
    for r in ex.itertuples():
        top = max(regions, key=lambda c: getattr(r, c))
        print(f"      {r.neuron_id:18s} MB {r.mb_4 * 100:5.1f}%  DFP {r.dfp_5 * 100:5.1f}%"
              f"  主腦區 {top}")

    print("\n  (d) 各腦區佔總重建量  [嗅覺 PN 路徑 = AL -> calyx -> LH]")
    print(f"      {'腦區':8s} {'組':11s} {'中位':>7} {'>=1%':>7} {'>=5%':>7} {'>=10%':>7}")
    for reg in D1_REGIONS + OLFACTORY_REGIONS:
        for g in C.GROUP_ORDER:
            x = share.loc[is_g[g], reg]
            rec = {"check": "region_share", "region": reg, "group": g,
                   "median": float(x.median())}
            for t in (0.01, 0.05, 0.10):
                rec[f"frac_ge_{t:g}"] = float((x >= t).mean())
            rows.append(rec)
            print(f"      {reg:8s} {g:11s} {x.median() * 100:6.1f}% {rec['frac_ge_0.01'] * 100:6.1f}%"
                  f" {rec['frac_ge_0.05'] * 100:6.1f}% {rec['frac_ge_0.1'] * 100:6.1f}%")
    for g in C.GROUP_ORDER:
        s = share[is_g[g]]
        al = s.al_3 >= 0.05
        pn_like = al & (s.cal_18 >= 0.05)
        rows.append({"check": "pn_like", "group": g, "n_al_ge5": int(al.sum()),
                     "n_al_and_calyx_ge5": int(pn_like.sum())})
        print(f"      {g:11s} AL >= 5% 的 {int(al.sum()):2d} 顆中, 同時 calyx >= 5% 的有"
              f" {int(pn_like.sum())} 顆")
    gh = share[share.neuron_id.str.startswith("GH146")]
    for r in gh.itertuples():
        print(f"      {r.neuron_id} ({r.group}): AL {r.al_3 * 100:.1f}%  calyx {r.cal_18 * 100:.1f}%"
              f"  LH {r.lh_25 * 100:.1f}%  MB {r.mb_4 * 100:.1f}%")
    print("      -> D1 幾乎不碰 AL / calyx / LH, 不是嗅覺 PN 的型態。D2 含 AL 的神經"
          "\n         也沒有 calyx, 較像侷限在 AL 內的神經。兩組都不以嗅覺 PN 為主。")

    print("\n  (e) 同側性")
    h1, h2 = df.side_1.str[-1], df.side_2.str[-1]
    for g in C.GROUP_ORDER:
        m = df.group == g
        ipsi = float((h1[m] == h2[m]).mean())
        right = float((h1[m] == "r").mean())
        rows.append({"check": "hemisphere", "group": g, "frac_top2_same_hemisphere": ipsi,
                     "frac_top1_right": right})
        print(f"      {g:11s} 前兩名 compartment 在同一半腦 {ipsi * 100:5.1f}%"
              f"   (主 compartment 在右半腦 {right * 100:5.1f}%, 這是資料的左右分布, 不是同側性)")

    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "region_wording_support.csv", index=False)
    return out


if __name__ == "__main__":
    d = build()
    identity_tables(d)
    threshold_scan(d)
    composite_rules(d)
    wording_support(d)
