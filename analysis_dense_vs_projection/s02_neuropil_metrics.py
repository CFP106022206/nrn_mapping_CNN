"""步驟 2 - neuropil 佔位描述子 (僅 FlyCircuit 神經)。

輸入: data/neuron1x1Coding_Ver2.csv -- 每顆神經在 58 個 compartment
      (29 個解剖腦區 x {左, 右}) 的 voxel 數, 外加 `volume` (總量) 與
      `other` (落在所有具名 neuropil 之外的 voxel)。
輸出: results/neuropil_metrics.csv

每個集中度統計量都算兩種正規化:
  * side 層級 (58 格) -- 一顆同時分布在 mb_4_l 與 mb_4_r 的神經算成兩格
  * region 層級 (29 格) -- 同一顆神經算成一個腦區, 因此統計量衡量的是
    「跨腦區投射」而不是「左右對稱」。
region 層級才是對應形態學主張 (「dense = 單一 neuropil」對「projection =
兩個 neuropil」) 的版本。

分母另外分成兩種, 兩種都算 (已驗證 neuropil 總和 + other == volume, 100 % 吻合):
  * 分母 = 58 個 neuropil 的總和 (排除 other) -- 問的是「落在具名腦區內的那部分
    arbor 是怎麼分布的」, 前綴 side_ / region_
  * 分母 = volume (含 other) -- 問的是「佔這顆神經的總體積多少比例」, 前綴
    sidetot_ / regiontot_
兩者不等價: other 佔比本身在兩組間就有差 (projection 0.225 vs dense 0.161),
所以排除 other 會系統性地把 other 較多的那一組的佔比灌大。要主張「佔總體積
百分之多少」時必須用含 other 的版本。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C


def concentration_stats(P: np.ndarray, prefix: str, min_share: float) -> dict:
    """列和為 1 的矩陣 P (n x k) 的集中度 / 分散度描述子。"""
    S = np.sort(P, axis=1)[:, ::-1]
    p1, p2, p3 = S[:, 0], S[:, 1], S[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    hhi = (P ** 2).sum(axis=1)
    return {
        f"{prefix}_top1": p1,
        f"{prefix}_top2": p2,
        f"{prefix}_top1_2_sum": p1 + p2,
        f"{prefix}_top3_sum": p1 + p2 + p3,
        # 前兩大 arbor 的均衡程度: 接近 0 = 單一團塊, 接近 1 = 兩個等大的 arbor
        f"{prefix}_balance21": np.divide(p2, p1, out=np.zeros_like(p2), where=p1 > 0),
        f"{prefix}_dominance_gap": p1 - p2,
        f"{prefix}_hhi": hhi,
        f"{prefix}_entropy": H,
        f"{prefix}_neff_shannon": np.exp(H),
        f"{prefix}_neff_simpson": np.divide(1.0, hhi, out=np.full_like(hhi, np.nan), where=hhi > 0),
        f"{prefix}_n_above_thr": (P >= min_share).sum(axis=1).astype(float),
        f"{prefix}_n_above_10pct": (P >= 0.10).sum(axis=1).astype(float),
        f"{prefix}_n_above_20pct": (P >= 0.20).sum(axis=1).astype(float),
    }


def main() -> pd.DataFrame:
    roster = pd.read_csv(C.OUT / "neuron_roster.csv")
    fc = roster[roster.source == "FC"].copy()
    fc["neuron_id"] = fc["neuron_id"].astype(str)

    code = pd.read_csv(C.NEUROPIL_CSV)
    code["neuron"] = code["neuron"].astype(str)
    npil_cols = [c for c in code.columns if c not in (["neuron"] + C.META_COLS)]
    assert len(npil_cols) == 58, len(npil_cols)

    code = code.set_index("neuron")
    have = fc["neuron_id"].isin(code.index)
    print(f"FC neurons with neuropil coding: {have.sum()}/{len(fc)}")
    fc = fc[have].reset_index(drop=True)

    sub = code.loc[fc["neuron_id"].to_numpy()]
    V = sub[npil_cols].to_numpy(float)                # side-resolved voxel counts
    volume = sub["volume"].to_numpy(float)
    other = sub["other"].to_numpy(float)

    # region 層級 (去掉結尾的 _l / _r 後合併)
    regions = sorted({c.rsplit("_", 1)[0] for c in npil_cols})
    R = np.zeros((V.shape[0], len(regions)))
    for j, r in enumerate(regions):
        cols = [i for i, c in enumerate(npil_cols) if c.rsplit("_", 1)[0] == r]
        R[:, j] = V[:, cols].sum(axis=1)

    tot = V.sum(axis=1)
    ok = (tot > 0) & (volume > 0)
    # 分母一: 只算落在具名 neuropil 內的部分
    P = np.divide(V, tot[:, None], out=np.zeros_like(V), where=tot[:, None] > 0)
    Q = np.divide(R, tot[:, None], out=np.zeros_like(R), where=tot[:, None] > 0)
    # 分母二: 總體積 (含 other), 對應「佔總體積百分之多少」的說法
    Pt = np.divide(V, volume[:, None], out=np.zeros_like(V), where=volume[:, None] > 0)
    Qt = np.divide(R, volume[:, None], out=np.zeros_like(R), where=volume[:, None] > 0)

    out = {"neuron_id": fc["neuron_id"], "group": fc["group"],
           "exclusive": fc["exclusive"], "n_pairs": fc["n_pairs"]}
    out.update(concentration_stats(P, "side", C.MIN_SHARE))
    out.update(concentration_stats(Q, "region", C.MIN_SHARE))
    out.update(concentration_stats(Pt, "sidetot", C.MIN_SHARE))
    out.update(concentration_stats(Qt, "regiontot", C.MIN_SHARE))

    top_region = np.array(regions)[Q.argmax(axis=1)]
    lr = np.zeros(V.shape[0])
    for i, r in enumerate(top_region):
        li = npil_cols.index(f"{r}_l") if f"{r}_l" in npil_cols else None
        ri = npil_cols.index(f"{r}_r") if f"{r}_r" in npil_cols else None
        l, rr = (V[i, li] if li is not None else 0.0), (V[i, ri] if ri is not None else 0.0)
        lr[i] = abs(l - rr) / (l + rr) if (l + rr) > 0 else np.nan

    out.update({
        "top_region": top_region,
        "second_region": np.array(regions)[np.argsort(Q, axis=1)[:, -2]],
        # 主腦區的偏側性: 1 = 完全單側, 0 = 左右對稱
        "top_region_laterality": lr,
        # 重建結果中落在所有具名 neuropil 之外的比例 (通常是纖維束)
        "other_fraction": np.divide(other, volume, out=np.full_like(other, np.nan), where=volume > 0),
        "neuropil_fraction": np.divide(tot, volume, out=np.full_like(tot, np.nan), where=volume > 0),
        "total_voxels": volume,
        "neuropil_voxels": tot,
    })

    df = pd.DataFrame(out)
    df = df[ok]
    df.to_csv(C.OUT / "neuropil_metrics.csv", index=False)
    np.save(C.OUT / "neuropil_profile_region.npy", Q)
    pd.Series(regions).to_csv(C.OUT / "neuropil_region_names.csv", index=False, header=["region"])
    print(df.groupby("group")[["region_top1", "region_top2", "regiontot_top1",
                               "regiontot_top2", "other_fraction"]].median().round(3))
    return df


if __name__ == "__main__":
    main()
