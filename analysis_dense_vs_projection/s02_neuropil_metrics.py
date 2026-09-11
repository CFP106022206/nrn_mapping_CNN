"""步驟 2 - neuropil 佔位描述子 (僅 FlyCircuit 神經)。

輸入: data/neuron1x1Coding_Ver2.csv -- 每顆神經在 58 個 compartment
      (29 個解剖腦區 x {左, 右}) 的 voxel 數, 外加 `volume` (總量) 與
      `other` (落在所有具名 neuropil 之外的 voxel)。
輸出: results/neuropil_metrics.csv
      results/neuropil_profile_region.npy  各腦區佔 volume 的比例 (n x 29)
      results/neuropil_region_names.csv

每個集中度統計量都算兩種層級:
  * side 層級 (58 格) -- 一顆同時分布在 mb_4_l 與 mb_4_r 的神經算成兩格
  * region 層級 (29 格) -- 同一顆神經算成一個腦區, 因此統計量衡量的是
    「跨腦區投射」而不是「左右對稱」。

分母一律是 volume (= 58 個 neuropil + other, 已驗證 100 % 吻合), 也就是這顆神經
完整的 tracing 點數。只用具名 neuropil 當分母會系統性地灌大 other 較多那一組的
佔比 (other 佔比 D1 0.225 vs D2 0.161), 因此不提供那個版本。前綴 sidetot_ /
regiontot_ 標明分母含 other。

  * 排序類 (top1 / top2 / top3_sum / balance21 / dominance_gap / n_above_*):
    只在具名 compartment 之間排序 (other 不是一個腦區, 不參與排名), 但除以 volume。
  * 分布類 (hhi / entropy / neff_*): 把 other 補成額外一格, 整條分布才加總為 1,
    熵與 HHI 才有定義。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C


def concentration_stats(P: np.ndarray, other_share: np.ndarray, prefix: str,
                        min_share: float) -> dict:
    """集中度 / 分散度描述子。

    P (n x k): 各具名 compartment 佔 volume 的比例, 列和 = 1 - other_share。
    """
    S = np.sort(P, axis=1)[:, ::-1]
    p1, p2, p3 = S[:, 0], S[:, 1], S[:, 2]
    F = np.column_stack([P, other_share])          # 補上 other 一格, 列和 = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(F > 0, F * np.log(F), 0.0), axis=1)
    hhi = (F ** 2).sum(axis=1)
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
    # 分母 = volume (完整 tracing 點數, 含 other)
    Pt = np.divide(V, volume[:, None], out=np.zeros_like(V), where=volume[:, None] > 0)
    Qt = np.divide(R, volume[:, None], out=np.zeros_like(R), where=volume[:, None] > 0)
    other_share = np.divide(other, volume, out=np.zeros_like(other), where=volume > 0)
    assert np.allclose((Pt.sum(axis=1) + other_share)[ok], 1.0), "neuropil 總和 + other != volume"

    out = {"neuron_id": fc["neuron_id"], "group": fc["group"],
           "exclusive": fc["exclusive"], "n_pairs": fc["n_pairs"]}
    out.update(concentration_stats(Pt, other_share, "sidetot", C.MIN_SHARE))
    out.update(concentration_stats(Qt, other_share, "regiontot", C.MIN_SHARE))

    top_region = np.array(regions)[Qt.argmax(axis=1)]
    lr = np.zeros(V.shape[0])
    for i, r in enumerate(top_region):
        li = npil_cols.index(f"{r}_l") if f"{r}_l" in npil_cols else None
        ri = npil_cols.index(f"{r}_r") if f"{r}_r" in npil_cols else None
        l, rr = (V[i, li] if li is not None else 0.0), (V[i, ri] if ri is not None else 0.0)
        lr[i] = abs(l - rr) / (l + rr) if (l + rr) > 0 else np.nan

    out.update({
        "top_region": top_region,
        "second_region": np.array(regions)[np.argsort(Qt, axis=1)[:, -2]],
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
    np.save(C.OUT / "neuropil_profile_region.npy", Qt)
    pd.Series(regions).to_csv(C.OUT / "neuropil_region_names.csv", index=False, header=["region"])
    print(df.groupby("group")[["regiontot_top1", "regiontot_top2", "sidetot_top2",
                               "other_fraction"]].median().round(3))
    return df


if __name__ == "__main__":
    main()
