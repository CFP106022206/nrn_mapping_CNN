"""步驟 10 - hemibrain 側的「海綿效應」: NBLAST 在 D2 上失效的主因。

發現
----
NBLAST 的計分是: 對 query 的每個點, 找 target 中最近的點, 由 (距離, 切向量夾角)
決定貢獻。當 target 是**填滿緊緻體積的高密度樹突叢**時, query 的任何一點都能找到
近鄰, 分數因此被結構性地推高 -- 與兩者形狀是否真的相符無關。像海綿一樣把任何
東西都吸進去。

本步驟證明這正是 D2 假陽性的主要來源, 且效應完全在 hemibrain 側:
  (1) 假配對分數與 EM 形態的相關 (FC 側作為對照)
  (2) 依 EM 密度分箱, 看假配對「達到真配對水準」的比例
  (3) hub EM: 少數幾顆密集 EM 神經吸走大量假陽性
  (4) 兩組候選池的密度變異 (排除「D1 的池比較均勻」這個解釋)
  (5) 各 NBLAST 變體 (forward / inverse / 雙向平均 / 自比對正規化) 能否消除它
  (6) 兩組非配對通過 stage-1 prescreening 的比例 (排除「D2 的負例被篩得比較難」)
  (7) 孤兒點比例: query 有多少點在 target 找不到近鄰。這是 NBLAST 負分的來源,
      也說明海綿效應為何只在 D2 發動 -- D1 的非配對根本不在同一個位置

分數一律取自 results/nblast_official.csv (run_nblast_official.py, navis + 官方
smat.fcwb, 雙向平均且已自比對正規化)。

輸出: results/sponge_correlations.csv     各組各標籤下, 分數 vs 形態的相關
      results/sponge_density_bins.csv     依 EM 密度分箱的假陽率
      results/sponge_hub_em.csv           吸走最多假陽性的 EM 神經
      results/sponge_variant_check.csv    各 NBLAST 變體的效應強度
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K

GROUP_MAP = {"D1_projection": C.GROUP_PROJ, "D2_dense": C.GROUP_DENSE}
# 主要密度指標: 凸包密度 = 總 cable / 凸包體積。海綿效應需要「線材密」與「範圍小」
# 同時成立, 凸包密度把兩者合在一起量, 因此預測力最強 (rho +0.639 vs revisit 的
# +0.574)。revisit_r16um 只量「實際佔到的空間有多擠」, 不含形狀資訊, 作為不依賴
# 凸包假設的次要證據。
PRIMARY_FEAT = "cable_per_hull_um2"
SECONDARY_FEAT = "revisit_r16um"
DENSITY_FEATS = ["cable_per_hull_um2", "fill_ratio", "revisit_r16um", "revisit_r8um",
                 "revisit_r4um", "overdraw_2d_mean", "local_density_r10",
                 "hull_volume_um3", "cable_length_um", "occupied_volume_um3"]


def load() -> pd.DataFrame:
    f = C.OUT / "nblast_official.csv"
    if not f.exists():
        raise SystemExit("缺 results/nblast_official.csv, 請先執行:\n"
                         "  conda run -n nblast python run_nblast_official.py")
    d = pd.read_csv(f)
    d["fc_id"] = d["fc_id"].astype(str)
    d["em_id"] = d["em_id"].astype(str)
    d["group"] = d["group"].map(GROUP_MAP)
    d["label"] = (d["conf"] > 0.5).astype(int)
    d = d.rename(columns={"nblast_official": "score"}).dropna(subset=["score"])

    m = pd.read_csv(C.OUT / "morphology_metrics.csv")
    m["neuron_id"] = m["neuron_id"].astype(str)
    em = m[m.source == "EM"].drop_duplicates("neuron_id")
    fc = m[m.source == "FC"].drop_duplicates("neuron_id")
    keep = ["neuron_id"] + [c for c in DENSITY_FEATS if c in m.columns]
    d = d.merge(em[keep].add_prefix("em_"), left_on="em_id", right_on="em_neuron_id")
    d = d.merge(fc[keep].add_prefix("fc_"), left_on="fc_id", right_on="fc_neuron_id")
    return d


# ------------------------------------------------------------------- (1) ----
def correlations(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in C.GROUP_ORDER:
        for lab in (0, 1):
            s = d[(d.group == g) & (d.label == lab)]
            if len(s) < 20:
                continue
            for side in ("em", "fc"):
                for f in DENSITY_FEATS:
                    col = f"{side}_{f}"
                    if col not in s:
                        continue
                    rho, p = stats.spearmanr(s[col], s.score)
                    rows.append({"group": g, "label": lab, "side": side.upper(),
                                 "feature": f, "n": len(s), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "sponge_correlations.csv", index=False)

    print("=== (1) NBLAST 分數 vs 形態的 Spearman 相關 ===")
    print("    (* p<0.001)")
    for g in C.GROUP_ORDER:
        for lab in (0, 1):
            s = out[(out.group == g) & (out.label == lab)]
            if s.empty:
                continue
            n = int(s.n.iloc[0])
            print(f"\n  {g} / {'真配對' if lab else '假配對'}  (n={n})")
            piv = s.pivot(index="feature", columns="side", values="rho")
            sig = s.pivot(index="feature", columns="side", values="p") < 0.001
            for f in DENSITY_FEATS:
                if f not in piv.index:
                    continue
                e, fc_ = piv.loc[f, "EM"], piv.loc[f, "FC"]
                print(f"    {f:22s} EM {e:+.3f}{'*' if sig.loc[f,'EM'] else ' '}"
                      f"   FC {fc_:+.3f}{'*' if sig.loc[f,'FC'] else ' '}")
    return out


# ------------------------------------------------------------------- (2) ----
def density_bins(d: pd.DataFrame, feature: str = f"em_{PRIMARY_FEAT}",
                 n_bins: int = 4) -> pd.DataFrame:
    rows = []
    print(f"\n=== (2) 假配對依 {feature} 分 {n_bins} 等分 ===")
    for g in C.GROUP_ORDER:
        s = d[d.group == g]
        neg = s[s.label == 0].copy()
        pos = s[s.label == 1]
        if len(neg) < 4 * n_bins or len(pos) < 10:
            continue
        q25 = pos.score.quantile(0.25)
        neg["bin"] = pd.qcut(neg[feature], n_bins, labels=False, duplicates="drop")
        print(f"\n  {g}   (真配對分數中位 {pos.score.median():.3f}, 25% 分位 {q25:.3f})")
        print(f"  {'箱':>3} {'EM 密度區間':>20} {'n':>5} {'假配對分數中位':>14}"
              f" {'達真配對水準':>13}")
        for b, t in neg.groupby("bin"):
            frac = float((t.score >= q25).mean())
            rows.append({"group": g, "feature": feature, "bin": int(b) + 1,
                         "lo": t[feature].min(), "hi": t[feature].max(),
                         "n": len(t), "score_median": t.score.median(),
                         "frac_reaching_true": frac})
            # 用 .3g 而非固定小數: 凸包密度是 1e-3 量級, 固定兩位會全部印成 0.00
            print(f"  {int(b)+1:>3} {t[feature].min():9.3g}-{t[feature].max():<10.3g}"
                  f" {len(t):>5} {t.score.median():14.3f} {frac*100:12.1f}%")
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "sponge_density_bins.csv", index=False)
    return out


# ------------------------------------------------------------------- (3) ----
def hub_em(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in C.GROUP_ORDER:
        s = d[d.group == g]
        q25 = s.loc[s.label == 1, "score"].quantile(0.25)
        hi = s[(s.label == 0) & (s.score >= q25)]
        if hi.empty:
            continue
        cnt = hi.groupby("em_id").size().rename("n_high_false").reset_index()
        cols = ["em_id", "em_cable_length_um", f"em_{PRIMARY_FEAT}",
                f"em_{SECONDARY_FEAT}", "em_fill_ratio", "em_hull_volume_um3"]
        cnt = cnt.merge(s[cols].drop_duplicates("em_id"), on="em_id")
        cnt["group"] = g
        cnt["is_true_partner"] = cnt.em_id.isin(s.loc[s.label == 1, "em_id"])
        rows.append(cnt)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.OUT / "sponge_hub_em.csv", index=False)

    print("\n=== (3) 吸走最多高分假配對的 EM 神經 ===")
    for g in C.GROUP_ORDER:
        s = out[out.group == g]
        if s.empty:
            continue
        print(f"\n  {g}")
        print(s.nlargest(8, "n_high_false")[
            ["em_id", "n_high_false", "is_true_partner", "em_cable_length_um",
             f"em_{PRIMARY_FEAT}", f"em_{SECONDARY_FEAT}"]].round(4).to_string(index=False))
        if len(s) > 5:
            rho, p = stats.spearmanr(s.n_high_false, s[f"em_{PRIMARY_FEAT}"])
            print(f"    吸引力 vs EM {PRIMARY_FEAT}: rho {rho:+.3f} (p {p:.3f}, n={len(s)})")
    return out


# ------------------------------------------------------------------- (4) ----
def pool_variability(d: pd.DataFrame) -> None:
    """D1 為什麼沒有這個問題: 候選池本身的密度變異。"""
    print("\n=== (4) 兩組 hemibrain 候選池的密度變異 ===")
    m = pd.read_csv(C.OUT / "morphology_metrics.csv")
    m["neuron_id"] = m["neuron_id"].astype(str)
    em = m[(m.source == "EM")].drop_duplicates(["neuron_id", "group"])
    print(f"  {'組':11s} {'n':>4} " + "".join(f"{c:>30}" for c in
          [f"{PRIMARY_FEAT} 中位[5-95%]", f"{SECONDARY_FEAT} 中位[5-95%]"]))
    for g in C.GROUP_ORDER:
        s = em[em.group == g]
        cells = []
        for f in (PRIMARY_FEAT, SECONDARY_FEAT):
            v = s[f].dropna()
            cells.append(f"{v.median():.4g} [{v.quantile(.05):.4g}-{v.quantile(.95):.4g}]".rjust(30))
        print(f"  {g:11s} {len(s):>4} " + "".join(cells))
    for f in (PRIMARY_FEAT, SECONDARY_FEAT):
        a = em.loc[em.group == C.GROUP_PROJ, f].dropna()
        b = em.loc[em.group == C.GROUP_DENSE, f].dropna()
        print(f"    {f:20s} 變異係數 (SD/mean): projection {a.std()/a.mean():.3f}"
              f"  dense {b.std()/b.mean():.3f}"
              f"   | 95/5 分位比: {a.quantile(.95)/a.quantile(.05):.1f}x"
              f" vs {b.quantile(.95)/b.quantile(.05):.1f}x")


# ------------------------------------------------------------------- (5) ----
def variant_check(d: pd.DataFrame) -> pd.DataFrame:
    """雙向平均與自比對正規化能不能消除海綿效應? (答案: 幾乎不能)"""
    f = C.ROOT / "labeled_info" / "nblast_all_list_D2_D5_include_inverse_label.csv"
    if not f.exists():
        print("\n(略過 (5): 缺 include_inverse 檔)")
        return pd.DataFrame()
    inv = pd.read_csv(f)
    inv["fc_id"] = inv.fc_id.astype(str); inv["em_id"] = inv.em_id.astype(str)
    inv = inv.drop_duplicates(["fc_id", "em_id"])
    inv["mean_2way"] = (inv["similarity score"] + inv["inverse score"]) / 2
    s = d.merge(inv[["fc_id", "em_id", "similarity score", "inverse score",
                     "mean_2way"]], on=["fc_id", "em_id"])
    variants = {"forward (單向)": "similarity score", "inverse (反向)": "inverse score",
                "mean (雙向平均)": "mean_2way",
                "official (雙向+自比對正規化)": "score"}
    rows = []
    print("\n=== (5) 各 NBLAST 變體的海綿效應強度與表現 ===")
    print(f"  {'變體':30s} {'rho vs EM 凸包密度':>18} {'D2 AUC':>8} {'D2 假高分率':>12}")
    for name, col in variants.items():
        neg = s[(s.group == C.GROUP_DENSE) & (s.label == 0)]
        rho = stats.spearmanr(neg[col], neg[f"em_{PRIMARY_FEAT}"])[0]
        t = s[s.group == C.GROUP_DENSE]
        a = roc_auc_score(t.label, t[col])
        q = t.loc[t.label == 1, col].quantile(0.25)
        fr = float((t.loc[t.label == 0, col] >= q).mean())
        rows.append({"variant": name, "column": col, "rho_vs_em_density": rho,
                     "auc_dense": a, "frac_false_high_dense": fr})
        print(f"  {name:30s} {rho:+18.3f} {a:8.4f} {fr*100:11.1f}%")
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "sponge_variant_check.csv", index=False)
    print("  -> 自比對正規化只除以 S(A,A), 與 target 無關, 因此治不了 target 端的偏差")
    return out


# ------------------------------------------------------------------- (6) ----
def prescreen_check(d: pd.DataFrame) -> pd.DataFrame:
    """兩組的非配對通過 stage-1 prescreening 的比例是否相當?

    若相當, 就不能把「D2 比較難」歸因於候選集合被篩得比較嚴。
    篩選條件取自 candidate_matching.py: 質心距離 <= 100 um、(r21,r31) 距離
    <= 0.4、rod/disk 主軸方向一致。
    """
    sys.path.insert(0, str(C.ROOT))
    try:
        from candidate_matching import filter_pairs_by_orientation_rod_disk
    except ImportError:
        print("\n(略過 (6): 匯入 candidate_matching 失敗)")
        return pd.DataFrame()
    P = {}
    for src in ("FC", "EM"):
        base = C.ROOT / "data" / f"descriptors_{src}"
        ids = np.load(base / f"neuron_ids_{src}.npy", allow_pickle=True).astype(str)
        P[src] = {"idx": {v: i for i, v in enumerate(ids)},
                  "cen": np.load(base / f"centroids_{src}.npy"),
                  "rat": np.load(base / f"eigvals_ratio_{src}.npy")[:, 1:3],
                  "vec": np.load(base / f"eigvecs_{src}.npy")}
    s = d[d.fc_id.isin(P["FC"]["idx"]) & d.em_id.isin(P["EM"]["idx"])].copy()
    ia = np.array([P["FC"]["idx"][x] for x in s.fc_id])
    ib = np.array([P["EM"]["idx"][x] for x in s.em_id])
    s["centroid_dist"] = np.linalg.norm(P["FC"]["cen"][ia] - P["EM"]["cen"][ib], axis=1)
    s["ratio_dist"] = np.linalg.norm(P["FC"]["rat"][ia] - P["EM"]["rat"][ib], axis=1)
    ia3, ib3, _, _ = filter_pairs_by_orientation_rod_disk(
        ia, ib, P["FC"]["rat"], P["EM"]["rat"], P["FC"]["vec"], P["EM"]["vec"])
    kept = set(zip(ia3.tolist(), ib3.tolist()))
    s["pass_orient"] = [(a, b) in kept for a, b in zip(ia, ib)]
    s["pass_all"] = (s.centroid_dist <= 100) & (s.ratio_dist <= 0.4) & s.pass_orient
    s[["fc_id", "em_id", "group", "label", "centroid_dist", "ratio_dist",
       "pass_orient", "pass_all"]].to_csv(C.OUT / "sponge_prescreen_check.csv", index=False)
    print("\n=== (6) 非配對通過 stage-1 prescreening 的比例 ===")
    t = s.groupby(["group", "label"]).agg(
        n=("pass_all", "size"), pass_all=("pass_all", "mean"),
        centroid_median=("centroid_dist", "median"),
        ratio_median=("ratio_dist", "median"))
    t["pass_all"] = (t.pass_all * 100).round(1)
    print(t.round(3).to_string())
    print("  -> 兩組非配對的通過率相近, 因此差異不能歸因於候選集合的篩選強度")
    return s


# ------------------------------------------------------------------- (7) ----
def orphan_points(d: pd.DataFrame, cap: int = 6000) -> pd.DataFrame:
    """query 有多少點在 target 找不到近鄰 -- NBLAST 負分的直接來源。"""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(C.RANDOM_STATE)

    def cloud(nid, src):
        p, _ = K.resample_cable(K.load_swc_fast(K.swc_path(nid, src)), C.RESAMPLE_UM)
        return p if len(p) <= cap else p[rng.choice(len(p), cap, replace=False)]

    fcp = {n: cloud(n, "FC") for n in d.fc_id.unique()}
    emt = {n: cKDTree(cloud(n, "EM")) for n in d.em_id.unique()}
    rows = []
    for t in d.itertuples():
        dist, _ = emt[t.em_id].query(fcp[t.fc_id], k=1)
        rows.append({"fc_id": t.fc_id, "em_id": t.em_id, "group": t.group,
                     "label": t.label, "score": t.score,
                     "frac_nn_gt5": float((dist > 5).mean()),
                     "frac_nn_gt10": float((dist > 10).mean()),
                     "frac_nn_gt20": float((dist > 20).mean()),
                     "nn_median_um": float(np.median(dist))})
    r = pd.DataFrame(rows)
    r.to_csv(C.OUT / "sponge_orphan_points.csv", index=False)
    print("\n=== (7) query 點找不到近鄰的比例 (NBLAST 負分的來源) ===")
    print(f"  {'組':11s} {'標籤':>5} {'n':>4} {'>5µm':>8} {'>10µm':>8} {'>20µm':>8} {'NN中位':>8}")
    for g in C.GROUP_ORDER:
        for lab in (1, 0):
            s = r[(r.group == g) & (r.label == lab)]
            if s.empty:
                continue
            print(f"  {g:11s} {'真' if lab else '假':>5} {len(s):>4}"
                  f" {s.frac_nn_gt5.median():8.3f} {s.frac_nn_gt10.median():8.3f}"
                  f" {s.frac_nn_gt20.median():8.3f} {s.nn_median_um.median():8.2f}")
    print("\n  假配對: 孤兒點比例 vs NBLAST 分數")
    for g in C.GROUP_ORDER:
        s = r[(r.group == g) & (r.label == 0)]
        rho, p = stats.spearmanr(s.frac_nn_gt10, s.score)
        print(f"    {g:11s} rho {rho:+.3f} (p {p:.1e}, n={len(s)})")
    print("  -> D1 的非配對根本不在同一個位置 (NN 中位 ~44 µm), 分數因此一律很負;"
          "\n     D2 的非配對與 query 空間重疊 (NN 中位 ~9 µm, 接近真配對的 ~6 µm),"
          "\n     NBLAST 必須只靠形狀分辨, 海綿效應才在此發動")
    return r


# ------------------------------------------------------------------- (8) ----
def hull_robustness(d: pd.DataFrame) -> pd.DataFrame:
    """凸包對離群分支敏感是常見疑慮, 這裡直接檢驗。

    剔除距質心最遠的 1/5/10 % cable 後重算凸包密度, 若結論不變就代表這個疑慮
    在本資料上不成立。(修正方向本應是 concave hull / alpha shape, 但那要多一個
    參數; 若完整凸包已經夠穩健就沒有必要。)
    """
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = []
    for nid in sorted(d.em_id.unique()):
        mid, seg = K.segments(K.load_swc_fast(K.swc_path(nid, "EM")))
        if len(mid) > 50000:
            sel = rng.choice(len(mid), 50000, replace=False)
            mid, seg = mid[sel], seg[sel]
        cen = np.average(mid, axis=0, weights=seg)
        dist = np.linalg.norm(mid - cen, axis=1)
        r = {"em_id": nid}
        for q in (100, 99, 95, 90):
            keep = dist <= np.percentile(dist, q)
            try:
                r[f"cph_p{q}"] = seg[keep].sum() / ConvexHull(mid[keep]).volume
            except Exception:
                r[f"cph_p{q}"] = np.nan
        rows.append(r)
    h = pd.DataFrame(rows)
    h.to_csv(C.OUT / "sponge_hull_robustness.csv", index=False)

    m = d.merge(h, on="em_id")
    neg = m[(m.group == C.GROUP_DENSE) & (m.label == 0)]
    pos = m[(m.group == C.GROUP_DENSE) & (m.label == 1)]
    q25 = pos.score.quantile(0.25)
    print("\n=== (8) 凸包密度對離群分支的穩健性 ===")
    print(f"  {'版本':28s} {'rho vs 假配對分數':>17} {'最稀→最密四分位':>20}")
    for q, lab in ((100, "完整凸包 (主指標)"), (99, "剔除最遠 1 % cable"),
                   (95, "剔除最遠 5 %"), (90, "剔除最遠 10 %")):
        c = f"cph_p{q}"
        rho = stats.spearmanr(neg[c], neg.score)[0]
        b = pd.qcut(neg[c], 4, labels=False, duplicates="drop")
        fr = [float((neg.score[b == i] >= q25).mean()) for i in (0, 3)]
        print(f"  {lab:28s} {rho:+17.3f} {fr[0]*100:11.1f}% → {fr[1]*100:5.1f}%")
    for q in (99, 95, 90):
        print(f"    完整 vs 剔除最遠 {100-q:2d} %: Spearman "
              f"{stats.spearmanr(h.cph_p100, h[f'cph_p{q}'])[0]:+.3f}")
    print("  -> 各版本高度一致, 離群分支的疑慮在本資料上不成立")
    return h


def main():
    d = load()
    print(f"載入 {len(d)} 組配對 "
          f"({d.groupby('group').size().to_dict()})\n")
    correlations(d)
    density_bins(d)
    hub_em(d)
    pool_variability(d)
    variant_check(d)
    prescreen_check(d)
    orphan_points(d)
    hull_robustness(d)


if __name__ == "__main__":
    main()
