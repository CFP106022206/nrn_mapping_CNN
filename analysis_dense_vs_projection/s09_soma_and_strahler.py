"""步驟 13 - soma 位置與 Strahler 加權: NBLAST 在 dense 上失效的一個具體機制。

標註者的觀察: 有些神經對 NBLAST 分數很高但人工判定不是同一顆, 特徵是分支高度
重疊在同一區, 但 soma 位置不同。他們的判準是「神經從 soma 長出來, 所以 soma
的位置與走向不能差太多」。

本步驟檢驗三件事:

(1) 這個判準在你們自己的標註資料上站得住嗎?
    soma 距離 (兩顆神經 SWC 根節點之間的歐氏距離) 對真/假配對的可分離度。

(2) NBLAST 為什麼看不到? NBLAST 是所有點的總和, soma 與初級神經突只佔全部
    cable 的極小一部分, 在總和裡被稀釋。這裡量化稀釋程度:
      frac_primary_neurite = 從 soma 走到第一個分岔點的路徑長 / 總 cable
      frac_cable_top_strahler = 最高 Strahler 階的 cable 佔比
    若 dense 的稀釋更嚴重, 就解釋了為什麼失效集中在 dense。

(3) soma 距離能不能補上 NBLAST 的不足, 而且對 dense 幫助更大?
    以 NBLAST 分數與 soma 距離分別、合併做真/假配對判別, 比較兩組的增益。

NBLAST 分數一律取自 results/nblast_official.csv (由 run_nblast_official.py 以
navis + 官方 smat.fcwb 重算)。專案內原有的兩套分數彼此不一致, 其中腦科中心版
的 (similarity+inverse)/2 與官方實作 Pearson 0.995, 舊版 D2p 家族僅 0.897,
因此以官方重算值為準。

(5) 負面結果, 保留以免誤用: 「骨幹 (高 Strahler 階) 比末梢更不容易被密集的 EM
    吸收, 所以加權骨幹能對抗海綿效應」-- 這個直覺在幾何上不成立, 見
    backbone_vs_terminal()。soma 之所以有用, 是因為它是**位置地標**
    (密集的 EM 團自己的胞體在別處), 不是因為骨幹的最近鄰距離比較可靠。

輸出: results/soma_strahler_metrics.csv
      results/soma_point_weight.csv
      results/strahler_orphan_test.csv
      results/soma_vs_nblast.csv
      results/soma_nblast_combined.csv
      results/nblast_cable_bins.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C
import common as K


def strahler(swc) -> np.ndarray:
    """每個節點的 Strahler 階。葉節點為 1; 內部節點若最大階出現 >=2 次則 +1。"""
    pidx = K._parent_index(swc)
    n = len(swc.nid)
    children = [[] for _ in range(n)]
    for i, p in enumerate(pidx):
        if p >= 0:
            children[p].append(i)
    order, stack = [], [i for i in range(n) if pidx[i] < 0]
    while stack:                                  # 前序走訪
        v = stack.pop(); order.append(v); stack.extend(children[v])
    s = np.ones(n, dtype=np.int32)
    for v in reversed(order):                     # 後序回推
        ch = children[v]
        if not ch:
            continue
        cs = np.array([s[c] for c in ch])
        mx = cs.max()
        s[v] = mx + 1 if (cs == mx).sum() >= 2 else mx
    return s


def neuron_metrics(path: Path) -> dict:
    swc = K.load_swc_fast(path)
    pidx = K._parent_index(swc)
    roots = np.where(pidx < 0)[0]
    root = int(roots[0])
    soma = swc.xyz[root].astype(float)

    s = strahler(swc)
    xyz = swc.xyz.astype(float)
    m = pidx >= 0
    seg_len = np.linalg.norm(xyz[m] - xyz[pidx[m]], axis=1)
    seg_strahler = s[m]                            # 以子節點的階代表該段
    L = float(seg_len.sum())

    # 從 soma 走到第一個分岔點的路徑 (初級神經突 / cell body fibre)
    children = [[] for _ in range(len(swc.nid))]
    for i, p in enumerate(pidx):
        if p >= 0:
            children[p].append(i)
    v, prim = root, 0.0
    while len(children[v]) == 1:
        c = children[v][0]
        prim += float(np.linalg.norm(xyz[c] - xyz[v]))
        v = c
    mid, seg = K.segments(swc)
    cen = np.average(mid, axis=0, weights=seg)

    top = seg_strahler == s.max()
    return {
        "soma_x": soma[0], "soma_y": soma[1], "soma_z": soma[2],
        "cable_length_um": L,
        "max_strahler": int(s.max()),
        "primary_neurite_um": prim,
        "frac_primary_neurite": prim / L if L > 0 else np.nan,
        "frac_cable_top_strahler": float(seg_len[top].sum() / L) if L > 0 else np.nan,
        "frac_cable_strahler1": float(seg_len[seg_strahler == 1].sum() / L) if L > 0 else np.nan,
        "soma_to_centroid_um": float(np.linalg.norm(soma - cen)),
    }


def build() -> pd.DataFrame:
    roster = pd.read_csv(C.OUT / "neuron_roster.csv")
    roster["neuron_id"] = roster["neuron_id"].astype(str)
    rows = []
    for i, r in enumerate(roster.itertuples(), 1):
        try:
            d = neuron_metrics(K.swc_path(r.neuron_id, r.source))
        except Exception as e:
            print(f"  ! {r.source}/{r.neuron_id}: {e}"); continue
        rows.append({"neuron_id": r.neuron_id, "source": r.source, "group": r.group,
                     "exclusive": r.exclusive, **d})
        if i % 200 == 0:
            print(f"  {i}/{len(roster)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "soma_strahler_metrics.csv", index=False)
    return df


def soma_point_weight(df: pd.DataFrame, radii=(10.0, 20.0)) -> pd.DataFrame:
    """soma 在 NBLAST 點雲中實際佔多少權重。

    NBLAST 等權求和, 所以 soma 的影響力就是「soma 鄰域的點數 / 總點數」。
    soma 鄰域的絕對點數幾乎不隨神經大小改變, 因此這個比例基本上是 1/cable。
    """
    from scipy.spatial import cKDTree
    rows = []
    for r in df.itertuples():
        swc = K.load_swc_fast(K.swc_path(r.neuron_id, r.source))
        pts, _ = K.resample_cable(swc, C.RESAMPLE_UM)
        d = np.linalg.norm(pts - np.array([r.soma_x, r.soma_y, r.soma_z]), axis=1)
        row = {"neuron_id": r.neuron_id, "source": r.source, "group": r.group,
               "exclusive": r.exclusive, "n_points": len(pts)}
        for rad in radii:
            row[f"n_points_within_{rad:g}um"] = int((d < rad).sum())
            row[f"frac_points_within_{rad:g}um"] = float((d < rad).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "soma_point_weight.csv", index=False)
    print("\n=== soma 在 NBLAST 點雲中的權重 ===")
    for src in ("FC", "EM"):
        s = out[(out.source == src) & out.exclusive]
        print(f"\n  {src}")
        for f in [c for c in out.columns if c.startswith(("n_points", "frac_points"))]:
            r = K.describe_split(s, f)
            if not r:
                continue
            pj, dn = r["projection_median"], r["dense_median"]
            ratio = pj / dn if dn else float("nan")
            print(f"    {f:28s} D1 {pj:10.4f}  D2 {dn:10.4f}  比值 {ratio:5.2f}x"
                  f"  AUC {r['auc']:.3f}")
    print("  -> soma 鄰域的絕對點數兩組相近, 差異全在分母 (總點數)")
    return out


def dilution(df: pd.DataFrame) -> None:
    print("\n=== (2) soma / 初級神經突在總 cable 裡被稀釋的程度 ===")
    feats = ["frac_primary_neurite", "primary_neurite_um", "frac_cable_top_strahler",
             "frac_cable_strahler1", "max_strahler", "soma_to_centroid_um"]
    for src in ("FC", "EM"):
        sub = df[(df.source == src) & df.exclusive]
        t = pd.DataFrame([r for r in (K.describe_split(sub, f) for f in feats) if r])
        print(f"\n  {src}")
        print(t[["feature", "auc", "auc_signed", "projection_median", "dense_median",
                 "mannwhitney_p"]].round(4).to_string(index=False))


GROUP_MAP = {"D1_projection": C.GROUP_PROJ, "D2_dense": C.GROUP_DENSE}


def load_official_scores() -> pd.DataFrame:
    f = C.OUT / "nblast_official.csv"
    if not f.exists():
        raise SystemExit("缺 results/nblast_official.csv, 請先用裝有 navis 的直譯器執行 "
                         "run_nblast_official.py")
    d = pd.read_csv(f)
    d["fc_id"] = d["fc_id"].astype(str); d["em_id"] = d["em_id"].astype(str)
    d["group"] = d["group"].map(GROUP_MAP)
    d = d.rename(columns={"nblast_official": "score"})
    # conf 是專家信心 (0-1); label 是二值化後的真/假, 與
    # result_analysis_make_figure.py 的 compute_all_metrics() 一致 (>0.5 為真)
    d["label"] = (d["conf"] > 0.5).astype(int)
    return d.dropna(subset=["score"])


def soma_vs_nblast(df: pd.DataFrame) -> pd.DataFrame:
    """(1) 與 (3): soma 距離對真/假配對的判別力, 以及與 NBLAST 的互補性。"""
    pos = df[df.source == "FC"].drop_duplicates("neuron_id").set_index("neuron_id")
    poe = df[df.source == "EM"].drop_duplicates("neuron_id").set_index("neuron_id")
    d = load_official_scores()
    d = d[d.fc_id.isin(pos.index) & d.em_id.isin(poe.index)].copy()
    a = pos.loc[d.fc_id, ["soma_x", "soma_y", "soma_z"]].to_numpy()
    b = poe.loc[d.em_id, ["soma_x", "soma_y", "soma_z"]].to_numpy()
    d["soma_dist_um"] = np.linalg.norm(a - b, axis=1)
    out = d[["fc_id", "em_id", "group", "score", "label", "soma_dist_um"]].reset_index(drop=True)
    out.to_csv(C.OUT / "soma_vs_nblast.csv", index=False)

    print("\n=== (1) soma 距離本身能不能分辨真/假配對? ===")
    for g in C.GROUP_ORDER:
        s = out[out.group == g]
        p = s.loc[s.label == 1, "soma_dist_um"]; n = s.loc[s.label == 0, "soma_dist_um"]
        auc_s = K.auc_from_u(n, p)          # 假配對距離較大 -> AUC>0.5
        auc_n = K.auc_from_u(s.loc[s.label == 1, "score"], s.loc[s.label == 0, "score"])
        print(f"\n  {g}  (真 {len(p)}, 假 {len(n)})")
        print(f"    soma 距離   真配對中位 {p.median():6.1f} um   假配對中位 {n.median():6.1f} um"
              f"   AUC {auc_s:.3f}  p {stats.mannwhitneyu(p, n).pvalue:.2e}")
        print(f"    NBLAST 分數 真配對中位 {s.loc[s.label==1,'score'].median():6.3f}   "
              f"假配對中位 {s.loc[s.label==0,'score'].median():6.3f}   AUC {auc_n:.3f}")

    print("\n=== (3) 標註者描述的失效模式: NBLAST 分數高但其實不是配對 ===")
    for g in C.GROUP_ORDER:
        s = out[out.group == g]
        thr = s.loc[s.label == 1, "score"].quantile(0.25)   # 真配對的低四分位當「高分」門檻
        hi_false = s[(s.label == 0) & (s.score >= thr)]
        lo_false = s[(s.label == 0) & (s.score < thr)]
        tp = s[s.label == 1]
        print(f"\n  {g}  (高分門檻 = 真配對的 25% 分位 = {thr:.3f})")
        print(f"    高分假配對 n={len(hi_false):3d} ({len(hi_false)/max((s.label==0).sum(),1)*100:4.1f}% 的假配對)"
              f"   soma 距離中位 {hi_false.soma_dist_um.median():6.1f} um")
        print(f"    低分假配對 n={len(lo_false):3d}                     soma 距離中位 {lo_false.soma_dist_um.median():6.1f} um")
        print(f"    真配對     n={len(tp):3d}                     soma 距離中位 {tp.soma_dist_um.median():6.1f} um")
        if len(hi_false) >= 5:
            auc = K.auc_from_u(hi_false.soma_dist_um, tp.soma_dist_um)
            print(f"    -> 在這些 NBLAST 分不開的高分配對裡, soma 距離的判別力 AUC {auc:.3f}"
                  f"  (p {stats.mannwhitneyu(hi_false.soma_dist_um, tp.soma_dist_um).pvalue:.2e})")
    return out


def combine(out: pd.DataFrame) -> pd.DataFrame:
    """soma 距離加進 NBLAST 之後的增益, 分組比較。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=C.RANDOM_STATE)
    rows = []
    print("\n=== (3b) soma 距離加進 NBLAST 的增益 (5-fold x20 交叉驗證 AUC) ===")
    for g in C.GROUP_ORDER:
        s = out[out.group == g]
        y = s.label.to_numpy()
        sets = {"NBLAST 單獨": s[["score"]],
                "soma 距離單獨": s[["soma_dist_um"]],
                "NBLAST + soma": s[["score", "soma_dist_um"]]}
        line = {}
        for name, X in sets.items():
            sc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()),
                                 X.to_numpy(float), y, cv=cv, scoring="roc_auc")
            line[name] = (sc.mean(), sc.std())
            rows.append({"group": g, "features": name, "auc_mean": sc.mean(),
                         "auc_sd": sc.std(), "n": len(y)})
        base = line["NBLAST 單獨"][0]; both = line["NBLAST + soma"][0]
        print(f"\n  {g} (n={len(y)})")
        for name, (m, sd) in line.items():
            print(f"    {name:16s} AUC {m:.4f} +- {sd:.4f}")
        print(f"    -> 加入 soma 的增益: {both - base:+.4f}"
              f"  (錯誤率 {1-base:.4f} -> {1-both:.4f}, 降低 {(1-(1-both)/(1-base))*100:.1f}%)")
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "soma_nblast_combined.csv", index=False)
    return df


def backbone_vs_terminal(cap: int = 8000) -> pd.DataFrame:
    """骨幹點是否比末梢點更不容易在密集 EM 中找到近鄰? (答案: 否)

    若成立, 加權骨幹就能直接對抗海綿效應。實測顯示末梢反而是更好的判別依據,
    而且在 EM 密度最高的四分位, 骨幹衰退得比末梢更快。因此 Strahler 加權的
    價值來自 soma 的**位置地標**性質, 不是骨幹的幾何抗性。
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(C.RANDOM_STATE)
    o = load_official_scores()
    o = o[o.group == C.GROUP_DENSE]

    def fc_pts(nid):
        swc = K.load_swc_fast(K.swc_path(nid, "FC"))
        st = strahler(swc)
        pidx = K._parent_index(swc)
        xyz = swc.xyz.astype(float)
        m = pidx >= 0
        a, b = xyz[m], xyz[pidx[m]]
        d = a - b
        L = np.linalg.norm(d, axis=1)
        keep = L > 0
        b, d, L, sb = b[keep], d[keep], L[keep], st[m][keep]
        nseg = np.maximum(np.ceil(L / C.RESAMPLE_UM).astype(int), 1)
        idx = np.repeat(np.arange(nseg.size), nseg)
        starts = np.concatenate([[0], np.cumsum(nseg)[:-1]])
        t = (np.arange(int(nseg.sum())) - np.repeat(starts, nseg) + 0.5) / nseg[idx]
        return b[idx] + t[:, None] * d[idx], sb[idx], int(st.max())

    def em_tree(nid):
        p, _ = K.resample_cable(K.load_swc_fast(K.swc_path(nid, "EM")), C.RESAMPLE_UM)
        return cKDTree(p if len(p) <= cap else p[rng.choice(len(p), cap, replace=False)])

    FCP = {n_: fc_pts(n_) for n_ in o.fc_id.unique()}
    EMT = {n_: em_tree(n_) for n_ in o.em_id.unique()}
    rows = []
    for t in o.itertuples():
        p, st, mx = FCP[t.fc_id]
        d, _ = EMT[t.em_id].query(p, k=1)
        hi = st >= max(mx - 1, 2)          # 骨幹 = 最高兩階
        lo = st == 1                        # 末梢
        if hi.sum() < 10 or lo.sum() < 10:
            continue
        rows.append({"fc_id": t.fc_id, "em_id": t.em_id, "label": t.label,
                     "score": t.score,
                     "nn_backbone": float(np.median(d[hi])),
                     "nn_terminal": float(np.median(d[lo])),
                     "orphan_backbone": float((d[hi] > 10).mean()),
                     "orphan_terminal": float((d[lo] > 10).mean()),
                     "frac_backbone_pts": float(hi.mean())})
    r = pd.DataFrame(rows)
    r.to_csv(C.OUT / "strahler_orphan_test.csv", index=False)
    print("\n=== (5) 骨幹 vs 末梢: 誰更能分辨真假配對? (D2) ===")
    for f in ("nn_backbone", "nn_terminal", "orphan_backbone", "orphan_terminal"):
        a = r.loc[r.label == 0, f].to_numpy()
        b = r.loc[r.label == 1, f].to_numpy()
        print(f"  {f:18s} AUC {K.auc_from_u(a, b):.3f}   真中位 {np.median(b):7.3f}"
              f"  假中位 {np.median(a):7.3f}   p {stats.mannwhitneyu(a, b).pvalue:.1e}")
    m = pd.read_csv(C.OUT / "morphology_metrics.csv")
    m["neuron_id"] = m["neuron_id"].astype(str)
    em = m[m.source == "EM"].drop_duplicates("neuron_id")[["neuron_id", "revisit_r16um"]]
    rr = r.merge(em, left_on="em_id", right_on="neuron_id")
    rr["q"] = pd.qcut(rr.revisit_r16um, 4, labels=False, duplicates="drop")
    print("\n  在 EM 密度最高處骨幹是否仍有效?")
    print(f"  {'EM密度箱':>9} {'n真':>4} {'n假':>4} {'NBLAST':>9} {'骨幹孤兒':>9} {'末梢孤兒':>9}")
    for q, t in rr.groupby("q"):
        a, b = t[t.label == 0], t[t.label == 1]
        if len(a) < 8 or len(b) < 8:
            continue
        print(f"  {int(q)+1:>9} {len(b):>4} {len(a):>4}"
              f" {K.auc_from_u(b.score.to_numpy(), a.score.to_numpy()):9.3f}"
              f" {K.auc_from_u(a.orphan_backbone.to_numpy(), b.orphan_backbone.to_numpy()):9.3f}"
              f" {K.auc_from_u(a.orphan_terminal.to_numpy(), b.orphan_terminal.to_numpy()):9.3f}")
    print("  -> 末梢優於骨幹, 且骨幹在最密的一箱衰退更快: 加權骨幹無法對抗海綿效應")
    return r


def cable_bins(out: pd.DataFrame) -> pd.DataFrame:
    """NBLAST 的判別力隨 FC 神經 cable 長度如何變化 (各組內部分箱)。"""
    m = pd.read_csv(C.OUT / "morphology_metrics.csv")
    m = m[m.source == "FC"].drop_duplicates("neuron_id")
    m["neuron_id"] = m["neuron_id"].astype(str)
    d = out.merge(m[["neuron_id", "cable_length_um"]], left_on="fc_id",
                  right_on="neuron_id")
    rows = []
    print("\n=== (4) NBLAST 判別力 vs cable 長度 (各組內部四等分) ===")
    for g in C.GROUP_ORDER:
        s = d[d.group == g].copy()
        if len(s) < 40:
            continue
        s["bin"] = pd.qcut(s.cable_length_um, 4, labels=False, duplicates="drop")
        print(f"\n  {g}")
        print(f"  {'箱':>3} {'區間(µm)':>16} {'n真':>5} {'n假':>5} {'真中位':>8}"
              f" {'假中位':>8} {'AUC':>7}")
        for b, t in s.groupby("bin"):
            p = t.loc[t.label == 1, "score"].to_numpy()
            q = t.loc[t.label == 0, "score"].to_numpy()
            if len(p) < 5 or len(q) < 5:
                continue
            a = K.auc_from_u(p, q)
            rows.append({"group": g, "bin": int(b) + 1,
                         "cable_lo": t.cable_length_um.min(),
                         "cable_hi": t.cable_length_um.max(),
                         "n_pos": len(p), "n_neg": len(q), "auc": a,
                         "pos_median": np.median(p), "neg_median": np.median(q)})
            print(f"  {int(b)+1:>3} {t.cable_length_um.min():7.0f}-"
                  f"{t.cable_length_um.max():<8.0f} {len(p):>5} {len(q):>5}"
                  f" {np.median(p):8.3f} {np.median(q):8.3f} {a:7.3f}")
    r = pd.DataFrame(rows)
    r.to_csv(C.OUT / "nblast_cable_bins.csv", index=False)
    return r


if __name__ == "__main__":
    d = build()
    soma_point_weight(d)
    dilution(d)
    o = soma_vs_nblast(d)
    combine(o)
    cable_bins(o)
    backbone_vs_terminal()
