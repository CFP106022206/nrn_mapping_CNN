"""步驟 11 - 論文用圖。

顏色固定對應 sub dataset, 不挪作他用: D1 (projection) = 藍, D2 (dense) = 橘。
這組配色對色盲 friendly, 且兩組在每張圖上另外用不同 marker 與不同位置區隔,
身分辨識不只靠顏色。圖面文字用英文 (投稿用), 註解用中文。

輸出: results/figures/fig{1..5}_*.{pdf,png}
  fig1 兩組的主要形態差異 (大小軸 + 佔位拓撲軸)
  fig2 兩項納入條件的決策區域
  fig3 專家信心 + 官方 NBLAST 的真假配對分布
  fig4 soma 距離: 人工判定所依據、而 NBLAST 看不到的線索
  fig5 hemibrain 海綿效應: 假配對分數如何被 EM 密度推高
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C

COL = {C.GROUP_PROJ: "#2a78d6", C.GROUP_DENSE: "#eb6834"}
MARK = {C.GROUP_PROJ: "o", C.GROUP_DENSE: "s"}
LABEL = {C.GROUP_PROJ: "Projection-type (D1)", C.GROUP_DENSE: "Dense-type (D2)"}
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#d9d8d4", "#8c8b85"
# fig5 的橫軸: 凸包密度是主指標 (rho +0.634); 改成 "revisit_r16um" 可畫次要證據版
DENSITY = "cable_per_hull_um2"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name: str) -> None:
    # PDF 預設會寫入 CreationDate, 使同樣的圖每次產生的位元組都不同 (git 會誤判為
    # 有改動)。設為 None 讓輸出可重現。
    meta = {"pdf": {"CreationDate": None}, "png": {}}
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG / f"{name}.{ext}", metadata=meta[ext])
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def despine(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def official_scores() -> pd.DataFrame:
    d = pd.read_csv(C.OUT / "nblast_official.csv")
    d["group"] = d["group"].map({"D1_projection": C.GROUP_PROJ,
                                 "D2_dense": C.GROUP_DENSE})
    d["label"] = (d["conf"] >= C.POS_CONF).astype(int)
    d["em_id"] = d["em_id"].astype(str)
    d["fc_id"] = d["fc_id"].astype(str)
    return d


# ------------------------------------------------------------------ fig 1 ---
def fig1_headline(fc: pd.DataFrame, contrast: pd.DataFrame) -> None:
    auc = contrast.set_index("feature")["auc"].to_dict()
    rng = np.random.default_rng(C.RANDOM_STATE)
    panels = [("cable_length_um", "Total cable length", "µm", True),
              ("n_branch_points", "Branch points", "count", True),
              ("sidetot_top2", "Second-compartment share", "fraction of total tracing", False),
              ("arbor_separation", "Two-lobe separation", "centroid dist. / spread", False)]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.9))
    for ax, (f, title, ylab, log) in zip(axes, panels):
        data = [fc.loc[fc.group == g, f].dropna().to_numpy() for g in C.GROUP_ORDER]
        ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False,
                   medianprops=dict(color=INK, lw=1.6),
                   boxprops=dict(color=INK2, lw=0.8),
                   whiskerprops=dict(color=INK2, lw=0.8),
                   capprops=dict(color=INK2, lw=0.8))
        for i, (g, v) in enumerate(zip(C.GROUP_ORDER, data)):
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=6, marker=MARK[g],
                       color=COL[g], alpha=0.55, linewidths=0, zorder=3)
        if log:
            ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Projection\n(D1)", "Dense\n(D2)"], color=INK2)
        ax.set_ylabel(ylab, color=INK)
        a = auc.get(f)
        ax.set_title(title if a is None else f"{title}\nAUC = {a:.2f}", color=INK)
        despine(ax)
    fig.suptitle("What separates the two sub datasets (FlyCircuit neurons)",
                 y=1.06, fontsize=10, color=INK)
    save(fig, "fig1_headline_features")


# ------------------------------------------------------------------ fig 2 ---
def fig2_rule(fc: pd.DataFrame, rules: pd.DataFrame | None) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for g in C.GROUP_ORDER:
        d = fc[fc.group == g]
        ax.scatter(d.cable_length_um, d.sidetot_top2, s=18, marker=MARK[g], color=COL[g],
                   alpha=0.7, linewidths=0.4, edgecolors="white",
                   label=f"{LABEL[g]}  (n={len(d)})", zorder=3)
    ax.set_xscale("log")
    if rules is not None and len(rules):
        r = rules.iloc[0]
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        ax.axvline(r.threshold_1, color=INK2, lw=1.0, ls="--", zorder=2)
        ax.axhline(r.threshold_2, color=INK2, lw=1.0, ls="--", zorder=2)
        ax.add_patch(plt.Rectangle((r.threshold_1, ylo), xhi - r.threshold_1,
                                   r.threshold_2 - ylo, color=COL[C.GROUP_DENSE],
                                   alpha=0.07, zorder=1))
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.text(0.97, 0.04, f"dense-type region\nbalanced acc. {r.balanced_acc_cv_mean:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("Total cable length (µm, log scale)", color=INK)
    ax.set_ylabel("Second-compartment share of total tracing", color=INK)
    ax.set_title("A two-term inclusion criterion", color=INK)
    ax.legend(loc="upper right", labelcolor=INK2, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor=GRID)
    despine(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    save(fig, "fig2_selection_rule")


# ------------------------------------------------------------------ fig 3 ---
def fig3_matching_difficulty() -> None:
    conf = pd.read_csv(C.OUT / "expert_confidence_pairs.csv")
    off = official_scores()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    ax = axes[0]
    bins = np.arange(0.05, 1.15, 0.1)
    for g in C.GROUP_ORDER:
        v = conf[(conf.group == g) & (conf.label > 0)].label
        h, _ = np.histogram(v, bins=bins)
        ax.plot(bins[:-1] + 0.05, h / h.sum(), marker=MARK[g], ms=5, lw=2,
                color=COL[g], label=f"{LABEL[g]}  (n={len(v)})", zorder=3)
    ax.set_xlabel("Annotator confidence in the accepted pair", color=INK)
    ax.set_ylabel("Fraction of accepted pairs", color=INK)
    ax.set_title("Experts are less certain about dense-type pairs", color=INK)
    ax.legend(frameon=False, labelcolor=INK2)
    despine(ax)

    ax = axes[1]
    for i, g in enumerate(C.GROUP_ORDER):
        d = off[off.group == g]
        for k, (lab, hatch) in enumerate([(0, "///"), (1, None)]):
            v = d.loc[d.label == lab, "nblast_official"].to_numpy()
            bp = ax.boxplot([v], positions=[i * 2.4 + k * 0.8], widths=0.6,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.5),
                            boxprops=dict(color=COL[g], lw=1.0),
                            whiskerprops=dict(color=COL[g], lw=0.9),
                            capprops=dict(color=COL[g], lw=0.9))
            bp["boxes"][0].set(facecolor=COL[g], alpha=0.25 if lab == 0 else 0.7,
                               hatch=hatch)
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xticks([0, 0.8, 2.4, 3.2])
    ax.set_xticklabels(["non-pair", "true pair", "non-pair", "true pair"], color=INK2)
    ax.set_xlabel("Projection (D1)                    Dense (D2)", color=INK)
    ax.set_ylabel("NBLAST score (navis, smat.fcwb)", color=INK)
    ax.set_title("Wrong candidates still score positive in D2", color=INK)
    despine(ax)
    save(fig, "fig3_matching_difficulty")


# ------------------------------------------------------------------ fig 4 ---
def fig4_soma_distance() -> None:
    sv = pd.read_csv(C.OUT / "soma_vs_nblast.csv")
    cb = pd.read_csv(C.OUT / "soma_nblast_combined.csv")
    rng = np.random.default_rng(C.RANDOM_STATE)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))

    ax, pos, ticks, labels = axes[0], 0.0, [], []
    for g in C.GROUP_ORDER:
        for lab, alpha in ((1, 0.75), (0, 0.28)):
            v = sv.loc[(sv.group == g) & (sv.label == lab), "soma_dist_um"].to_numpy()
            bp = ax.boxplot([v], positions=[pos], widths=0.55, showfliers=False,
                            patch_artist=True,
                            medianprops=dict(color=INK, lw=1.6),
                            boxprops=dict(color=COL[g], lw=1.0),
                            whiskerprops=dict(color=COL[g], lw=0.9),
                            capprops=dict(color=COL[g], lw=0.9))
            bp["boxes"][0].set(facecolor=COL[g], alpha=alpha)
            ax.scatter(pos + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color=COL[g],
                       alpha=0.3, linewidths=0, zorder=3)
            ticks.append(pos); labels.append("true" if lab else "non")
            pos += 0.75
        pos += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labels, color=INK2)
    ax.set_ylabel("Soma-to-soma distance (µm)", color=INK)
    ax.set_xlabel("Projection (D1)              Dense (D2)", color=INK)
    ax.set_title("The cue annotators use", color=INK)
    despine(ax)

    ax = axes[1]
    order = ["NBLAST 單獨", "soma 距離單獨", "NBLAST + soma"]
    names = ["NBLAST\nalone", "Soma dist.\nalone", "NBLAST\n+ soma"]
    x, w = np.arange(len(order)), 0.36
    for i, g in enumerate(C.GROUP_ORDER):
        sub = cb[cb.group == g].set_index("features")
        v = [float(sub.loc[o, "auc_mean"]) for o in order]
        e = [float(sub.loc[o, "auc_sd"]) for o in order]
        ax.bar(x + (i - 0.5) * w, v, w, yerr=e, capsize=2.5, color=COL[g],
               label=LABEL[g], zorder=3, error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_xticks(x); ax.set_xticklabels(names, color=INK2)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Cross-validated AUC", color=INK)
    ax.set_title("Soma helps only where NBLAST is weak", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right", fontsize=7)
    despine(ax)
    save(fig, "fig4_soma_distance")


# ------------------------------------------------------------------ fig 5 ---
def fig5_em_sponge() -> None:
    """只畫 D2。

    D1 的曲線是平的, 因為 D1 的非配對候選在空間上本來就離得遠 (query 點
    的最近鄰中位 45 µm, D2 只有 9 µm), 海綿效應沒有空間接觸就不會發動, 並排只會多一條沒有資訊的平線。
    """
    d = official_scores()
    m = pd.read_csv(C.OUT / "morphology_metrics.csv")
    m["neuron_id"] = m["neuron_id"].astype(str)
    em = m[m.source == "EM"].drop_duplicates("neuron_id")[["neuron_id", DENSITY]]
    d = d.merge(em, left_on="em_id", right_on="neuron_id")
    d = d[d.group == C.GROUP_DENSE]
    pos, neg = d[d.label == 1], d[d.label == 0]

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    lo, hi = pos.nblast_official.quantile(0.25), pos.nblast_official.quantile(0.75)
    ax.axhspan(lo, hi, color=NEUTRAL, alpha=0.15, zorder=1)
    ax.axhline(lo, color=INK2, lw=0.9, ls="--", zorder=2)
    ax.scatter(pos[DENSITY], pos.nblast_official, s=13, marker="o",
               color=NEUTRAL, alpha=0.55, linewidths=0,
               label=f"true pairs (n={len(pos)})", zorder=3)
    ax.scatter(neg[DENSITY], neg.nblast_official, s=15, marker="s",
               color=COL[C.GROUP_DENSE], alpha=0.7, linewidths=0.3,
               edgecolors="white", label=f"non-pairs (n={len(neg)})", zorder=4)
    b = pd.qcut(neg[DENSITY], 6, labels=False, duplicates="drop")
    med = neg.groupby(b).agg(x=(DENSITY, "median"),
                             y=("nblast_official", "median"))
    ax.plot(med.x, med.y, color=COL[C.GROUP_DENSE], lw=2.2, marker="D", ms=5, zorder=5)
    rho = neg[[DENSITY, "nblast_official"]].corr(method="spearman").iloc[0, 1]
    ax.text(0.03, 0.97, f"non-pairs:  Spearman ρ = {rho:+.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=INK)
    ax.text(0.985, (lo + hi) / 2, "true-pair IQR", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("hemibrain neuron density\n(cable length per convex-hull volume, µm/µm³)",
                  color=INK)
    ax.set_ylabel("NBLAST score", color=INK)
    ax.set_title("Dense hemibrain neurons absorb any query", color=INK)
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right", fontsize=7.5)
    despine(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    save(fig, "fig5_em_sponge")


def main() -> None:
    npil = pd.read_csv(C.OUT / "neuropil_metrics.csv")
    morph = pd.read_csv(C.OUT / "morphology_metrics.csv")
    fc = npil.merge(morph[morph.source == "FC"].drop(columns=["group", "exclusive"]),
                    on="neuron_id", how="inner")
    fc = fc[fc.exclusive]
    contrast = pd.concat([pd.read_csv(C.OUT / "contrast_morphology_FC.csv"),
                          pd.read_csv(C.OUT / "contrast_neuropil_FC.csv")],
                         ignore_index=True)
    rules = None
    f = C.OUT / "selection_rule_thresholds.csv"
    if f.exists():
        r = pd.read_csv(f)
        r = r[r.feature_2 == "sidetot_top2"].sort_values("balanced_acc_cv_mean",
                                                      ascending=False)
        rules = r if len(r) else None

    print("figures:")
    fig1_headline(fc, contrast)
    fig2_rule(fc, rules)
    fig3_matching_difficulty()
    fig4_soma_distance()
    fig5_em_sponge()


if __name__ == "__main__":
    main()
