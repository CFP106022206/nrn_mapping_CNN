#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
result_analysis_make_figure5.py  —  MorphoMatcher 論文 Fig.5 完整拼圖腳本 (獨立可執行)
=============================================================================

一次輸出完整的四面板 Fig.5, 全部在 matplotlib 內完成拼接,
不需要 Keynote, 因此四個子圖的字體/字號/線寬完全一致。

  (a) Score distribution histogram
  (b) Precision / Recall / F1  vs  Threshold
  (c) ROC curve + Confusion matrix inset   <- Style C
  (d) Recall at K  bar chart

-----------------------------------------------------------------------------
使用方式
-----------------------------------------------------------------------------
  # 1) 先用假資料測試排版 (不需要任何資料檔, 馬上可跑)
  python result_analysis_make_figure5.py --demo

  # 2) 接上真實資料 (在下方 CONFIG 區塊設定路徑後)
  python result_analysis_make_figure5.py

  # 3) 產生 NBLAST 版本 (即論文的 Fig.9 / Fig.10)
  python result_analysis_make_figure5.py --mode nblast

  # 4) 產生子資料集 D1 / D2 的版本 (Fig.6 / Fig.7)
  python result_analysis_make_figure5.py --subset ./labeled_info/D1_ID.csv --out Figure6

-----------------------------------------------------------------------------
輸出
-----------------------------------------------------------------------------
  ./Figure/Figure5.pdf   矢量圖, 投稿用 (Type42 字體)
  ./Figure/Figure5.png   300 DPI, 預覽/Word 用

-----------------------------------------------------------------------------
重要: 為什麼字體會一致
-----------------------------------------------------------------------------
  figure 的物理尺寸 = 論文版面中的最終尺寸 (7.0 x 5.0 英吋)
  所有字號以 pt 設定, 輸出後直接置入論文, 全程不縮放
  --> 8pt 就是 8pt
=============================================================================
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG  —  對應你 result_analysis.py 中的設定, 請依實際情況修改
# =============================================================================
CONFIG = {
    'model_name':        'FineTune_miniLR',
    'test_mode':         'cross',      # 'cross' | 'single' | 'nblast'
    'test_set_num':      0,            # 只有 test_mode=='single' 時使用
    'cross_num':         10,           # cross validation fold 數
    'selected_test_set': False,#'./labeled_info/D2+D6_ID.csv',   # False 表示用完整 test set
    'label_csv_prefix':  './result/test_label_{model_name}_D1-D6_',
    'nblast_path':       './labeled_info/nblast_all_list_D2_D5_label.csv',
    'top_k':             5,
    'out_dir':           './Figure',
    'out_name':          'Figure5',
}


# =============================================================================
# 全論文統一樣式  —  這段也可以複製到你其他繪圖腳本
# =============================================================================
PAPER_STYLE = {
    # 字體族 (神經科學期刊標準)
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],

    # 字號 (pt)
    'font.size':        8,
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'legend.fontsize':  6.5,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,

    # 線條
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.5,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'xtick.minor.width':  0.4,
    'ytick.minor.width':  0.4,
    'xtick.minor.size':   1.8,
    'ytick.minor.size':   1.8,

    # 輸出
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':     42,     # TrueType, 期刊投稿必要設定
    'ps.fonttype':      42,
    'mathtext.default': 'regular',
}

# 全文統一配色
COLORS = {
    'label0':     '#001BC2',   # 藍 — non-matched (Label 0)
    'label1':     '#E90132',   # 紅 — matched     (Label 1)
    'precision':  '#1F3B73',   # 深藍
    'recall':     '#008367',   # 綠
    'f1':         '#BB0F1E',   # 暗紅
    'roc':        '#1F77B4',   # 藍
    'threshold':  '#A62C3A',   # 阈值虛線
    'bar':        '#BB0F1E',   # 柱狀圖
    'cell_ok':    '#DDEEDD',   # 混淆矩陣: 正確
    'cell_err':   '#F7DDDD',   # 混淆矩陣: 錯誤
    'cell_head':  '#EFEFEF',   # 混淆矩陣: 表頭
    'grid':       '#CCCCCC',
}

# 版面尺寸 (英吋) — 這是字體一致的關鍵
# 7.0 x 6.2 使每格繪圖區約 2.74 x 2.20 吋 (寬高比 1.25),
# 若用 7.0 x 5.0 則每格為 2.74 x 1.77 (寬高比 1.55), 會有明顯橫向拉伸感
FIG_WIDTH  = 7.0    # 全寬圖, = 178 mm
FIG_HEIGHT = 6.2


# =============================================================================
# 資料載入  —  複製自 result_analysis.py 的邏輯
# =============================================================================
def load_real_data(cfg):
    """回傳 (y_true, y_pred, predict_df, roc_color, tag)"""

    mode = cfg['test_mode']
    label_prefix = cfg['label_csv_prefix'].format(model_name=cfg['model_name'])

    if mode == 'single':
        df = pd.read_csv(f"{label_prefix}{cfg['test_set_num']}.csv")
        y_pred = df['model_pred'].to_numpy()
        y_true = df['label'].to_numpy()
        return y_true, y_pred, df, COLORS['roc'], 'Model'

    elif mode == 'cross':
        parts = []
        for i in range(cfg['cross_num']):
            parts.append(pd.read_csv(f'{label_prefix}{i}.csv'))
        df = pd.concat(parts, ignore_index=True)

        if cfg['selected_test_set']:
            sel = pd.read_csv(cfg['selected_test_set'])[['fc_id', 'em_id']]
            sel = sel.drop_duplicates(subset=['fc_id', 'em_id'])
            df = df.merge(sel, on=['fc_id', 'em_id'], how='inner')

        y_pred = df['model_pred'].to_numpy()
        y_true = df['label'].to_numpy()
        return y_true, y_pred, df, COLORS['roc'], 'Model'

    elif mode == 'nblast':
        df = pd.read_csv(cfg['nblast_path'])
        df = df.drop_duplicates(subset=['fc_id', 'em_id'])

        if cfg['selected_test_set']:
            sel = pd.read_csv(cfg['selected_test_set'])[['fc_id', 'em_id']]
            sel = sel.drop_duplicates(subset=['fc_id', 'em_id'])
            df = df.merge(sel, on=['fc_id', 'em_id'], how='inner')

        y_pred = df['similarity score'].to_numpy()
        y_true = df['label'].to_numpy()
        df = df.rename(columns={'similarity score': 'model_pred'})
        return y_true, y_pred, df, '#D2691E', 'NBLAST'

    raise ValueError(f"unknown test_mode: {mode}")


def make_demo_data(n_pos=575, n_neg=644, seed=0):
    """產生接近論文 Fig.5 的假資料, 供排版測試用"""
    rng = np.random.default_rng(seed)

    # 參數調整為接近論文實際資料的分離程度 (AUC ≈ 0.975)
    pos = rng.beta(2.8, 1.1, n_pos)          # matched     偏高分
    neg = rng.beta(1.0, 4.5, n_neg)          # non-matched 偏低分

    y_pred = np.concatenate([pos, neg])
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # 造出 fc_id / em_id 讓 Recall@K 可以計算
    fc_ids, em_ids = [], []
    for i in range(n_pos):
        fc_ids.append(f'FC{i % 150:04d}')
        em_ids.append(f'EM{i:06d}')
    for i in range(n_neg):
        fc_ids.append(f'FC{i % 150:04d}')
        em_ids.append(f'EM{n_pos + i:06d}')

    df = pd.DataFrame({
        'fc_id': fc_ids,
        'em_id': em_ids,
        'label': y_true,
        'model_pred': y_pred,
    })
    return y_true, y_pred, df, COLORS['roc'], 'Model (demo)'


# =============================================================================
# 指標計算  —  複製自 result_analysis.py 的邏輯
# =============================================================================
def gen_conf_matrix(y_true, y_pred, threshold):
    y_bin = (np.asarray(y_pred) > threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).tolist(), y_bin.tolist(), labels=[1, 0])
    return y_bin, cm


def compute_all_metrics(y_true, y_pred):
    """回傳計算 Fig.5 所需的全部量"""

    # 二值化 soft label
    y_true = np.array([1 if v > 0.5 else 0 for v in y_true])

    # min-max normalize
    lo, hi = np.min(y_pred), np.max(y_pred)
    y_pred = (y_pred - lo) / (hi - lo)

    # ROC
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # 最接近 (0,1) 的操作點
    finite = np.isfinite(thr)
    d = np.sqrt(fpr[finite] ** 2 + (tpr[finite] - 1.0) ** 2)
    bi = int(np.argmin(d))
    best_thr = float(thr[finite][bi])
    best_fpr = float(fpr[finite][bi])
    best_tpr = float(tpr[finite][bi])

    # threshold 掃描曲線
    # 用 1.0001 當上界, 確保網格包含 threshold = 1.0 這個端點,
    # 否則 np.arange(0,1,0.01)[::5] 最後一點只到 0.95, 曲線畫不到右端
    thr_grid = np.arange(0, 1.0001, 0.01)
    prec_lst, rec_lst, f1_lst = [], [], []
    for t in thr_grid:
        y_bin, cm = gen_conf_matrix(y_true, y_pred, t)
        denom_p = cm[0, 0] + cm[1, 0]     # 模型判為 positive 的總數
        denom_r = cm[0, 0] + cm[0, 1]     # 實際 positive 的總數
        # threshold = 1.0 時模型不會判任何 positive, precision 在數學上未定義
        # 依 sklearn precision_recall_curve 的慣例取 1.0
        prec_lst.append(cm[0, 0] / denom_p if denom_p else 1.0)
        rec_lst.append(cm[0, 0] / denom_r if denom_r else 0.0)
        f1_lst.append(f1_score(y_true, y_bin, average=None)[1] if denom_p else 0.0)

    # 最佳操作點的指標
    y_bin_best, cm_best = gen_conf_matrix(y_true, y_pred, best_thr)
    precision = cm_best[0, 0] / (cm_best[0, 0] + cm_best[1, 0])
    recall    = cm_best[0, 0] / (cm_best[0, 0] + cm_best[0, 1])
    f1        = f1_score(y_true, y_bin_best, average=None)[1]

    return dict(
        y_true=y_true, y_pred=y_pred,
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        best_thr=best_thr, best_fpr=best_fpr, best_tpr=best_tpr,
        thr_grid=thr_grid,
        prec_lst=np.array(prec_lst),
        rec_lst=np.array(rec_lst),
        f1_lst=np.array(f1_lst),
        cm=cm_best,
        precision=precision, recall=recall, f1=f1,
    )


def compute_recall_at_k(df, top_k=5):
    """複製自 result_analysis.py 的 ranking analysis"""
    d = df[['fc_id', 'em_id', 'label', 'model_pred']].copy()
    d['bi_label'] = [1 if v > 0.5 else 0 for v in d['label']]

    groups = {}
    for name, g in d.groupby('fc_id'):
        groups[name] = g.sort_values(by='model_pred', ascending=False)

    filt = {k: v for k, v in groups.items()
            if len(v) >= top_k and 1 in v['bi_label'].values}

    if not filt:
        return [], [], 0

    acc = []
    for k in range(top_k, 0, -1):
        correct = sum(1 for key in filt
                      if filt[key].iloc[0:k]['bi_label'].sum() > 0)
        acc.append(correct / len(filt))

    names = [f'Top {k}' for k in range(top_k, 0, -1)]
    return names, acc, len(filt)


# =============================================================================
# 四個子圖的繪製函式
# =============================================================================
def panel_a_distribution(ax, M):
    """(a) Score distribution"""
    y_true, y_pred = M['y_true'], M['y_pred']
    s0 = y_pred[y_true == 0]
    s1 = y_pred[y_true == 1]

    bins = np.linspace(min(s0.min(), s1.min()), max(s0.max(), s1.max()), 40)

    ax.hist(s0, bins=bins, color=COLORS['label0'], alpha=0.65,
            label='Label 0', edgecolor='white', linewidth=0.3)
    ax.hist(s1, bins=bins, color=COLORS['label1'], alpha=0.65,
            label='Label 1', edgecolor='white', linewidth=0.3)

    ax.axvline(M['best_thr'], color=COLORS['threshold'],
               linestyle='--', linewidth=1.0,
               label='Threshold of max F1')

    ax.set_xlabel('Score (Normalized)')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    # 預留上方空間, 確保 legend 不會壓到最高的長條
    n0, _ = np.histogram(s0, bins=bins)
    n1, _ = np.histogram(s1, bins=bins)
    ax.set_ylim(0, max(n0.max(), n1.max()) * 1.30)

    ax.minorticks_on()
    ax.legend(frameon=True, edgecolor=COLORS['grid'],
              framealpha=0.95, loc='upper center', borderpad=0.35,
              handlelength=1.3, handletextpad=0.5)


def panel_b_threshold_curve(ax, M):
    """(b) Precision / Recall / F1 vs Threshold"""
    step = 5  # 每 5 個取一點, 和原腳本一致
    t = M['thr_grid'][::step]
    p = M['prec_lst'][::step]
    r = M['rec_lst'][::step]
    f = M['f1_lst'][::step]

    # 把最佳操作點的數值直接寫進 legend, 避免另開一個文字框造成碰撞
    ax.plot(t, p, '*-', color=COLORS['precision'], alpha=0.85,
            markersize=3.5, label=f"Precision   {M['precision']:.2f}")
    ax.plot(t, r, 'd-', color=COLORS['recall'], alpha=0.85,
            markersize=3.0, label=f"Recall      {M['recall']:.2f}")
    ax.plot(t, f, 'o--', color=COLORS['f1'],
            markersize=3.0, label=f"F1 score    {M['f1']:.2f}")

    # 用箭頭指向 F1 最高點, 不再畫貫穿全圖的垂直虛線
    f1_at_best = float(np.interp(M['best_thr'], M['thr_grid'], M['f1_lst']))
    ax.annotate(f"Threshold = {M['best_thr']:.2f}",
                xy=(M['best_thr'], f1_at_best),
                xytext=(M['best_thr'] + 0.06, 0.28),
                fontsize=6.5,
                arrowprops=dict(arrowstyle='->', color='#555555',
                                linewidth=0.7, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=COLORS['grid'], linewidth=0.5, alpha=0.95),
                zorder=6)

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.minorticks_on()
    leg = ax.legend(loc='lower left', frameon=True, edgecolor=COLORS['grid'],
                    framealpha=0.95, borderpad=0.35,
                    handlelength=1.5, handletextpad=0.5)
    # 等寬字讓數值對齊
    for txt_obj in leg.get_texts():
        txt_obj.set_family('monospace')
        txt_obj.set_fontsize(6.2)


def panel_c_roc_confmat(ax, M, roc_color, human_label='Human'):
    """(c) ROC + confusion matrix inset  —  Style C"""

    ax.plot(M['fpr'], M['tpr'], color=roc_color, linewidth=1.8, zorder=3)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#AAAAAA',
            linewidth=0.8, zorder=1)
    ax.scatter([M['best_fpr']], [M['best_tpr']], s=42, c=COLORS['f1'],
               edgecolors='white', linewidths=1.0, zorder=5)

    ax.annotate(f"Threshold = {M['best_thr']:.2f}",
                xy=(M['best_fpr'], M['best_tpr']),
                xytext=(M['best_fpr'] + 0.14, M['best_tpr'] - 0.22),
                fontsize=6.5,
                arrowprops=dict(arrowstyle='->', color='#555555', linewidth=0.7),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=COLORS['grid'], linewidth=0.5, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # AUC 放在混淆矩陣正上方的空白帶, 避免壓到 ROC 曲線陡升段
    ax.text(0.97, 0.51, f"AUC = {M['roc_auc']:.3f}",
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                      edgecolor=COLORS['grid'], linewidth=0.5, alpha=0.95),
            zorder=6)

    # ---- confusion matrix inset (右下空白處) ----
    cm = np.asarray(M['cm'])
    inset = ax.inset_axes([0.42, 0.05, 0.56, 0.46])
    inset.axis('off')

    table = inset.table(
        cellText=[[f'{cm[0,0]}', f'{cm[0,1]}'],
                  [f'{cm[1,0]}', f'{cm[1,1]}']],
        rowLabels=[f'Same\n({human_label})', f'Diff.\n({human_label})'],
        colLabels=['Same\n(Model)', 'Diff.\n(Model)'],
        cellColours=[[COLORS['cell_ok'],  COLORS['cell_err']],
                     [COLORS['cell_err'], COLORS['cell_ok']]],
        rowColours=[COLORS['cell_head']] * 2,
        colColours=[COLORS['cell_head']] * 2,
        cellLoc='center', loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#999999')
        cell.set_linewidth(0.5)
        if row == 0 or col == -1:
            cell.set_text_props(fontweight='bold', fontsize=5.2)
        else:
            cell.set_text_props(fontsize=8, fontweight='bold')


def panel_d_recall_at_k(ax, names, acc, n_total):
    """(d) Recall at K"""
    if not names:
        ax.text(0.5, 0.5, 'Recall@K unavailable\n(insufficient candidates)',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        return

    ax.bar(names, acc, color=COLORS['bar'], width=0.62, linewidth=0)
    # 上方僅留必要空間: 長條頂端百分比 + 右上角 n = 樣本數
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Recall at K')
    ax.grid(axis='y', linestyle='--', alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    for x, y in enumerate(acc):
        ax.text(x, y + 0.02, f'{y:.1%}', ha='center', va='bottom', fontsize=6.5)

    ax.text(0.99, 0.995, f'n = {n_total}', transform=ax.transAxes,
            fontsize=6, ha='right', va='top', color='#555555')


# =============================================================================
# 主組裝
# =============================================================================
def build_figure(M, df, roc_color, cfg, human_label='Human'):
    plt.rcParams.update(PAPER_STYLE)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        wspace=0.30, hspace=0.42,
        left=0.075, right=0.975, top=0.94, bottom=0.085,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a_distribution(ax_a, M)
    panel_b_threshold_curve(ax_b, M)
    panel_c_roc_confmat(ax_c, M, roc_color, human_label=human_label)

    names, acc, n_total = compute_recall_at_k(df, cfg['top_k'])
    panel_d_recall_at_k(ax_d, names, acc, n_total)

    # 子圖標籤 (a)(b)(c)(d) — 統一位置與字號
    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], ['(a)', '(b)', '(c)', '(d)']):
        ax.text(-0.155, 1.09, lab, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')

    return fig


def main():
    ap = argparse.ArgumentParser(description='Build MorphoMatcher Figure 5')
    ap.add_argument('--demo', action='store_true',
                    help='用假資料測試排版, 不需要任何資料檔')
    ap.add_argument('--mode', default=None,
                    choices=['cross', 'single', 'nblast'],
                    help='覆寫 CONFIG 中的 test_mode')
    ap.add_argument('--subset', default=None,
                    help='指定子資料集 csv (含 fc_id, em_id), 例如 D1 / D2')
    ap.add_argument('--out', default=None, help='輸出檔名 (不含副檔名)')
    ap.add_argument('--outdir', default=None, help='輸出資料夾')
    ap.add_argument('--human-label', default='Human',
                    help="混淆矩陣中人工標註的名稱 (預設 Human, 不建議用 BRC)")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.mode:    cfg['test_mode'] = args.mode
    if args.subset:  cfg['selected_test_set'] = args.subset
    if args.out:     cfg['out_name'] = args.out
    if args.outdir:  cfg['out_dir'] = args.outdir

    # ---- 載入資料 ----
    if args.demo:
        print('[demo] 使用合成資料測試排版')
        y_true, y_pred, df, roc_color, tag = make_demo_data()
    else:
        print(f"[data] test_mode = {cfg['test_mode']}")
        y_true, y_pred, df, roc_color, tag = load_real_data(cfg)
    print(f'[data] {len(y_pred)} pairs loaded ({tag})')

    # ---- 計算 ----
    M = compute_all_metrics(y_true, y_pred)
    print(f"[metric] best threshold = {M['best_thr']:.4f}  "
          f"(FPR={M['best_fpr']:.4f}, TPR={M['best_tpr']:.4f})")
    print(f"[metric] Precision = {M['precision']:.4f}")
    print(f"[metric] Recall    = {M['recall']:.4f}")
    print(f"[metric] F1        = {M['f1']:.4f}")
    print(f"[metric] AUC       = {M['roc_auc']:.4f}")
    print(f"[metric] Confusion matrix (labels=[1,0]):\n{M['cm']}")

    # ---- 繪圖 ----
    fig = build_figure(M, df, roc_color, cfg, human_label=args.human_label)

    os.makedirs(cfg['out_dir'], exist_ok=True)
    stem = os.path.join(cfg['out_dir'], cfg['out_name'])
    fig.savefig(f'{stem}.pdf', format='pdf')
    fig.savefig(f'{stem}.png', format='png')
    plt.close(fig)

    print()
    print(f'✓ 輸出完成')
    print(f'   {stem}.pdf   矢量圖 (Type42 字體), 投稿用')
    print(f'   {stem}.png   300 DPI, 預覽用')
    print(f'   物理尺寸 {FIG_WIDTH}" x {FIG_HEIGHT}" '
          f'= {FIG_WIDTH*25.4:.0f} x {FIG_HEIGHT*25.4:.0f} mm')
    print(f'   ⚠ 置入論文時請勿縮放, 直接以 100% 尺寸插入')


if __name__ == '__main__':
    main()
