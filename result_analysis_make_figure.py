#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
result_analysis_make_figure.py  —  MorphoMatcher 論文 Fig.5 完整拼圖腳本 (獨立可執行)
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
  python result_analysis_make_figure.py --demo

  # 2) 接上真實資料 Fig.5 (在下方 CONFIG 區塊設定路徑後)
  python result_analysis_make_figure.py --out fig5_result

  # 3) 產生子資料集 D1 / D2 的版本 (Fig.6 / Fig.7)
    # 步驟 1: 產生三張拼接好的疊合圖
    python render_pairs.py --composite

    # 步驟 2: 組成 Fig 6
    python result_analysis_make_figure.py --layout subset \
        --subset ./labeled_info/D5_conf.csv \
        --overlay output_image/G0239-F-000012_5813068729_composite.png \
                output_image/Gad1-F-402516_862705904_composite.png \
                output_image/TH-F-200081_331662710_composite.png \
        --out fig6_Subset1Result
        
        可手動覆蓋顯示的ID為指定ID(不指定默認由overlay路徑解析)
        # --neuron-fc fru-F-500297 fru-F-500435 TH-F-000101 \
        # --neuron-em 1051630846  1142011140   331662710 \

    Fig 7
    python result_analysis_make_figure.py --layout subset \
        --subset ./labeled_info/D2+D6_ID.csv \
        --overlay output_image/TH-F-000019_859265651_composite.png \
                output_image/VGlut-F-200251_5813020996_composite.png \
                output_image/Trh-F-000008_1671638278_composite.png \
        --out fig7_Subset2Result
        
  # 3) 產生 NBLAST 版本 (即論文的 Fig.9 / Fig.10)
    Fig 9
    python result_analysis_make_figure.py --mode nblast --layout subset \
        --subset ./labeled_info/D2+D6_ID.csv \
        --overlay output_image/Trh-M-200043_1671620613_composite.png \
                output_image/VGlut-F-400049_297251714_composite.png \
                output_image/VGlut-F-700541_1702306037_composite.png \
        --out fig9_Nblast_D2

  

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

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.text as mtext
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
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
    # 不用 bbox='tight': 它會裁掉白邊, 使輸出的 PDF 比 figsize 窄
    # (實測 cube 版 6.5" -> 4.88")。之後用 width=\textwidth 置入時
    # 會被「放大」, 字級同樣跑掉。設為 None 才能保證輸出 = figsize,
    # 搭配 width=\textwidth 為 1:1。留白由 gridspec 的 left/right 控制。
    'savefig.bbox':     None,
    'savefig.pad_inches': 0.0,
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
    'cell_ok':    '#BFEEE5',   # 混淆矩陣: 正確 (較原本更淡)
    'cell_err':   '#F4D5DB',   # 混淆矩陣: 錯誤 (較原本更淡)
    'cell_head':  '#F2F2F2',   # 混淆矩陣: 表頭
    'grid':       '#CCCCCC',
}

# 版面尺寸 (英吋) — 這是字體一致的關鍵
# 7.0 x 6.2 使每格繪圖區約 2.74 x 2.20 吋 (寬高比 1.25),
# 若用 7.0 x 5.0 則每格為 2.74 x 1.77 (寬高比 1.55), 會有明顯橫向拉伸感
# 混淆矩陣表格外觀
# TABLE_ALPHA=1.0 (不透明) 是建議值: 表格正好壓在 ROC 的對角參考線上,
# 半透明會讓虛線從數字中間穿過, 反而更難讀。要讓表格「輕」一點,
# 降低色塊飽和度 (見 cell_ok / cell_err) 比降透明度有效。
TABLE_ALPHA = 1.0
# 表頭 (兩個欄標題 + 兩個列標題, 共四格) 可以比數字格輕一點: 表頭沒有
# 要讀的數值, 底下透出一小段對角參考線不影響判讀, 灰色塊也不會那麼搶戲。
# 只改 facecolor 的 alpha 而不用 cell.set_alpha(): 後者連框線一起變淡,
# 表頭與數字格的格線就會粗細不一。
TABLE_HEAD_ALPHA = 0.8
TABLE_EDGE  = '#B0B0B0'

# 圖例 / 標註框的底色透明度
# LEGEND_ALPHA 給圖例: (a) 的圖例正好蓋在最高的幾根長條上, 0.95 幾乎不透明,
#   等於把資料挖掉一塊。0.70 可以看見底下的長條輪廓, 但文字仍讀得清楚
#   (白底 0.70 混上飽和色塊後對比度仍 > 4.5:1)。再低會開始糊。
# LABEL_ALPHA 給指向性標註框 (Threshold / AUC): 這些框直接壓在曲線上,
#   太透明會讓線從字中間穿過, 所以維持比圖例高一點。
LEGEND_ALPHA = 0.70
LABEL_ALPHA  = 0.80

# (c) 混淆矩陣 inset 在 axes 座標的位置 [x0, y0, w, h]
# 上緣 (y0+h = 0.51) 同時也是 Threshold 標註可以往下走的底線。
INSET_RECT = (0.42, 0.05, 0.56, 0.46)

FIG_WIDTH  = 6.5    # = 165 mm, 對齊單欄 LaTeX \textwidth
# 原本 5.8" 讓每格是 2.54 x 2.05 (寬高比 1.24) —— 比 subset 版的
# 1.62 x 1.28 大得多, 但內容量相近, 所以顯得空。
# 改為 4.6" 後每格 2.54 x 1.63 (比 1.57), 資訊密度接近 subset 版。
# 不改成 1x4 橫排是因為每格只剩 1.17" 寬, (c) 的混淆矩陣塞不下。
FIG_HEIGHT = 4.6

# --layout row: 1x4 橫排, 最省版面高度
FIG_WIDTH_ROW  = 6.5
FIG_HEIGHT_ROW = 2.05

# --layout subset (Fig.6/7/9/10): 上排 (a)(b)(c), 下排 (d1)(d2)(d3) 疊圖
FIG_WIDTH_SUBSET  = 6.5
# 上排每格寬固定 1.62" (由 FIG_WIDTH 與欄距決定)。
# 原本 4.2" 讓每格變成 1.62 x 1.81, 寬高比 0.89 —— 比正方形還高。
# 改為 3.3" 後是 1.62 x 1.36, 寬高比 1.19, 略扁。
FIG_HEIGHT_SUBSET = 3.3

# 疊圖的神經元配色 — 必須與 render_pairs.py 的 Mayavi 設定一致,
# 否則圖例圓點與圖上神經元顏色對不起來。
#   visualize_neuron(..., color=(0.8, 0.2, 0.2)) -> #CC3333
#   visualize_neuron(..., color=(0.2, 0.2, 0.8)) -> #3333CC
NEURON_FC = '#CC3333'
NEURON_EM = '#3333CC'


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
def _extent_in_axes(artist, ax, renderer):
    """回傳 artist 的外框 (含 bbox patch) 在 axes 座標系的 Bbox

    注意: Annotation.get_window_extent() 會把箭頭一起算進去, 量出來的框
    從文字一路延伸到被標註的資料點, 比實際的白底方框大得多 (實測高度多
    一倍), 拿它做碰撞判斷會過度保守。這裡強制走 Text 的版本, 只量文字。
    """
    bb = mtext.Text.get_window_extent(artist, renderer=renderer)
    patch = getattr(artist, 'get_bbox_patch', lambda: None)()
    if patch is not None:
        # bbox patch 的 padding 是以 points 計, get_window_extent 不含它
        pad_px = patch.get_boxstyle().pad * artist.get_fontsize() \
                 * ax.figure.dpi / 72.0
        bb = bb.expanded(1.0, 1.0).padded(pad_px)
    return bb.transformed(ax.transAxes.inverted())


def _avoid_overlap(moving, fixed, ax, x_lo=0.03, x_hi=0.98, y_floor=0.0,
                   pad=0.02, min_fontsize=5.0, fixed_min=5.2,
                   curve=None, curve_min_fontsize=5.6):
    """把 moving 這個標註排在不與 fixed / 邊界 / 曲線相撞的位置。

    版面一改 (full 每格 2.5" vs subset 1.62" vs row 1.28"), 同一段文字
    佔的寬度比例就差到 2 倍, 位置和字級都不能寫死 —— 這裡量實際的算繪
    寬度再決定, 優先順序是:
      1. 維持置中掛在資料點正下方 (和 (b) 一樣, 一眼看得出在標什麼)
      2. 置中放不下 -> 先縮標註字級 (到 min_fontsize)
      3. 還是放不下 -> 縮 fixed (AUC) 的字級 (到 fixed_min)
      4. 都到底了 -> 才把標註往左推, 推到剛好不撞為止

    curve=(x, y): ROC 曲線 (資料座標)。標註框要留在曲線下方的空白楔形裡,
    否則白底會蓋掉曲線。tpr 單調遞增, 所以只要框的左上角在曲線下方即可,
    也就是框的左緣不得越過「曲線高度 = 框頂」的那個 x。這個限制在字被逼到
    curve_min_fontsize 以下時放棄 —— 字小到讀不出來, 比壓到一小段曲線糟。

    座標一律用 axes fraction 算, 最後才換回資料座標寫回 set_position():
    (c) 的 ylim 是 (0, 1.02) 而不是 (0, 1), 直接混用會有 2% 的偏移,
    正好夠讓「避開曲線」算到錯的高度上。
    """
    fig = ax.figure
    fig.canvas.draw()                       # 先算繪一次才有 renderer
    renderer = fig.canvas.get_renderer()

    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    to_data = lambda fx, fy: (xa + fx * (xb - xa), ya + fy * (yb - ya))
    to_frac = lambda dx, dy: ((dx - xa) / (xb - xa), (dy - ya) / (yb - ya))

    x_want = to_frac(*moving.get_position())[0]   # 理想值: 掛在資料點正下方

    def measure():
        return (_extent_in_axes(moving, ax, renderer),
                _extent_in_axes(fixed,  ax, renderer))

    def move_to(fx, fy):
        moving.set_position(to_data(fx, fy))
        fig.canvas.draw()

    for _ in range(10):
        mb, fb = measure()

        # 先把高度喬好 (別掉進混淆矩陣), 再算水平位置 ——
        # 反過來做的話, 避開曲線的計算會用到搬動前的框頂高度。
        if mb.y0 < y_floor + pad:
            fx, fy = to_frac(*moving.get_position())
            move_to(fx, fy + (y_floor + pad - mb.y0))
            mb, fb = measure()

        fs = moving.get_fontsize()
        x0_min = x_lo
        if curve is not None and fs > curve_min_fontsize:
            cx, cy = np.asarray(curve[0]), np.asarray(curve[1])
            top_y = to_data(0.0, mb.y1)[1]
            x_curve = to_frac(float(np.interp(top_y, cy, cx)), 0.0)[0]
            x0_min = max(x_lo, x_curve + pad)

        # 只有垂直方向重疊時, 水平位置才需要為 fixed 讓路
        v_overlap = (mb.y0 < fb.y1 + pad) and (mb.y1 > fb.y0 - pad)
        right_limit = (fb.x0 - pad) if v_overlap else x_hi

        half = mb.width / 2.0
        lo, hi = x0_min + half, right_limit - half

        if lo <= hi:
            move_to(min(max(x_want, lo), hi), to_frac(*moving.get_position())[1])
            return

        # 放不下: 依序縮標註 -> 縮 AUC -> 認了, 靠左擺
        if fs > min_fontsize + 1e-6:
            moving.set_fontsize(max(min_fontsize, fs * 0.92))
        elif fixed_min is not None and fixed.get_fontsize() > fixed_min + 1e-6:
            fixed.set_fontsize(max(fixed_min, fixed.get_fontsize() * 0.92))
        else:
            move_to(max(x_lo + half, right_limit - half),
                    to_frac(*moving.get_position())[1])
            return
        fig.canvas.draw()


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
               label='Threshold')

    ax.set_xlabel('Score (Normalized)')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

    # 預留上方空間, 確保 legend 不會壓到最高的長條
    n0, _ = np.histogram(s0, bins=bins)
    n1, _ = np.histogram(s1, bins=bins)
    ax.set_ylim(0, max(n0.max(), n1.max()) * 1.15)

    ax.minorticks_on()
    ax.legend(frameon=True, edgecolor=COLORS['grid'],
              framealpha=LEGEND_ALPHA, loc='upper left', borderpad=0.35,
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
    # 標註放在 F1 峰值斜下方一點點即可; 原本固定在 y=0.28,
    # 而峰值約在 y=0.9, 連接線橫跨了 0.6 的高度, 過長。
    f1_at_best = float(np.interp(M['best_thr'], M['thr_grid'], M['f1_lst']))
    # 置中於峰值正下方: 三條曲線在最佳 threshold 附近都在 0.9 以上,
    # 因此峰值下方一段是空的; 靠右擺會壓到 recall/F1 的下降段。
    ax.annotate(f"Threshold = {M['best_thr']:.2f}",
                xy=(M['best_thr'], f1_at_best),
                xytext=(M['best_thr'], f1_at_best - 0.22), ha='center',
                fontsize=6.5,
                arrowprops=dict(arrowstyle='->', color='#555555',
                                linewidth=0.7, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=COLORS['grid'], linewidth=0.5,
                          alpha=LABEL_ALPHA),
                zorder=6)

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.minorticks_on()
    leg = ax.legend(loc='lower left', frameon=True, edgecolor=COLORS['grid'],
                    framealpha=LEGEND_ALPHA, borderpad=0.35,
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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # AUC 放在混淆矩陣正上方的空白帶, 避免壓到 ROC 曲線陡升段
    auc_txt = ax.text(0.97, 0.52, f"AUC = {M['roc_auc']:.2f}",
                      transform=ax.transAxes, fontsize=7, ha='right',
                      va='bottom',
                      bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                                edgecolor=COLORS['grid'], linewidth=0.5,
                                alpha=LABEL_ALPHA),
                      zorder=6)

    # Threshold 標註: 與 (b) 同樣的作法 —— 置中掛在操作點正下方, 用短箭頭
    # 指回去。原本靠右偏移 (+0.14) 會直接撞到右側的 AUC 方框, 在 subset
    # 版 (每格只有 1.62" 寬) 尤其明顯。
    # 位置與字級交給 _avoid_overlap() 依實際文字寬度決定, 不寫死。
    thr_annot = ax.annotate(
        f"Threshold = {M['best_thr']:.2f}",
        xy=(M['best_fpr'], M['best_tpr']),
        xytext=(M['best_fpr'], M['best_tpr'] - 0.20),
        ha='center', va='top', fontsize=6.5,
        arrowprops=dict(arrowstyle='->', color='#555555', linewidth=0.7,
                        shrinkA=1, shrinkB=3),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=COLORS['grid'], linewidth=0.5, alpha=LABEL_ALPHA),
        zorder=6)

    _avoid_overlap(thr_annot, auc_txt, ax,
                   x_lo=0.03, x_hi=0.98,
                   y_floor=INSET_RECT[1] + INSET_RECT[3],
                   curve=(M['fpr'], M['tpr']))

    # ---- confusion matrix inset (右下空白處) ----
    cm = np.asarray(M['cm'])
    inset = ax.inset_axes(list(INSET_RECT))
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
        # rowLoc 預設是 'left', 會讓左側的 Same/Diff (Human) 貼齊左緣;
        # 設為 'center' 才與上方欄標題的對齊方式一致。
        cellLoc='center', rowLoc='center', colLoc='center', loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(TABLE_EDGE)
        cell.set_linewidth(0.5)
        if row == 0 or col == -1:
            # Patch._set_facecolor() 會拿 self._alpha 去覆寫顏色裡的 alpha
            # (to_rgba(color, self._alpha)), 所以必須先把 artist 的 alpha
            # 設回 None, 帶 alpha 的 facecolor 才留得住。先 set_alpha(1.0)
            # 再 set_facecolor 的話 0.8 會被吃掉, 看起來完全沒生效。
            cell.set_alpha(None)
            cell.set_facecolor(to_rgba(COLORS['cell_head'], TABLE_HEAD_ALPHA))
            cell.set_text_props(fontweight='bold', fontsize=5.2, ha='center')
        else:
            cell.set_alpha(TABLE_ALPHA)
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
def ids_from_overlay(path):
    """
    由 render_pairs.py 的輸出檔名解析出 FC / EM 的 ID。

    檔名格式: {fc_id}_{em_id}_composite.png  或  {fc_id}_{em_id}_main.png
    fc_id 本身含有 '-' 但不含 '_', em_id 是純數字, 因此以最後兩個
    底線分段來切: [...fc_id...]_[em_id]_[composite|main]

    解析失敗回傳 (None, None), 呼叫端可退回手動指定。
    """
    stem = Path(path).stem                      # 去掉 .png
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None
    # 最後一段必須是已知後綴, 否則像 'xxx_yyy_0_big' 這種檔名
    # 會被誤判成 em_id = '0'
    if parts[-1] not in ('composite', 'main', 'zoom'):
        return None, None
    em = parts[-2]
    fc = '_'.join(parts[:-2])
    if not em.isdigit():
        return None, None
    return fc, em


def _norm_id(v):
    """把 ID 統一成去空白的字串: em_id 在不同 CSV 裡可能被 pandas 讀成
    int64 或 str, 直接比對會全部找不到。"""
    s = str(v).strip()
    return s[:-2] if s.endswith('.0') and s[:-2].isdigit() else s


def lookup_pair_scores(df, fc_ids, em_ids, score_col='model_pred',
                       normalize=True, tag='score'):
    """查出三組配對在 df 裡的分數, 回傳 list[float|None] (查無回 None)。

    normalize=True 時做 min-max 正規化, 與 compute_all_metrics() 內部
    一致 —— 圖上的 threshold 是正規化後的數值, 案例分數若用原始尺度,
    讀者拿去和 (a) 的虛線比對會對不上。
    """
    if df is None or score_col not in df.columns:
        return [None] * len(fc_ids)

    d = df[['fc_id', 'em_id', score_col]].copy()
    d['fc_id'] = d['fc_id'].map(_norm_id)
    d['em_id'] = d['em_id'].map(_norm_id)

    v = d[score_col].to_numpy(dtype=float)
    if normalize:
        lo, hi = np.nanmin(v), np.nanmax(v)
        print(f'[score] {tag}: min-max 取自 {len(d)} 筆, '
              f'原始範圍 {lo:.3f} ~ {hi:.3f}')
        d[score_col] = (v - lo) / (hi - lo) if hi > lo else v

    out = []
    for fc, em in zip(fc_ids, em_ids):
        hit = d[(d['fc_id'] == _norm_id(fc)) & (d['em_id'] == _norm_id(em))]
        if hit.empty:
            print(f'[warn] {tag}: 查無配對 {fc} <-> {em}, 圖例留空')
            out.append(None)
            continue
        if len(hit) > 1 and hit[score_col].std() > 1e-6:
            print(f'[warn] {tag}: {fc} <-> {em} 有 {len(hit)} 筆且分數不一致 '
                  f'({hit[score_col].to_list()}), 取平均')
        out.append(float(hit[score_col].mean()))
    return out


def load_model_scores_for_nblast(cfg):
    """nblast 模式下, 另外把「我們模型」的預測讀進來供圖例並列顯示。

    重用 load_real_data() 的 cross 分支, 不另外假設檔案結構: 讀
    label_csv_prefix 那組 fold CSV, 且不套 selected_test_set —— 案例
    神經元不一定落在 --subset 指定的子集裡, 限縮了反而會查不到。
    """
    sub = dict(cfg)
    sub['test_mode'] = 'cross'
    sub['selected_test_set'] = False
    try:
        _, _, df, _, _ = load_real_data(sub)
        return df
    except FileNotFoundError as e:
        print(f'[warn] 讀不到模型預測檔 ({e}), 圖例只顯示 NBLAST 分數')
        return None


def panel_overlay(ax, img, fc_id, em_id, score_lines=None):
    """
    一格 3D 疊合圖 (render_pairs.py 產生的 *_composite.png) + 文字標註。

    score_lines: [(名稱, 數值|None), ...] 最多兩項, 以 legend 的第二欄
    呈現 —— 第 0 項與 FC 同列, 第 1 項與 EM 同列。分數不另起新行是為了
    省高度: 每多一行 6pt 文字, 整張圖就要長高 0.125", 三張圖累積起來很可觀。
    nblast 模式下的語意是 NBLAST 分數與模型分數; 兩者都是「這一對」的
    分數, 與排在哪一列無關, 純粹是排版考量。

    尺寸一致性: imshow 的 aspect='equal' 只保證影像本身不變形,
    但「顯示大小」由 gridspec 的格子決定 —— 三張來源影像若像素尺寸
    不同, 會被縮放到不同倍率, 三個腦就一大一小。
    因此 build_figure_subset() 會檢查三張圖的 shape 是否一致並警告。
    """
    if img is not None:
        ax.imshow(img, interpolation='lanczos', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=3.5,
               markerfacecolor=NEURON_FC, markeredgecolor='none',
               label=f'FC: {fc_id}'),
        Line2D([], [], marker='o', linestyle='none', markersize=3.5,
               markerfacecolor=NEURON_EM, markeredgecolor='none',
               label=f'EM: {em_id}'),
    ]

    if not score_lines:
        ax.legend(handles=handles, loc='upper left',
                  bbox_to_anchor=(-0.02, -0.01), frameon=False,
                  fontsize=6, handletextpad=0.4,
                  labelspacing=0.3, borderpad=0.0)
        return

    # 分數另開一欄, 不是接在 ID 字串後面補空格。
    # 補空格對不齊: Arial 是比例字型, 空格寬度固定 (6pt 字約 1.67pt),
    # 而 ID 的寬度是連續值, 只能對到最近的空格倍數, 誤差最大半個空格。
    # legend 的 ncol 佈局是真正的表格, 欄寬取該欄最寬的項目,
    # 所以第二欄的兩個標籤起點必然對齊, 與 ID 多長無關。
    # 填充順序是 column-major: 前兩項成為第一欄, 後兩項成為第二欄。
    for i in range(2):
        if i < len(score_lines):
            name, val = score_lines[i]
            txt = f'{name}: ' + ('n/a' if val is None else f'{val:.2f}')
        else:
            txt = ' '                          # 佔位, 維持兩欄兩列的結構
        handles.append(Line2D([], [], linestyle='none', marker='none',
                              label=txt))

    ax.legend(handles=handles, ncol=2, loc='upper left',
              bbox_to_anchor=(-0.02, -0.01), frameon=False,
              fontsize=6, handlelength=0.9, handletextpad=0.4,
              columnspacing=0.8, labelspacing=0.3, borderpad=0.0)


def build_figure_subset(M, roc_color, overlays, fc_ids, em_ids,
                        human_label='Human', score_lines=None):
    """
    Fig.6 / 7 / 9 / 10 的版面:
        上排  (a) score distribution
              (b) precision / recall / F1 vs threshold
              (c) ROC + confusion matrix inset   <- 與 Fig.5 的 (c) 同款
        下排  (d1)(d2)(d3) 三組配對在標準腦中的疊合圖
    """
    plt.rcParams.update(PAPER_STYLE)

    # 分數改接在 FC/EM 兩行的行尾, 圖例行數不變, 因此畫布高度維持 3.3"。
    # (先前把分數另起新行的版本需要把畫布加高, 否則 savefig.bbox=None
    #  不會自動放大, 文字會直接被裁掉。)
    fig = plt.figure(figsize=(FIG_WIDTH_SUBSET, FIG_HEIGHT_SUBSET))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.00, 0.80],
        wspace=0.34, hspace=0.42,
        left=0.070, right=0.985, top=0.935, bottom=0.075,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a_distribution(ax_a, M)
    panel_b_threshold_curve(ax_b, M)
    panel_c_roc_confmat(ax_c, M, roc_color, human_label=human_label)

    # 標號改用 figure 座標。用 transAxes 的相對值會隨面板高度變動:
    # 版面從 4.2" 縮到 3.3" 後, 1.18 換算成絕對距離超過了 top=0.935
    # 留下的空間, 標號基線落在 y = -1.1 pt, 整個被裁到畫布外。
    fig.canvas.draw()
    y_top = max(a.get_position().y1 for a in (ax_a, ax_b, ax_c))
    for ax, lab in zip([ax_a, ax_b, ax_c], ['(a)', '(b)', '(c)']):
        fig.text(ax.get_position().x0 - 0.058, y_top + 0.012, lab,
                 fontsize=10, fontweight='bold', va='bottom', ha='left')

    axes_d = []
    for i in range(3):
        axd = fig.add_subplot(gs[1, i])
        panel_overlay(axd, overlays[i] if overlays else None,
                      fc_ids[i], em_ids[i],
                      score_lines=score_lines[i] if score_lines else None)
        axes_d.append(axd)

    # 標號以 figure 座標統一貼齊該排頂端: imshow(aspect='equal') 會改變
    # 各軸的實際 bbox, 用 transAxes 會讓三個標號高度不一。
    fig.canvas.draw()
    y_lab = max(a.get_position().y1 for a in axes_d) + 0.012
    for i, axd in enumerate(axes_d):
        fig.text(axd.get_position().x0 - 0.058, y_lab + 0.010, f'(d{i+1})',
                 fontsize=10, fontweight='bold', va='bottom', ha='left')

    return fig


def build_figure_row(M, df, roc_color, cfg, human_label='Human'):
    """
    1x4 橫排: (a)(b)(c)(d) 並列。

    最省版面高度 (僅佔 textheight 的 21%), 但每格只有約 1.27" 寬,
    比 2x2 的 2.54" 窄一半。主要壓力在 (c) 的混淆矩陣 inset:
    每欄約 0.33" 寬, 而 "Diff.(Model)" 這種兩行標籤最少需要 0.22",
    勉強放得下但沒有餘裕。
    """
    plt.rcParams.update(PAPER_STYLE)

    fig = plt.figure(figsize=(FIG_WIDTH_ROW, FIG_HEIGHT_ROW))
    gs = gridspec.GridSpec(
        1, 4, figure=fig,
        wspace=0.30,
        left=0.048, right=0.992, top=0.90, bottom=0.185,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_a, ax_b, ax_c, ax_d = axes

    panel_a_distribution(ax_a, M)
    panel_b_threshold_curve(ax_b, M)
    panel_c_roc_confmat(ax_c, M, roc_color, human_label=human_label)
    names, acc, n_total = compute_recall_at_k(df, cfg['top_k'])
    panel_d_recall_at_k(ax_d, names, acc, n_total)

    # (d) 的柱頂百分比在 1.27" 寬的格子裡會互相重疊 (實測 53 pt^2),
    # 改用整數並縮小字級。
    for t in ax_d.texts:
        if t.get_text().endswith('%'):
            v = t.get_text().rstrip('%')
            t.set_text(f'{float(v):.0f}%')
            t.set_fontsize(5.0)

    # 窄格子: 縮小圖例與刻度字級, 否則會擠爆
    for ax in axes:
        ax.tick_params(labelsize=5.5)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(5.2)

    fig.canvas.draw()
    for ax, lab in zip(axes, ['(a)', '(b)', '(c)', '(d)']):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.040, bb.y1 + 0.020, lab,
                 fontsize=9, fontweight='bold', va='bottom', ha='left')

    return fig


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

    # 子圖標籤: 用 figure 座標而非 transAxes。
    # transAxes 的相對值會隨面板高度變動 —— 版面從 5.8" 縮到 4.6" 後,
    # 1.09 換算成的絕對距離不足以避開 y 軸頂端刻度, 標號會壓到 "1.0"。
    fig.canvas.draw()
    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], ['(a)', '(b)', '(c)', '(d)']):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.048, bb.y1 + 0.012, lab,
                 fontsize=10, fontweight='bold', va='bottom', ha='left')

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
    ap.add_argument('--layout', default='full', choices=['full', 'subset', 'row'],
                    help="full=Fig.5 的 2x2; row=1x4 橫排 (最矮); "
                         "subset=Fig.6/7/9/10 的上排3格+下排3張疊合圖")
    ap.add_argument('--overlay', nargs=3, default=None,
                    help='三張疊合圖 (render_pairs.py 的 *_composite.png)')
    ap.add_argument('--neuron-fc', nargs=3, default=None,
                    help='三個 FC id; 留空則由 --overlay 的檔名自動解析')
    ap.add_argument('--neuron-em', nargs=3, default=None,
                    help='三個 EM id; 留空則由 --overlay 的檔名自動解析')
    ap.add_argument('--human-label', default='Human',
                    help="混淆矩陣中人工標註的名稱 (預設 Human, 不建議用 BRC)")
    ap.add_argument('--no-case-scores', dest='case_scores',
                    action='store_false',
                    help='不要在 (d1)-(d3) 的圖例顯示分數')
    ap.set_defaults(case_scores=True)
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
    if args.layout == 'subset':
        overlays = None
        if args.overlay:
            overlays = []
            for p_ in args.overlay:
                if not os.path.exists(p_):
                    raise FileNotFoundError(f'找不到疊合圖: {p_}')
                overlays.append(mpimg.imread(p_))
            shapes = {im.shape[:2] for im in overlays}
            if len(shapes) > 1:
                print(f'[warn] 三張疊合圖的像素尺寸不一致: {shapes}')
                print('[warn] 它們會被縮放到相同的格子寬度, 導致三個腦的')
                print('[warn] 顯示大小不同。請用相同畫布尺寸重新匯出。')
            else:
                print(f'[image] 三張疊合圖尺寸一致 {shapes.pop()}')
        else:
            print('[image] 未提供 --overlay, 下排留白')

        # ID 只用於圖例文字。優先由 --overlay 的檔名解析,
        # 這樣換配對時只要改 --overlay 一處即可。
        fc_ids, em_ids = args.neuron_fc, args.neuron_em
        if (fc_ids is None or em_ids is None) and args.overlay:
            parsed = [ids_from_overlay(p_) for p_ in args.overlay]
            if all(f and e for f, e in parsed):
                fc_ids = [f for f, _ in parsed]
                em_ids = [e for _, e in parsed]
                print('[ids]   由檔名解析:')
                for f, e in parsed:
                    print(f'          FC {f}  <->  EM {e}')
            else:
                bad = [p_ for p_, (f, e) in zip(args.overlay, parsed)
                       if not (f and e)]
                raise SystemExit(
                    '無法由檔名解析 ID:\n  ' + '\n  '.join(bad) +
                    '\n檔名需為 {fc_id}_{em_id}_composite.png 格式, '
                    '或改用 --neuron-fc / --neuron-em 手動指定。')
        if fc_ids is None or em_ids is None:
            fc_ids = ['', '', '']
            em_ids = ['', '', '']

        # ---- 圖例中的分數 ----
        # 主分數 = 這張圖正在評估的方法 (cross/single 是模型, nblast 是
        # NBLAST), 直接取自已經載入的 df。nblast 模式再多讀一次模型預測,
        # 讓「NBLAST 給高分但我們的模型不給」這件事直接寫在圖上。
        score_lines = None
        if args.case_scores and not args.demo:
            primary = lookup_pair_scores(df, fc_ids, em_ids, tag=tag)
            rows = [[(tag, v)] for v in primary]
            if cfg['test_mode'] == 'nblast':
                mdf = load_model_scores_for_nblast(cfg)
                second = lookup_pair_scores(mdf, fc_ids, em_ids, tag='Model')
                for r, v in zip(rows, second):
                    r.append(('Model', v))
            score_lines = rows
            for (f_, e_), r in zip(zip(fc_ids, em_ids), rows):
                shown = '  '.join(f'{n}={v:.3f}' if v is not None else f'{n}=n/a'
                                  for n, v in r)
                print(f'[score] {f_} <-> {e_}   {shown}')

        fig = build_figure_subset(M, roc_color, overlays,
                                  fc_ids, em_ids,
                                  human_label=args.human_label,
                                  score_lines=score_lines)
    elif args.layout == 'row':
        fig = build_figure_row(M, df, roc_color, cfg,
                               human_label=args.human_label)
    else:
        fig = build_figure(M, df, roc_color, cfg, human_label=args.human_label)

    os.makedirs(cfg['out_dir'], exist_ok=True)
    stem = os.path.join(cfg['out_dir'], cfg['out_name'])
    fw, fh = fig.get_size_inches()
    fig.savefig(f'{stem}.pdf', format='pdf')
    fig.savefig(f'{stem}.png', format='png')
    plt.close(fig)

    print()
    print(f'✓ 輸出完成')
    print(f'   {stem}.pdf   矢量圖 (Type42 字體), 投稿用')
    print(f'   {stem}.png   300 DPI, 預覽用')
    print(f'   物理尺寸 {fw:.1f}" x {fh:.1f}" '
          f'= {fw*25.4:.0f} x {fh*25.4:.0f} mm')
    print(f'   ⚠ 置入論文時請勿縮放, 直接以 100% 尺寸插入')


if __name__ == '__main__':
    main()
