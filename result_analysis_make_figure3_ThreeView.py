#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
make_figure3.py  —  MorphoMatcher Fig.3  三視圖配對比較圖
=============================================================================

每一列 = 一組 FC-EM 配對:
    左段  FlyCircuit (LM)  的三視圖   xy / yz / xz
    中段  hemibrain (EM)   的三視圖   xy / yz / xz
    右段  兩者在標準腦中的 3D 疊合圖 (含腦殼背景)

刻意安排 2 組判定為 same type + 1 組判定為 different type。

-----------------------------------------------------------------------------
為什麼要放一組 different type
-----------------------------------------------------------------------------
只放正例時, 讀者無法校準「模型在區分什麼」——看完只知道相同的神經元長得像。
真正有資訊量的是負例, 且必須是「難負例」: 空間位置接近、NBLAST 會給高分、
但形態確實不同的那種。隨便挑一個明顯不同的神經元當負例等於沒放。

這也直接呼應論文 Discussion 中「NBLAST 會把 spatially close yet
morphologically distinct 的神經元判成匹配」的論證。

-----------------------------------------------------------------------------
3D 疊合圖的準備方式 (重要)
-----------------------------------------------------------------------------
從你的 3D 工具匯出時, 請匯出「乾淨的圖」:
    - 不要燒進任何文字 (FC/EM ID、面板標籤、圖例)
    - 不要燒進座標軸指示箭頭
    - 三張圖使用相同的腦部視角、相同的畫布尺寸
    - 背景設為白色或透明 (PNG)
    - 神經元配色請與本腳本的 NEURON_FC / NEURON_EM 一致
所有文字標註交給本腳本處理, 這樣字體才會與其他圖一致。

-----------------------------------------------------------------------------
使用方式
-----------------------------------------------------------------------------
  # 用合成資料 + 佔位腦圖測版面
  python result_analysis_make_figure3_ThreeView.py --demo

  # 接真實資料
  python result_analysis_make_figure3_ThreeView.py \
      --pair1 fru-F-500297 1051630846 same \
      --pair2 TH-F-000101  331662710  same \
      --pair3 VGlut-F-700541 1702306037 diff \
      --overlay ./brain_png/p1.png ./brain_png/p2.png ./brain_png/p3.png
=============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import cm as mcm
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# --- 沿用 Fig.2 的樣式與繪圖函式, 確保兩張圖視覺語法一致 ---------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import importlib
    _f2 = importlib.import_module('result_analysis_make_figure2_3DProjection')
    PAPER_STYLE    = _f2.PAPER_STYLE
    PROJ_CMAP      = _f2.PROJ_CMAP
    CMAP_FLOOR     = _f2.CMAP_FLOOR
    truncated_cmap = _f2.truncated_cmap
    draw_grid_panel = _f2.draw_grid_panel
    max_projection = _f2.max_projection
    cube_limits    = _f2.cube_limits
    interpolate_skeleton = _f2.interpolate_skeleton
    compute_strahler = _f2.compute_strahler
    make_demo_skeleton = _f2.make_demo_skeleton
    parent_to_index = _f2.parent_to_index
    views_from_npz  = _f2.views_from_npz
except Exception as e:
    raise SystemExit(
        f'無法匯入 Fig.2 腳本: {e}\n'
        '請把本檔與 result_analysis_make_figure2_3DProjection.py 放在同一資料夾。'
    )

try:
    from swc_util import load_swc_fast
except ImportError as e:
    raise SystemExit(f'無法匯入 swc_util: {e}')


# =============================================================================
# 設定
# =============================================================================
# 神經元配色 — 需與外部 3D 渲染工具的設定一致
NEURON_FC = '#D62728'      # 紅 — FlyCircuit
NEURON_EM = '#1F3FBF'      # 藍 — hemibrain (EM)

# 判定標籤的配色
VERDICT_STYLE = {
    'same': dict(text='Same type',      fg='#1B5E20', bg='#E3F2E5', ec='#7CB342'),
    'diff': dict(text='Different type', fg='#8E1B1B', bg='#FBE6E6', ec='#D98A8A'),
}

FC_DIR_DEFAULT  = 'data/standard_views/FC'
EM_DIR_DEFAULT  = 'data/standard_views/EM'
SWC_FC_DEFAULT  = 'data/SWC/FC'
SWC_EM_DEFAULT  = 'data/SWC/EM'

VIEW_NAMES = ('xy', 'yz', 'xz')

FIG_WIDTH  = 7.2
FIG_HEIGHT = 4.6


# =============================================================================
# 資料取得
# =============================================================================
def load_views_for(nid, views_dir, swc_dir, view_order=VIEW_NAMES):
    """
    優先讀 standard_views 的 npz (與模型輸入一致);
    找不到就退回用 SWC 現場重算。
    回傳 dict{plane: HxH array}
    """
    npz = Path(views_dir) / f'{nid}_views.npz'
    if npz.exists():
        v50, _ = views_from_npz(npz)
        return {view_order[k]: v50[k] for k in range(3)}, 'npz'

    for ext in ('.swc', '.SWC'):
        swc_p = Path(swc_dir) / f'{nid}{ext}'
        if swc_p.exists():
            swc = load_swc_fast(swc_p)
            xyz = swc.xyz.astype(np.float64)
            pidx = parent_to_index(swc)
            stra = compute_strahler(pidx)
            lims = cube_limits(xyz)
            pts, val = interpolate_skeleton(xyz, pidx, stra)
            return {p: max_projection(pts, val, p, lims) for p in VIEW_NAMES}, 'swc'

    raise FileNotFoundError(
        f'找不到 {nid} 的 views npz ({npz}) 或 SWC ({swc_dir}/{nid}.swc)')


def demo_views(seed):
    """合成一組三視圖, 供 --demo 測版面"""
    xyz, pidx, stra = make_demo_skeleton(seed=seed)
    lims = cube_limits(xyz)
    pts, val = interpolate_skeleton(xyz, pidx, stra)
    return {p: max_projection(pts, val, p, lims) for p in VIEW_NAMES}


def demo_overlay(seed, w=900, h=620):
    """合成一張帶「腦殼」的疊合圖佔位圖"""
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3))

    yy, xx = np.mgrid[0:h, 0:w]
    # 兩個側腦葉 + 中央腦
    for cx, cy, rx, ry in [(0.26, 0.45, 0.22, 0.30),
                           (0.74, 0.45, 0.22, 0.30),
                           (0.50, 0.46, 0.30, 0.33)]:
        m = (((xx - w * cx) / (w * rx)) ** 2 +
             ((yy - h * cy) / (h * ry)) ** 2) < 1.0
        img[m] = 0.93

    def hex2rgb(hx):
        return np.array([int(hx[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0

    for color, off in [(NEURON_FC, 0), (NEURON_EM, 11)]:
        c = hex2rgb(color)
        r2 = np.random.default_rng(seed * 17 + off)
        x, y = w * 0.46, h * 0.40
        for _ in range(520):
            x += r2.normal(0, 8); y += r2.normal(0, 6)
            xi = int(np.clip(x, 1, w - 2)); yi = int(np.clip(y, 1, h - 2))
            img[yi - 1:yi + 1, xi - 1:xi + 1] = c
    return img


# =============================================================================
# 繪圖
# =============================================================================
def draw_overlay_panel(ax, img, fc_id, em_id, verdict,
                       scalebar_um=None, px_per_um=None):
    """3D 疊合圖 + 所有文字標註 (由 matplotlib 繪製, 非燒進圖片)"""
    if img is not None:
        ax.imshow(img, interpolation='bilinear', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 比例尺
    if scalebar_um and px_per_um and img is not None:
        h, w = img.shape[0], img.shape[1]
        bar = scalebar_um * px_per_um
        x0, y0 = w * 0.05, h * 0.95
        ax.plot([x0, x0 + bar], [y0, y0], color='black',
                linewidth=1.6, solid_capstyle='butt')
        ax.text(x0 + bar / 2, y0 - h * 0.025, f'{scalebar_um} µm',
                ha='center', va='bottom', fontsize=5.5)

    # 判定標籤放在面板內部「左下角」。
    # 放在上方 (面板外或面板內頂端) 都會與欄群組標題
    # (Overlay in standard brain) 相撞, 實測重疊達 69 pt^2。
    v = VERDICT_STYLE[verdict]
    ax.text(0.02, 0.03, v['text'], transform=ax.transAxes,
            fontsize=6.5, fontweight='bold', ha='left', va='bottom',
            color=v['fg'], zorder=8,
            bbox=dict(boxstyle='round,pad=0.28', facecolor=v['bg'],
                      edgecolor=v['ec'], linewidth=0.7))


def build_figure(pairs, overlays, cmap=None, scalebar_um=None, px_per_um=None):
    """
    pairs : list of dict(fc_id, em_id, verdict, fc_views, em_views)
    overlays : list of ndarray or None
    """
    plt.rcParams.update(PAPER_STYLE)
    cmap = cmap or PROJ_CMAP
    nrow = len(pairs)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    # 欄位: FC三視圖 | 間隔 | EM三視圖 | 間隔 | 疊合圖
    gs = gridspec.GridSpec(
        nrow, 9, figure=fig,
        # 第 7 欄是 EM 與 overlay 之間的間隔, 需容納 colorbar + 刻度 + 標籤
        width_ratios=[1, 1, 1, 0.32, 1, 1, 1, 0.88, 2.45],
        wspace=0.16, hspace=0.30,
        left=0.065, right=0.965, top=0.855, bottom=0.085)

    for r, pr in enumerate(pairs):
        # ---- FC 三視圖 ----
        for k, name in enumerate(VIEW_NAMES):
            ax = fig.add_subplot(gs[r, k])
            draw_grid_panel(ax, pr['fc_views'][name], '',
                            cmap=cmap, annotate_size=False)
            ax.set_xticks([]); ax.set_yticks([])
            # 平面名稱放在最後一列下方, 而非第一列上方:
            # 上方需留給 (a1)(b1)(c1) 標號, 兩者會互相壓字。
            if r == nrow - 1:
                ax.set_xlabel(name, fontsize=6.5, labelpad=2)
            if k == 0:
                ax.set_ylabel(pr['fc_id'], fontsize=6, labelpad=2,
                              color=NEURON_FC)

        # ---- EM 三視圖 ----
        for k, name in enumerate(VIEW_NAMES):
            ax = fig.add_subplot(gs[r, 4 + k])
            draw_grid_panel(ax, pr['em_views'][name], '',
                            cmap=cmap, annotate_size=False)
            ax.set_xticks([]); ax.set_yticks([])
            if r == nrow - 1:
                ax.set_xlabel(name, fontsize=6.5, labelpad=2)
            if k == 0:
                ax.set_ylabel(pr['em_id'], fontsize=6, labelpad=2,
                              color=NEURON_EM)

        # ---- 3D 疊合圖 ----
        axo = fig.add_subplot(gs[r, 8])
        draw_overlay_panel(axo, overlays[r], pr['fc_id'], pr['em_id'],
                           pr['verdict'],
                           scalebar_um=scalebar_um, px_per_um=px_per_um)

        # 標號稍後統一以 figure 座標放置 (見 tag_blocks)

    # ---- 區塊標號 ----
    # 依原圖慣例按「欄」分組: a=FC 三視圖, b=EM 三視圖, c=疊合圖,
    # 數字為列序, 故 (a1)(a2)(a3) 由上而下對應三組配對。
    #
    # 用 figure 座標而非各軸的 transAxes: 疊合圖的軸高度與三視圖面板不同
    # (imshow aspect='equal' 會改變軸的實際 bbox), 用 transAxes 會讓
    # (c1) 比 (a1)(b1) 高出一截。統一貼齊該列三視圖面板的頂端。
    fig.canvas.draw()
    for r in range(nrow):
        y_top = fig.axes[r * 7].get_position().y1
        for tag, ax_ref in ((f'(a{r+1})', fig.axes[r * 7]),
                            (f'(b{r+1})', fig.axes[r * 7 + 3]),
                            (f'(c{r+1})', fig.axes[r * 7 + 6])):
            fig.text(ax_ref.get_position().x0, y_top + 0.012, tag,
                     fontsize=8.5, fontweight='bold', va='bottom', ha='left')

    # ---- 欄群組標題 ----
    top = fig.axes[0].get_position()
    def group_title(c0, c1, text, color='#333333'):
        b0 = fig.axes[c0].get_position()
        b1 = fig.axes[c1].get_position()
        fig.text((b0.x0 + b1.x1) / 2, top.y1 + 0.062, text,
                 ha='center', va='bottom', fontsize=7.5,
                 fontweight='bold', color=color)

    group_title(0, 2, 'FlyCircuit (LM)', NEURON_FC)
    group_title(3, 5, 'hemibrain (EM)', NEURON_EM)
    bo = fig.axes[6].get_position()
    fig.text(bo.x0 + bo.width / 2, top.y1 + 0.062,
             'Overlay in standard brain',
             ha='center', va='bottom', fontsize=7.5, fontweight='bold',
             color='#333333')

    # ---- FC / EM 圖例 (放在疊合圖欄下方) ----
    handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=4,
               markerfacecolor=NEURON_FC, markeredgecolor='none', label='FlyCircuit'),
        Line2D([], [], marker='o', linestyle='none', markersize=4,
               markerfacecolor=NEURON_EM, markeredgecolor='none', label='hemibrain'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(bo.x0 + bo.width / 2, 0.002),
               ncol=2, frameon=False, fontsize=6.5,
               handletextpad=0.4, columnspacing=1.2)

    # ---- colorbar: 每列一支, 與該列三視圖面板等高並對齊 ----
    # 不用一支貫穿全圖的長色條: 三列面板高度僅 0.56", 長色條會遠高於內容,
    # 視覺上突兀且與任何一列都對不齊。
    #
    # 標籤用兩行縮寫: 完整的 "Normalized Strahler number" 在 fontsize 6 下
    # 需 1.19", 遠超過面板高度 0.56"; 兩行寫法最長一行僅約 0.38"。
    # 完整量綱名稱寫在圖說中。
    fig.canvas.draw()
    for r in range(nrow):
        bp = fig.axes[r * 7].get_position()          # 該列第一個三視圖面板
        b_em = fig.axes[r * 7 + 5].get_position()    # 該列 EM 區塊最右面板
        sm = mcm.ScalarMappable(cmap=truncated_cmap(cmap),
                                norm=plt.Normalize(0, 1))
        sm.set_array([])
        # 緊鄰 EM 三視圖右側, 位於 overlay 欄左方
        cax = fig.add_axes([b_em.x1 + 0.013, bp.y0, 0.011, bp.height])
        cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1.0])
        cb.set_label('Normalized\nStrahler', fontsize=5.5, labelpad=2)
        cb.ax.tick_params(labelsize=5, width=0.5, length=1.8)
        cb.outline.set_linewidth(0.5)

    return fig


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description='Build MorphoMatcher Figure 3')
    ap.add_argument('--demo', action='store_true', help='合成資料 + 佔位腦圖')
    for i in (1, 2, 3):
        ap.add_argument(f'--pair{i}', nargs=3, metavar=('FC_ID', 'EM_ID', 'VERDICT'),
                        default=None,
                        help='FC id, EM id, 判定 (same|diff)')
    ap.add_argument('--overlay', nargs=3, default=None,
                    help='三張 3D 疊合圖 (乾淨圖, 不含文字)')
    ap.add_argument('--fc-views', default=FC_DIR_DEFAULT)
    ap.add_argument('--em-views', default=EM_DIR_DEFAULT)
    ap.add_argument('--fc-swc',   default=SWC_FC_DEFAULT)
    ap.add_argument('--em-swc',   default=SWC_EM_DEFAULT)
    ap.add_argument('--scalebar-um', type=float, default=None)
    ap.add_argument('--px-per-um',   type=float, default=None)
    ap.add_argument('--cmap', default=PROJ_CMAP)
    ap.add_argument('--out',    default='Figure3')
    ap.add_argument('--outdir', default='./Figure')
    args = ap.parse_args()

    # ---- 組出三組配對 ----
    pairs = []
    if args.demo:
        print('[demo] 使用合成資料')
        spec = [('fru-F-500297', '1051630846', 'same'),
                ('TH-F-000101',  '331662710',  'same'),
                ('VGlut-F-700541', '1702306037', 'diff')]
        for i, (fc, em, vd) in enumerate(spec):
            pairs.append(dict(fc_id=fc, em_id=em, verdict=vd,
                              fc_views=demo_views(3 + i * 5),
                              em_views=demo_views(4 + i * 5)))
    else:
        specs = [getattr(args, f'pair{i}') for i in (1, 2, 3)]
        if any(s is None for s in specs):
            raise SystemExit('請提供 --pair1 --pair2 --pair3, 或用 --demo')
        for fc, em, vd in specs:
            if vd not in ('same', 'diff'):
                raise SystemExit(f'判定必須是 same 或 diff, 收到 {vd}')
            fcv, src1 = load_views_for(fc, args.fc_views, args.fc_swc)
            emv, src2 = load_views_for(em, args.em_views, args.em_swc)
            print(f'[data] {fc} ({src1})  vs  {em} ({src2})  -> {vd}')
            pairs.append(dict(fc_id=fc, em_id=em, verdict=vd,
                              fc_views=fcv, em_views=emv))

    n_same = sum(1 for p in pairs if p['verdict'] == 'same')
    print(f'[data] {len(pairs)} 組配對: {n_same} same, {len(pairs)-n_same} diff')
    if n_same == len(pairs):
        print('[warn] 全部都是正例。建議至少放一組難負例 (空間接近但形態不同),')
        print('[warn] 否則讀者無法校準模型在區分什麼。')

    # ---- 疊合圖 ----
    if args.overlay:
        overlays = []
        for p in args.overlay:
            if not os.path.exists(p):
                raise FileNotFoundError(f'找不到疊合圖: {p}')
            overlays.append(mpimg.imread(p))
        shapes = {im.shape[:2] for im in overlays}
        if len(shapes) > 1:
            print(f'[warn] 三張疊合圖尺寸不一致 {shapes}')
            print('[warn] 建議用相同畫布尺寸匯出, 否則腦的大小看起來會不同')
    else:
        print('[image] 未提供 --overlay, 使用佔位腦圖')
        overlays = [demo_overlay(i + 1) for i in range(len(pairs))]

    fig = build_figure(pairs, overlays, cmap=args.cmap,
                       scalebar_um=args.scalebar_um, px_per_um=args.px_per_um)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, args.out)
    fig.savefig(f'{stem}.pdf', format='pdf')
    fig.savefig(f'{stem}.png', format='png')
    plt.close(fig)

    fw, fh = FIG_WIDTH, FIG_HEIGHT
    print()
    print('✓ 輸出完成')
    print(f'   {stem}.pdf   /   {stem}.png')
    print(f'   物理尺寸 {fw:.1f}" x {fh:.1f}" = {fw*25.4:.0f} x {fh*25.4:.0f} mm')


if __name__ == '__main__':
    main()
