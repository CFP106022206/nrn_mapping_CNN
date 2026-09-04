#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figure4.py — Fig.4 架構圖 (依實際 MVCNN_Siamese 定義繪製)

輸出 PDF / PNG / SVG。SVG 用 Inkscape 開啟後按 Ctrl+Shift+G 數次解散群組即可編輯。

架構要點 (取自 MVCNN_Siamese()):
  單一視角  50x50x1
    conv1 3x3 x32 -> gelu -> conv2 3x3 x32 -> gelu -> maxpool 2x2   -> 25x25x32
    conv3 3x3 x64 -> gelu -> conv4 3x3 x64 -> BN(每視角獨立) -> gelu
                                            -> maxpool 2x2          -> 12x12x64
    dropout 0.1 -> flatten                                          -> 9216
  View pooling  3 views -> element-wise max                         -> 9216
  Siamese       concat(FC, EM)                                      -> 18432
  分類頭        dropout 0.3 -> Dense 256 -> BN -> gelu -> Dense 1 sigmoid

注意兩個容易畫錯的地方:
  1. 是「兩階段融合」: 先 view pooling 把 3 視角併成 1 個向量,
     才把 FC/EM 兩個向量 concatenate。不是 6 支直接匯流。
     view pooling 正是 MVCNN 的核心, 少畫它等於抽掉引用 MVCNN 的理由。
  2. conv1-4 與 activation 共享權重, 但第 4 層的 BatchNormalization
     是每個視角各自獨立的 (原始碼有明確註解), 圖上需標示。
"""
# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.bbox': None, 'savefig.pad_inches': 0.0,
    'figure.dpi': 300, 'savefig.dpi': 300,
    # 強制白底: 預設的 'auto'/透明在深色檢視器或投影片上會變黑底
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.transparent': False,
})

# figure 的長寬比 —— 用來校正歸一化座標下的形變。
# ax 的座標是 0-1 歸一化, 但 figure 是 6.5 x 3.6 吋,
# 若寬高都用同一個數值, 正方形會被橫向拉伸 6.5/3.6 = 1.806 倍。
# 因此凡是需要「正方形」的元素 (三視圖是 50x50), 高度都要乘上 AR。

C_FC    = '#D62728'
C_EM    = '#1F3FBF'
C_CONV  = '#DCE7F2'
C_CONVE = '#6E8FB4'
C_POOL  = '#FCE9D6'
C_POOLE = '#D89A5B'
C_VP    = '#E4F0E4'
C_VPE   = '#6FA46F'
C_HEAD  = '#EFE6F5'
C_HEADE = '#8E6BA8'
C_LINE  = '#555555'
C_GRAY  = '#999999'
C_SHARE = '#B03A2E'

FIG_W, FIG_H = 6.5, 3.6
AR = FIG_W / FIG_H          # 歸一化座標的形變校正係數


def rbox(ax, x, y, w, h, fc, ec, lw=0.7, r=0.008, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f'round,pad=0,rounding_size={r}',
                 facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(ax, x1, y1, x2, y2, color=C_LINE, lw=0.8, z=4, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=6, linewidth=lw, color=color,
                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))


def real_proj(ax, x, y, sw, sh, grid):
    """
    畫一張真實的 50x50 投影。
    sw / sh 分開給: 在歸一化座標下要畫出正方形, 高度必須乘上 figure 長寬比。
    """
    ax.add_patch(Rectangle((x, y), sw, sh, facecolor='white',
                 edgecolor='#666666', linewidth=0.5, zorder=3))
    n = grid.shape[0]
    cm = plt.get_cmap('YlGnBu')
    for i, j in np.argwhere(grid > 0):
        ax.add_patch(Rectangle((x + j*sw/n, y + (n-1-i)*sh/n), sw/n, sh/n,
                     facecolor=cm(0.35 + 0.6*grid[i, j]),
                     edgecolor='none', zorder=4))


def mini_proj(ax, x, y, sw, sh, seed):
    rng = np.random.default_rng(seed)
    ax.add_patch(Rectangle((x, y), sw, sh, facecolor='white',
                 edgecolor='#666666', linewidth=0.5, zorder=3))
    n = 12
    g = np.zeros((n, n)); px = py = n // 2
    for _ in range(38):
        g[np.clip(py, 0, n-1), np.clip(px, 0, n-1)] = rng.uniform(.4, 1)
        px = int(np.clip(px + rng.integers(-1, 2), 0, n-1))
        py = int(np.clip(py + rng.integers(-1, 2), 0, n-1))
    cm = plt.get_cmap('YlGnBu')
    for i in range(n):
        for j in range(n):
            if g[i, j] > 0:
                ax.add_patch(Rectangle((x + j*sw/n, y + i*sh/n), sw/n, sh/n,
                             facecolor=cm(0.35 + 0.6*g[i, j]),
                             edgecolor='none', zorder=4))


def load_real_views(path='fig4_views.npz'):
    """讀 dump_views.py 產生的 npz; 找不到就回傳 None (改用示意圖)"""
    import os
    if not os.path.exists(path):
        return None
    z = np.load(path)
    out = {}
    for tag in ('fc', 'em'):
        out[tag] = {pl: z[f'{tag}_{pl}'] for pl in ('xy', 'yz', 'xz')}
    print(f'[views] 使用真實三視圖: {path}')
    return out


def main(views_npz='fig4_views.npz'):
    real = load_real_views(views_npz)
    if real is None:
        print('[views] 找不到 npz, 使用合成示意圖')
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # ================= (a) 共享的單視角 CNN =================
    ax.text(0.008, 0.975, '(a)', fontsize=9.5, fontweight='bold', va='top')
    ax.text(0.048, 0.975, 'Shared single-view CNN', fontsize=8,
            fontweight='bold', va='top', color='#333333')

    yb, bh = 0.70, 0.105
    x = 0.085
    ax.text(x - 0.006, yb + bh/2, r'$50\times50$'+'\n'+r'$\times 1$', fontsize=5.6,
            ha='right', va='center', color='#444444', linespacing=1.5)

    blocks = [
        ('conv1\n3×3, 32', C_CONV, C_CONVE, 0.072),
        ('conv2\n3×3, 32', C_CONV, C_CONVE, 0.072),
        ('pool\n2×2',      C_POOL, C_POOLE, 0.050),
        ('conv3\n3×3, 64', C_CONV, C_CONVE, 0.072),
        ('conv4\n3×3, 64', C_CONV, C_CONVE, 0.072),
        ('pool\n2×2',      C_POOL, C_POOLE, 0.050),
    ]
    dims = [None, r'$50^2\!\times\!32$', None, r'$25^2\!\times\!32$',
            None, r'$12^2\!\times\!64$']

    for i, (lbl, fc, ec, bw) in enumerate(blocks):
        rbox(ax, x, yb, bw, bh, fc, ec)
        ax.text(x + bw/2, yb + bh/2, lbl, fontsize=5.6, ha='center',
                va='center', linespacing=1.6)
        if dims[i]:
            ax.text(x + bw + 0.009, yb - 0.030, dims[i], fontsize=5,
                    ha='center', va='center', color='#666666')
        if i < len(blocks) - 1:
            arrow(ax, x + bw + 0.002, yb + bh/2, x + bw + 0.017, yb + bh/2, lw=0.7)
        x += bw + 0.019

    # gelu 標註
    ax.text(0.085 + 0.036, yb + bh + 0.022, 'gelu after each conv',
            fontsize=5.5, ha='left', va='bottom', color='#666666',
            style='italic')

    # dropout + flatten
    rbox(ax, x, yb, 0.085, bh, '#F2F2F2', '#999999')
    ax.text(x + 0.0425, yb + bh/2, 'dropout 0.1\nflatten',
            fontsize=5.6, ha='center', va='center', linespacing=1.6)
    ax.text(x + 0.0425, yb - 0.030, '9216', fontsize=5.5, ha='center',
            va='center', color='#666666', fontweight='bold')

    # 共享權重框
    ax.plot([0.080, x + 0.090, x + 0.090, 0.080, 0.080],
            [yb - 0.012, yb - 0.012, yb + bh + 0.012, yb + bh + 0.012, yb - 0.012],
            color=C_SHARE, lw=0.7, ls=(0, (3, 2)), zorder=6)
    ax.text(x + 0.093, yb + bh/2,
            'weights shared across\nall views and both\ndatabases',
            fontsize=5.0, ha='left', va='center', color=C_SHARE,
            style='italic', linespacing=1.6)

    # BN 例外標註
    ax.annotate('BatchNorm is per-view (not shared)',
                xy=(0.085 + 3*(0.072+0.019) + 0.050 + 0.019 + 0.036, yb),
                xytext=(0.44, 0.545), fontsize=5.3, ha='center', va='top',
                color=C_SHARE, style='italic', linespacing=1.3,
                arrowprops=dict(arrowstyle='->', color=C_SHARE, lw=0.6))

    # ================= (b) 完整流程 =================
    ax.text(0.008, 0.455, '(b)', fontsize=9.5, fontweight='bold', va='top')
    ax.text(0.048, 0.455, 'Multi-view Siamese network', fontsize=8,
            fontweight='bold', va='top', color='#333333')

    # 三視圖是 50x50, 必須是正方形: 寬用 VSX, 高用 VSX*AR
    VSX, VG = 0.043, 0.010
    VSY = VSX * AR
    views = ['xy', 'yz', 'xz']

    for tag, col, y0, sd in [('FC', C_FC, 0.285, 10), ('EM', C_EM, 0.075, 40)]:
        ax.plot([0.020, 0.020], [y0 - 0.004, y0 + VSY + 0.004],
                color=col, lw=1.5, solid_capstyle='butt')
        ax.text(0.026, y0 + VSY/2, tag, fontsize=7, fontweight='bold',
                color=col, ha='left', va='center')
        for k in range(3):
            xx = 0.055 + k*(VSX + VG)
            if real is not None:
                real_proj(ax, xx, y0, VSX, VSY, real[tag.lower()][views[k]])
            else:
                mini_proj(ax, xx, y0, VSX, VSY, sd + k)
            ax.text(xx + VSX/2, y0 - 0.016, views[k], fontsize=5,
                    ha='center', va='top', color='#555555')

        # -> 共享 CNN
        xcnn = 0.055 + 3*VSX + 2*VG + 0.030
        for k in range(3):
            arrow(ax, 0.055 + 3*VSX + 2*VG + 0.004, y0 + VSY/2,
                  xcnn - 0.002, y0 + VSY/2, color=C_GRAY, lw=0.55)
        rbox(ax, xcnn, y0 + VSY/2 - 0.030, 0.072, 0.060, C_CONV, C_CONVE)
        ax.text(xcnn + 0.036, y0 + VSY/2, 'shared\nCNN (a)', fontsize=5.4,
                ha='center', va='center', linespacing=1.6)
        ax.text(xcnn + 0.036, y0 + VSY/2 + 0.036, r'$3\times$9216',
                fontsize=5, ha='center', va='bottom', color='#666666')

        # -> view pooling
        xvp = xcnn + 0.072 + 0.030
        arrow(ax, xcnn + 0.072 + 0.003, y0 + VSY/2, xvp - 0.002, y0 + VSY/2, lw=0.7)
        rbox(ax, xvp, y0 + VSY/2 - 0.030, 0.080, 0.060, C_VP, C_VPE)
        ax.text(xvp + 0.040, y0 + VSY/2, 'view pooling\nmax over 3',
                fontsize=5.4, ha='center', va='center', linespacing=1.6)
        ax.text(xvp + 0.040, y0 + VSY/2 - 0.036, '9216', fontsize=5,
                ha='center', va='top', color='#666666')

    # --- concatenate (兩支匯流) ---
    xvp_end = xvp + 0.080
    xcat = xvp_end + 0.032
    ycat = 0.185
    rbox(ax, xcat, ycat - 0.048, 0.072, 0.096, C_VP, C_VPE)
    ax.text(xcat + 0.036, ycat, 'concat\nFC + EM', fontsize=5.4,
            ha='center', va='center', linespacing=1.6)
    ax.text(xcat + 0.036, ycat - 0.058, '18432', fontsize=5, ha='center',
            va='top', color='#666666')
    for y0 in (0.285, 0.075):
        arrow(ax, xvp_end + 0.003, y0 + VSY/2, xcat - 0.002, ycat,
              color=C_GRAY, lw=0.6)

    # --- 分類頭 ---
    xh = xcat + 0.072 + 0.030
    arrow(ax, xcat + 0.072 + 0.003, ycat, xh - 0.002, ycat, lw=0.7)
    rbox(ax, xh, ycat - 0.058, 0.088, 0.116, C_HEAD, C_HEADE)
    ax.text(xh + 0.044, ycat + 0.030, 'dropout 0.3', fontsize=5.2,
            ha='center', va='center')
    ax.text(xh + 0.044, ycat + 0.004, 'Dense 256', fontsize=5.2,
            ha='center', va='center')
    ax.text(xh + 0.044, ycat - 0.022, 'BN + gelu', fontsize=5.2,
            ha='center', va='center')

    # --- 輸出 ---
    xo = xh + 0.088 + 0.030
    arrow(ax, xh + 0.088 + 0.003, ycat, xo - 0.002, ycat, lw=0.7)
    rbox(ax, xo, ycat - 0.048, 0.078, 0.096, '#FFFFFF', '#555555')
    ax.text(xo + 0.039, ycat + 0.026, 'Dense 1', fontsize=5.4,
            ha='center', va='center')
    ax.text(xo + 0.039, ycat - 0.006, 'sigmoid', fontsize=5.4,
            ha='center', va='center', style='italic', color='#555555')
    ax.text(xo + 0.039, ycat - 0.030, 'same / different',
            fontsize=5.2, ha='center', va='center', color='#333333')

    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'./Figure/Figure4.{ext}',
                    facecolor='white', edgecolor='none',
                    transparent=False)
    plt.close(fig)
    print(f'✓ Figure4.pdf / .png / .svg   ({FIG_W}" x {FIG_H}")')


if __name__ == '__main__':
    main()

# %%
