
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
result_analysis_make_figure2_3DProjection.py  —  MorphoMatcher Fig.2  三視圖投影示意圖
=============================================================================

建構在專案既有的 swc_util.py 之上, 確保圖中呈現的投影
與模型實際吃到的輸入完全一致。

用到的 swc_util 函式:
    load_swc_fast()        讀取骨架 (xyz, parent, radius)
    _load_views_from_npz() 讀取預先算好的 standard views
    _ensure_3hw_views()    統一成 (3,H,W)
    _to_uint8_views()      統一成 uint8
    _resize_to_50()        adaptive MAX pooling 降到 50x50

-----------------------------------------------------------------------------
兩種投影來源
-----------------------------------------------------------------------------
  A. --views  讀取 standard_views 的 npz  <-- 建議, 與模型輸入位元級一致
  B. 不給 --views 時, 由 SWC 現場重算, 重算邏輯刻意與管線對齊:
       - 正方形 (立方體) 包圍盒, 不做逐軸拉伸
       - MAX pooling (非求和), 與 _resize_to_50 一致
       - uint8 0-255

-----------------------------------------------------------------------------
使用方式
-----------------------------------------------------------------------------
  # 合成骨架, 測版面
  python result_analysis_make_figure2_3DProjection.py --demo

  # 只有 SWC (投影現場重算)
  python result_analysis_make_figure2_3DProjection.py --swc ./data/SWC/FC/Trh-F-000008.swc
  # 橫式
  python result_analysis_make_figure2_3DProjection.py --swc ./data/SWC/FC/Trh-F-000008.swc --layout side

  # SWC + 真實 standard views (最忠實, 建議論文用這個)
  python result_analysis_make_figure2_3DProjection.py \
      --swc   ./data/swc/FC/fru-F-500297.swc \
      --views ./data/standard_views/FC/fru-F-500297_views.npz

  # 三個 view 的軸對應順序若不同, 用 --view-order 指定
  python result_analysis_make_figure2_3DProjection.py --swc a.swc --views a.npz --view-order xz,yz,xy

  # 調整視角
  python result_analysis_make_figure2_3DProjection.py --demo --elev 18 --azim -62
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
from matplotlib import cm as mcm
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings('ignore')

# --- 匯入專案的 swc_util ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from swc_util import (
        load_swc_fast,
        _load_views_from_npz,
        _ensure_3hw_views,
        _to_uint8_views,
        _resize_to_50,
    )
except ImportError as e:
    raise SystemExit(
        f'無法匯入 swc_util: {e}\n'
        '請把 result_analysis_make_figure2_3DProjection.py 放在與 swc_util.py 同一個資料夾。'
    )


# =============================================================================
# 全論文統一樣式 (與 Fig.5 / Fig.6 腳本一致)
# =============================================================================
PAPER_STYLE = {
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        8,
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'legend.fontsize':  6.5,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'axes.linewidth':   0.8,
    'figure.dpi':       600,
    'savefig.dpi':      600,
    # 不用 bbox='tight': 它會裁掉白邊, 使輸出的 PDF 比 figsize 窄
    # (實測 cube 版 6.5" -> 4.88")。之後用 width=\textwidth 置入時
    # 會被「放大」, 字級同樣跑掉。設為 None 才能保證輸出 = figsize,
    # 搭配 width=\textwidth 為 1:1。留白由 gridspec 的 left/right 控制。
    'savefig.bbox':     None,
    'savefig.pad_inches': 0.0,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'mathtext.default': 'regular',
}

PROJ_CMAP  = 'YlGnBu'      # 投影面色階; 見下方說明
SKEL_CMAP  = 'OrRd'        # 3D 骨架色階; 刻意與 PROJ_CMAP 分開, 見下方說明
PLANE_BG   = 'none'        # 'none' = 零值透明 ; 'dark' = 深色面板
CMAP_FLOOR = 0.35          # 色階下端截斷點, 見下方說明
GRID_COLOR = '#D8D8D8'

# -----------------------------------------------------------------------------
# 為什麼骨架與投影要用不同色階
# -----------------------------------------------------------------------------
# 兩者若共用同一色階, 在 (a) 立方體中骨架與其投影會呈現幾乎相同的顏色,
# 讀者無法分辨「哪個是 3D 結構, 哪個是它在牆面上的投影」。
# 因此:
#     投影面 -> PROJ_CMAP (YlGnBu, 冷色系)
#     3D骨架 -> SKEL_CMAP (OrRd,   暖色系)
# 兩者色相互補, 且都是「深 = 高 Strahler」, 語意方向一致。
# 代價是 (a) 需要兩支 colorbar, 這是必要的資訊成本。
# -----------------------------------------------------------------------------
swc_default = 'data/SWC/FC/Trh-F-000008.swc'


# -----------------------------------------------------------------------------
# 關於 colormap 的選擇 (實測數據)
# -----------------------------------------------------------------------------
# 白色背景上, 各 colormap 高權重端對背景的對比度 (WCAG, >=3.0 才可辨識):
#
#     magma      v=1.00 -> 1.05     權重最高的主幹幾乎看不見
#     viridis    v=1.00 -> 1.26     同上
#     YlGnBu     v=1.00 -> 15.83    佳
#     magma_r    v=1.00 -> 20.95    佳
#
# 「亮=高」的 colormap 與亮背景在語意上互斥, 故白底需用 YlGnBu / magma_r 這類
# 「深=高」的色階。本檔預設 YlGnBu。
#
# 但 YlGnBu 的低端同樣接近白色, 對比度為:
#     v=0.25 -> 1.34     v=0.33 -> 1.64     v=0.50 -> 2.42
# normalized Strahler 的最小非零值 (細枝) 常落在 0.25 附近,
# 直接使用會讓細枝在白底上消失。因此把資料映射到色階的
# [CMAP_FLOOR, 1.0] 區間, 避開最淡的一段。
#
# 用 --cmap / --plane-bg / --cmap-floor 切換, 例如:
#   --cmap YlGnBu  --plane-bg none      (預設, 白底)
#   --cmap magma   --plane-bg dark      (深色面板, 同現行 Fig.3)
# -----------------------------------------------------------------------------


def truncated_cmap(name, floor=None, n=256):
    """
    取 colormap 的 [floor, 1.0] 區段重新組成新的 colormap。
    避開亮色階最淡的一段, 否則低權重的細枝在白底上會消失。
    """
    from matplotlib.colors import LinearSegmentedColormap
    floor = CMAP_FLOOR if floor is None else floor
    base  = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f'{name}_trunc', base(np.linspace(floor, 1.0, n)))

# 兩種 layout 各有尺寸: --layout cube 用 FIG_*, --layout combined 用 FIG_*_COMBINED
# 寬度對齊單欄 LaTeX 的 \textwidth (A4, margin=2.2cm) = 6.535 inch,
# 搭配 \includegraphics[width=\textwidth] 為 1:1 置入, 字級不會被縮放。
FIG_WIDTH  = 6.5          # --layout cube
FIG_HEIGHT = 5.5

FIG_WIDTH_COMBINED  = 6.5   # --layout combined
FIG_HEIGHT_COMBINED = 5.6   # 原 6.9; 實測有 22.7% 為純空白, 已壓掉

FIG_WIDTH_SIDE  = 6.5       # --layout side (橫式, 最省高度)
FIG_HEIGHT_SIDE = 3.3
N_GRID     = 50          # 與 _resize_to_50 的預設一致

# 平面 -> 使用哪兩個座標軸
PLANE_AXES = {'xy': (0, 1), 'yz': (1, 2), 'xz': (0, 2)}


# =============================================================================
# 骨架處理
# =============================================================================
def parent_to_index(swc) -> np.ndarray:
    """
    SWC 的 parent 欄位存的是 node ID, 不是陣列索引。
    轉成索引陣列, 根節點為 -1。
    """
    id2idx = {int(v): i for i, v in enumerate(swc.nid)}
    return np.array([id2idx.get(int(p), -1) for p in swc.parent], dtype=np.int64)


def compute_strahler(parent_idx: np.ndarray) -> np.ndarray:
    """由 parent 索引計算 Strahler number (向心分支排序)。迭代式後序走訪。"""
    n = len(parent_idx)
    children = [[] for _ in range(n)]
    roots = []
    for i, p in enumerate(parent_idx):
        if p < 0:
            roots.append(i)
        else:
            children[p].append(i)

    order = np.zeros(n, dtype=int)
    for r in roots:
        stack = [(r, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                ch = children[node]
                if not ch:
                    order[node] = 1
                else:
                    vals = [order[c] for c in ch]
                    mx = max(vals)
                    order[node] = mx + 1 if vals.count(mx) > 1 else mx
            else:
                stack.append((node, True))
                for c in children[node]:
                    stack.append((c, False))
    return order


def interpolate_skeleton(xyz, parent_idx, strahler, step=1.0):
    """
    沿每段 node-parent 線段線性內插, 對應論文 Step 1 的 linear interpolation。
    回傳 (點雲 Nx3, 每點的 Strahler 值)
    """
    pts, val = [], []
    for i, p in enumerate(parent_idx):
        if p < 0:
            continue
        a, b = xyz[p], xyz[i]
        d = float(np.linalg.norm(b - a))
        k = max(int(d / step), 1)
        for t in np.linspace(0, 1, k, endpoint=False):
            pts.append(a + (b - a) * t)
            val.append(strahler[i])
    return np.asarray(pts, dtype=np.float64), np.asarray(val, dtype=np.float64)


def cube_limits(xyz, pad_frac=0.06):
    """
    立方體包圍盒。
    swc_util._pad_to_same_size 要求 views 為正方形 (H==W),
    代表管線用的是等向的立方體包圍盒, 而非逐軸拉伸, 這裡保持一致。
    """
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    span   = float((xyz.max(axis=0) - xyz.min(axis=0)).max()) * (1 + 2 * pad_frac)
    half   = span / 2.0
    return [(float(center[k] - half), float(center[k] + half)) for k in range(3)]


# =============================================================================
# 投影
# =============================================================================
def max_projection(pts, weights, plane, lims, n_grid=N_GRID):
    """
    最大值投影 (MAX pooling), 而非加權求和。

    這一點很重要: swc_util._resize_to_50 用的是 np.maximum.reduceat,
    也就是 adaptive MAX pooling。若這裡改用 histogram2d 求和,
    畫出來的圖會比模型實際輸入更「厚」, 密集區被過度強調。
    """
    i, j = PLANE_AXES[plane]
    (lo_i, hi_i), (lo_j, hi_j) = lims[i], lims[j]

    bi = np.clip(((pts[:, i] - lo_i) / (hi_i - lo_i) * n_grid).astype(int), 0, n_grid - 1)
    bj = np.clip(((pts[:, j] - lo_j) / (hi_j - lo_j) * n_grid).astype(int), 0, n_grid - 1)

    H = np.zeros((n_grid, n_grid), dtype=np.float64)
    np.maximum.at(H, (bi, bj), weights)

    if H.max() > 0:
        H = H / H.max()
    return H


def views_from_npz(npz_path, n_grid=N_GRID):
    """
    讀取 standard_views 的 npz, 走一遍與訓練相同的前處理。
    回傳 (3, n_grid, n_grid) 的 float 陣列, 值域 0-1。
    """
    raw = _load_views_from_npz(npz_path)
    v = _ensure_3hw_views(_to_uint8_views(raw))
    v50 = _resize_to_50(v, (n_grid, n_grid))
    return v50.astype(np.float64) / 255.0, v.shape


# =============================================================================
# 繪圖
# =============================================================================
def draw_projection_plane(ax, H, plane, lims, cmap=None,
                          plane_bg=None, alpha=1.0):
    """
    把 2D 投影貼到立方體的背面上。

    plane_bg='none' : 零值全透明, 面板融入白色背景。
                      搭配 magma_r / viridis_r 等「深=高」的 colormap。
    plane_bg='dark' : 零值畫成 colormap 最低色 (深色), 形成不透明面板。
                      搭配 magma / viridis 等「亮=高」的 colormap,
                      與現行 Fig.3 的黑底三視圖一致。
    """
    cmap     = cmap or PROJ_CMAP
    plane_bg = plane_bg or PLANE_BG

    (x0, x1), (y0, y1), (z0, z1) = lims
    n = H.shape[0]

    facecolors = truncated_cmap(cmap)(H.T)
    if plane_bg == 'dark':
        facecolors[..., 3] = alpha            # 整面不透明, 零值即 cmap 最低色
    else:
        facecolors[..., 3] = np.where(H.T > 0, alpha, 0.0)

    if plane == 'xy':
        X, Y = np.meshgrid(np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1))
        Z = np.full_like(X, z0)
    elif plane == 'yz':
        Y, Z = np.meshgrid(np.linspace(y0, y1, n + 1), np.linspace(z0, z1, n + 1))
        X = np.full_like(Y, x0)
    elif plane == 'xz':
        X, Z = np.meshgrid(np.linspace(x0, x1, n + 1), np.linspace(z0, z1, n + 1))
        Y = np.full_like(X, y1)
    else:
        raise ValueError(plane)

    ax.plot_surface(X, Y, Z, facecolors=facecolors,
                    rstride=1, cstride=1, shade=False,
                    antialiased=False, linewidth=0, zorder=0)


def draw_skeleton(ax, xyz, parent_idx, strahler=None, cmap=None):
    """
    Line3DCollection 一次畫完所有線段。
    骨架依 Strahler 上色, 但使用與投影面「不同」的色階 (SKEL_CMAP),
    否則骨架與其在牆面上的投影顏色幾乎相同, 讀者無法分辨兩者。

    取色範圍刻意避開 colormap 的最淡端, 否則細枝在白底上會消失。
    """
    cmap = cmap or SKEL_CMAP
    idx  = [(i, p) for i, p in enumerate(parent_idx) if p >= 0]
    segs = [[xyz[p], xyz[i]] for i, p in idx]

    if strahler is not None and len(segs):
        s = np.array([strahler[i] for i, _ in idx], dtype=float)
        s = s / s.max() if s.max() > 0 else s
        colors = truncated_cmap(cmap)(0.10 + 0.90 * s)
        lws    = 0.45 + 1.10 * s
    else:
        colors, lws = '#B3402A', 0.7

    ax.add_collection3d(Line3DCollection(segs, colors=colors,
                                         linewidths=lws, zorder=10))


def draw_grid_panel(ax, H, title, cmap=None, plane_bg=None,
                    show_pixel_grid=True, annotate_size=True):
    """
    把單一張投影以「模型視角」攤平顯示: 座標軸為像素索引而非 µm,
    讓讀者一眼看出這是 50x50 的離散網格。

    離散性靠三件事表達 (不使用放大框):
      1. interpolation='nearest'  不做插值, 保留方格邊界
      2. 每 1 px 細格線 + 每 10 px 粗格線
      3. 刻度每 10 px 一格, 可直接數出 50
    """
    cmap     = cmap or PROJ_CMAP
    plane_bg = plane_bg or PLANE_BG
    n        = H.shape[0]
    dark     = (plane_bg == 'dark')

    # 零值在白底方案要透明, 深色方案則畫成 cmap 最低色
    Hm = H if dark else np.ma.masked_where(H <= 0, H)
    cm_obj = truncated_cmap(cmap).copy()
    cm_obj.set_bad(alpha=0.0)

    ax.imshow(Hm.T, origin='lower', cmap=cm_obj, vmin=0, vmax=1,
              interpolation='nearest',       # 關鍵: 不做插值, 保留方格邊界
              extent=[0, n, 0, n], zorder=2)

    # ---- 像素格線: 讓離散性可見 ----
    # alpha 需足夠高。實測 alpha=0.13 時, 格線僅比白底暗 24 個灰階
    # (255 -> 231, 對比 1.2), 印刷後等同看不見。
    if show_pixel_grid:
        for k in range(n + 1):
            if k % 10 == 0:
                lw, al = 0.55, 0.55
            else:
                lw, al = 0.22, 0.22
            ax.axvline(k, color='#555555', linewidth=lw, alpha=al, zorder=3)
            ax.axhline(k, color='#555555', linewidth=lw, alpha=al, zorder=3)

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect('equal')
    # 每 10 px 一個刻度, 讀者可直接數出 50
    ticks = list(range(0, n + 1, 10))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=5.5, width=0.6, length=2.5)
    ax.set_title(title, fontsize=8, pad=3)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color('#444444')

    if annotate_size:
        ax.text(0.975, 0.03, f'{n}x{n}', transform=ax.transAxes,
                fontsize=6, ha='right', va='bottom',
                color='#DDDDDD' if dark else '#555555',
                zorder=6)



def build_figure_combined(xyz, parent_idx, strahler, planes, lims,
                          elev=20, azim=-56, cmap=None, plane_bg=None,
                          order=('xy', 'yz', 'xz'), skel_cmap=None):
    """
    上排: 3D 立方體 (幾何關係) + 兩支 colorbar (骨架 / 投影)
    下排: 三張 50x50 攤平圖 (模型實際輸入), 座標軸為像素索引
    """
    plt.rcParams.update(PAPER_STYLE)
    cmap      = cmap or PROJ_CMAP
    skel_cmap = skel_cmap or SKEL_CMAP
    plane_bg = plane_bg or PLANE_BG
    dark     = (plane_bg == 'dark')
    n        = next(iter(planes.values())).shape[0]

    fig = plt.figure(figsize=(FIG_WIDTH_COMBINED, FIG_HEIGHT_COMBINED))
    # mplot3d 的軸本身就保留大量內部留白, 因此 (a) 不需要太大的
    # height_ratio; hspace 也壓到最小, 兩者合計可省下約 20% 的圖高。
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[1.15, 1.0],
                            wspace=0.30, hspace=0.04,
                            left=0.06, right=0.87, top=1.00, bottom=0.08)

    # ---------- (a) 3D 立方體 ----------
    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    for name, H in planes.items():
        draw_projection_plane(ax3d, H, name, lims, cmap=cmap, plane_bg=plane_bg)
    draw_skeleton(ax3d, xyz, parent_idx, strahler=strahler, cmap=skel_cmap)

    c = xyz.mean(axis=0)
    (x0, x1), (y0, y1), (z0, z1) = lims
    g = dict(color='#BBBBBB' if dark else '#888888',
             linestyle=':', linewidth=0.7, zorder=8)
    ax3d.plot([c[0], c[0]], [c[1], c[1]], [c[2], z0], **g)
    ax3d.plot([c[0], x0],   [c[1], c[1]], [c[2], c[2]], **g)
    ax3d.plot([c[0], c[0]], [c[1], y1],   [c[2], c[2]], **g)

    ax3d.set_xlim(lims[0]); ax3d.set_ylim(lims[1]); ax3d.set_zlim(lims[2])
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_xlabel('x  (µm)', labelpad=2)
    ax3d.set_ylabel('y  (µm)', labelpad=2)
    ax3d.set_zlabel('z  (µm)', labelpad=2)
    # 3 個刻度即可。用 4 個時, x 與 y 軸最前端的標籤會在角落互相壓字
    # (例如渲染成 "120-210")。
    ax3d.tick_params(axis='both', which='major', pad=0, labelsize=6)
    for setter, (lo, hi) in zip([ax3d.set_xticks, ax3d.set_yticks,
                                 ax3d.set_zticks], lims):
        setter(np.linspace(lo, hi, 3).round(-1))
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor('white'); pane.set_edgecolor(GRID_COLOR)
        pane.set_alpha(1.0)
    ax3d.grid(True, color=GRID_COLOR, linewidth=0.4)

    lerp = lambda a, b, t: a + (b - a) * t
    ann = dict(fontsize=7, fontweight='bold',
               color='#DDDDDD' if dark else '#444444', zorder=20)
    ax3d.text(lerp(x0, x1, 0.86), lerp(y0, y1, 0.10), z0, 'xy', **ann)
    ax3d.text(x0, lerp(y0, y1, 0.88), lerp(z0, z1, 0.90), 'yz', **ann)
    ax3d.text(lerp(x0, x1, 0.10), y1, lerp(z0, z1, 0.90), 'xz', **ann)

    ax3d.text2D(0.02, 0.97, '(a)', transform=ax3d.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # ---------- (b) 三張 50x50 ----------
    for k, name in enumerate(order):
        axp = fig.add_subplot(gs[1, k])
        draw_grid_panel(axp, planes[name], f'{name} projection',
                        cmap=cmap, plane_bg=plane_bg)
        axp.set_xlabel(f'{name[0]} (px)', fontsize=7, labelpad=2.5)
        if k == 0:
            axp.set_ylabel(f'{name[1]} (px)', fontsize=7, labelpad=2.5)
            axp.text(-0.30, 1.16, '(b)', transform=axp.transAxes,
                     fontsize=10, fontweight='bold', va='top')
        else:
            axp.set_ylabel(f'{name[1]} (px)', fontsize=7, labelpad=2.5)

    # ---------- colorbars ----------
    # (a) 兩支: 骨架與投影使用不同色階
    # (b) 一支, 與下排面板垂直置中對齊
    #
    # 每支都標上完整的量綱名稱, 而非共用一行說明。
    # 共用說明會落在 (a)(b) 之間的空隙, 不明確屬於任何一支,
    # 讀者得把「Projection」與「Normalized Strahler number」自行拼接。
    # 豎向空間足夠, 重複三次反而更清楚。
    #
    # 位置直接取自各軸的實際 bbox, 而非寫死座標,
    # 否則改動 gridspec 時 colorbar 會失去對齊。
    fig.canvas.draw()

    CBAR_LABEL = 'Normalized Strahler number'

    def add_cbar(rect, cmap_name, title, ticks=(0, 0.5, 1.0)):
        sm = mcm.ScalarMappable(cmap=truncated_cmap(cmap_name),
                                norm=plt.Normalize(0, 1))
        sm.set_array([])
        cax = fig.add_axes(rect)
        cb = fig.colorbar(sm, cax=cax, ticks=list(ticks))
        cb.set_label(CBAR_LABEL, fontsize=6, labelpad=3)
        cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)
        # 元素名稱放在色條上方。需靠左對齊並加大 pad:
        # 置中時標題會橫跨到色條右側, 與頂端刻度 "1.0" 重疊。
        cax.set_title(title, fontsize=6, pad=8, color='#333333',
                      loc='left')
        return cb

    CB_W = 0.014          # colorbar 寬度 (figure 座標)
    CB_X = 0.895          # colorbar 左緣

    # --- (a) 兩支 colorbar ---
    # 完整標籤在 fontsize 6 下約 1.20 inch, 每支色條需 >=1.25 inch 才不溢出。
    # 3D 軸高約 3.43 inch, 故取 90% 並把兩支之間的間隔壓到 12%,
    # 得每支約 1.36 inch, 留有餘裕。
    b3d  = ax3d.get_position()
    a_h  = b3d.height * 0.90
    a_y0 = b3d.y0 + b3d.height * 0.05
    gap  = a_h * 0.12
    each = (a_h - gap) / 2.0

    add_cbar([CB_X, a_y0 + each + gap, CB_W, each], skel_cmap, '3D skeleton')
    add_cbar([CB_X, a_y0, CB_W, each], cmap, 'Projection')

    # --- (b) 一支 colorbar, 與下排面板等高並垂直置中 ---
    bp = fig.axes[1].get_position()       # 第一個 50x50 面板
    add_cbar([CB_X, bp.y0, CB_W, bp.height], cmap, 'Projection')

    return fig


def build_figure_side(xyz, parent_idx, strahler, planes, lims,
                      elev=20, azim=-56, cmap=None, plane_bg=None,
                      order=('xy', 'yz', 'xz'), skel_cmap=None):
    """
    橫式佈局: 左為 3D 立方體 (a), 右為三張 50x50 直向堆疊 (b)。

    相較 combined 的上下排, 這個佈局高度只有約一半 (3.3" vs 5.6"),
    對「單欄 + 多張大圖」的論文友善得多 —— LaTeX 的 \topfraction
    預設只允許頂部浮動體佔頁面 70%, 過高的圖會被延後,
    而浮動體是 FIFO, 一張卡住後面全部跟著往後推。
    """
    plt.rcParams.update(PAPER_STYLE)
    cmap      = cmap or PROJ_CMAP
    skel_cmap = skel_cmap or SKEL_CMAP
    plane_bg  = plane_bg or PLANE_BG
    dark      = (plane_bg == 'dark')
    n         = next(iter(planes.values())).shape[0]

    fig = plt.figure(figsize=(FIG_WIDTH_SIDE, FIG_HEIGHT_SIDE))
    # 三欄: [0]=3D 立方體, [1]=空欄 (留給 (a) 的兩支色條), [2]=三張面板
    # 色條放在各自說明的對象旁邊, 而非全部擠在最右側。
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            width_ratios=[1.52, 0.22, 1.0],
                            wspace=0.30, hspace=0.32,
                            left=0.01, right=0.86, top=0.90, bottom=0.11)

    # ---------- (a) 3D 立方體, 佔左側整欄 ----------
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    for name, H in planes.items():
        draw_projection_plane(ax3d, H, name, lims, cmap=cmap, plane_bg=plane_bg)
    draw_skeleton(ax3d, xyz, parent_idx, strahler=strahler, cmap=skel_cmap)

    c = xyz.mean(axis=0)
    (x0, x1), (y0, y1), (z0, z1) = lims
    g = dict(color='#BBBBBB' if dark else '#888888',
             linestyle=':', linewidth=0.7, zorder=8)
    ax3d.plot([c[0], c[0]], [c[1], c[1]], [c[2], z0], **g)
    ax3d.plot([c[0], x0],   [c[1], c[1]], [c[2], c[2]], **g)
    ax3d.plot([c[0], c[0]], [c[1], y1],   [c[2], c[2]], **g)

    ax3d.set_xlim(lims[0]); ax3d.set_ylim(lims[1]); ax3d.set_zlim(lims[2])
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_xlabel('x  (µm)', labelpad=1, fontsize=7)
    ax3d.set_ylabel('y  (µm)', labelpad=1, fontsize=7)
    ax3d.set_zlabel('z  (µm)', labelpad=1, fontsize=7)
    ax3d.tick_params(axis='both', which='major', pad=0, labelsize=5.5)
    for setter, (lo, hi) in zip([ax3d.set_xticks, ax3d.set_yticks,
                                 ax3d.set_zticks], lims):
        setter(np.linspace(lo, hi, 3).round(-1))
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor('white'); pane.set_edgecolor(GRID_COLOR)
        pane.set_alpha(1.0)
    ax3d.grid(True, color=GRID_COLOR, linewidth=0.4)

    lerp = lambda a, b, t: a + (b - a) * t
    ann = dict(fontsize=6.5, fontweight='bold',
               color='#DDDDDD' if dark else '#444444', zorder=20)
    ax3d.text(lerp(x0, x1, 0.86), lerp(y0, y1, 0.10), z0, 'xy', **ann)
    ax3d.text(x0, lerp(y0, y1, 0.88), lerp(z0, z1, 0.90), 'yz', **ann)
    ax3d.text(lerp(x0, x1, 0.10), y1, lerp(z0, z1, 0.90), 'xz', **ann)
    # (a) 標號稍後與 (b) 一起以 figure 座標放置, 兩者才會等高

    # ---------- (b) 三張 50x50 直向堆疊 ----------
    for k, name in enumerate(order):
        axp = fig.add_subplot(gs[k, 2])
        draw_grid_panel(axp, planes[name], f'{name} projection',
                        cmap=cmap, plane_bg=plane_bg)
        axp.set_title(f'{name} projection', fontsize=6.5, pad=2)
        axp.tick_params(labelsize=5)
        axp.set_ylabel(f'{name[1]} (px)', fontsize=6, labelpad=2.5)
        if k == len(order) - 1:
            axp.set_xlabel(f'{name[0]} (px)', fontsize=6, labelpad=2.5)
        else:
            axp.set_xticklabels([])
        # (b) 標號稍後以 figure 座標放置; 用 transAxes 的 1.30 會超出
        # gridspec 的 top, 在畫布外被裁掉。

    # ---------- colorbars ----------
    fig.canvas.draw()
    # 橫式版的色條較短 (約 0.9"), 完整的 'Normalized Strahler number'
    # 在 fontsize 5 下需 0.99", 會溢出並撞到相鄰色條的標題。
    # 改用兩行縮寫, 完整名稱寫在圖說。
    CBAR_LABEL = 'Normalized\nStrahler'

    def add_cbar(rect, cmap_name, title):
        sm = mcm.ScalarMappable(cmap=truncated_cmap(cmap_name),
                                norm=plt.Normalize(0, 1))
        sm.set_array([])
        cax = fig.add_axes(rect)
        cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1.0])
        cb.set_label(CBAR_LABEL, fontsize=5, labelpad=2)
        cb.ax.tick_params(labelsize=4.5, width=0.4, length=1.5)
        cb.outline.set_linewidth(0.5)
        cax.set_title(title, fontsize=5.5, pad=7, color='#333333', loc='left')

    b3d    = ax3d.get_position()
    bp_top = fig.axes[1].get_position()      # (b) 第一張面板 (xy)
    bp_bot = fig.axes[3].get_position()      # (b) 最後一張面板 (xz)

    # --- (a) 旁邊兩支: 立方體同時含骨架與投影兩種色階 ---
    x_a = b3d.x1 + 0.055
    # 兩支之間需要足夠間隔: 上支的縱向標籤底端會撞到下支的標題。
    add_cbar([x_a, b3d.y0 + b3d.height * 0.58, 0.011, b3d.height * 0.26],
             skel_cmap, '3D skeleton')
    add_cbar([x_a, b3d.y0 + b3d.height * 0.06, 0.011, b3d.height * 0.26],
             cmap, 'Projection')

    # --- (b) 右側每張面板各一支, 與該面板等高並對齊 ---
    # 不用一支貫穿三張的長色條: 面板高僅約 0.72", 長色條會遠高於
    # 任何單一面板, 視覺上與誰都對不齊。
    x_b = bp_top.x1 + 0.020
    for k in range(3):
        bpk = fig.axes[1 + k].get_position()
        add_cbar([x_b, bpk.y0, 0.011, bpk.height],
                 cmap, 'Projection' if k == 0 else '')

    # --- (a)(b) 標號: 兩者共用同一個 figure y, 才會上下對齊 ---
    # 3D 軸的 bbox 上緣與面板欄的上緣不一定相同, 若各自用 transAxes
    # 定位, 兩個標號會落在不同高度 (實測相差約 24 pt)。
    y_lab = max(b3d.y1, bp_top.y1) + 0.015
    fig.text(b3d.x0 + 0.015, y_lab, '(a)',
             fontsize=10, fontweight='bold', va='bottom', ha='left')
    fig.text(bp_top.x0 - 0.075, y_lab, '(b)',
             fontsize=10, fontweight='bold', va='bottom', ha='left')

    return fig


def build_figure(xyz, parent_idx, strahler, planes, lims,
                 elev=20, azim=-56, show_colorbar=True,
                 cmap=None, plane_bg=None):
    plt.rcParams.update(PAPER_STYLE)
    cmap     = cmap or PROJ_CMAP
    plane_bg = plane_bg or PLANE_BG
    dark     = (plane_bg == 'dark')

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax  = fig.add_subplot(111, projection='3d')

    for name, H in planes.items():
        draw_projection_plane(ax, H, name, lims, cmap=cmap, plane_bg=plane_bg)
    draw_skeleton(ax, xyz, parent_idx, strahler=strahler, cmap=cmap)

    # ---- 投影引線: 表達「投影」這個動作 ----
    # 深色面板時引線需提亮, 否則末端沒入面板看不見
    c = xyz.mean(axis=0)
    (x0, x1), (y0, y1), (z0, z1) = lims
    g = dict(color='#BBBBBB' if dark else '#888888',
             linestyle=':', linewidth=0.7, zorder=8)
    ax.plot([c[0], c[0]], [c[1], c[1]], [c[2], z0], **g)
    ax.plot([c[0], x0],   [c[1], c[1]], [c[2], c[2]], **g)
    ax.plot([c[0], c[0]], [c[1], y1],   [c[2], c[2]], **g)

    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('x  (µm)', labelpad=2)
    ax.set_ylabel('y  (µm)', labelpad=2)
    ax.set_zlabel('z  (µm)', labelpad=2)
    # 3 個刻度即可。用 4 個時 x 與 y 軸最前端的標籤會在角落互相壓字
    # (渲染成 "180-130" 之類)。與 build_figure_combined 保持一致。
    ax.tick_params(axis='both', which='major', pad=0, labelsize=6)
    for setter, (lo, hi) in zip([ax.set_xticks, ax.set_yticks, ax.set_zticks], lims):
        setter(np.linspace(lo, hi, 3).round(-1))

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor('white')
        pane.set_edgecolor(GRID_COLOR)
        pane.set_alpha(1.0)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4)

    # ---- 平面標註 ----
    lerp = lambda a, b, t: a + (b - a) * t
    ann = dict(fontsize=7, fontweight='bold',
               color='#DDDDDD' if dark else '#444444', zorder=20)
    ax.text(lerp(x0, x1, 0.86), lerp(y0, y1, 0.10), z0, 'xy', **ann)
    ax.text(x0, lerp(y0, y1, 0.88), lerp(z0, z1, 0.90), 'yz', **ann)
    ax.text(lerp(x0, x1, 0.10), y1, lerp(z0, z1, 0.90), 'xz', **ann)

    if show_colorbar:
        sm = mcm.ScalarMappable(cmap=truncated_cmap(cmap), norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.45, aspect=16,
                          pad=0.02, location='right')
        cb.set_label('Normalized Strahler number', fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.5)

    return fig, ax


# =============================================================================
# demo 骨架
# =============================================================================
def make_demo_skeleton(seed=3, n_branch=26):
    """合成一個投射神經元: 一條長主幹 + 兩端展開的樹突/軸突叢"""
    rng = np.random.default_rng(seed)
    coords, parents = [np.array([-120.0, -20.0, 0.0])], [-1]

    cur = 0
    for _ in range(46):
        p = coords[cur] + np.array([5.4, 1.5, 0.7]) + rng.normal(0, 1.3, 3)
        coords.append(p); parents.append(cur); cur = len(coords) - 1
    trunk_end = cur
    anchors = [3, 12, 22, 34, trunk_end]

    def grow(frm, d, steps, spread):
        c = frm
        for _ in range(steps):
            coords.append(coords[c] + d * 4.0 + rng.normal(0, spread, 3))
            parents.append(c); c = len(coords) - 1
        return c

    for tuft in range(2):
        for _ in range(n_branch // 2):
            a = anchors[rng.integers(0, 3)] if tuft == 0 else trunk_end
            d = rng.normal(0, 1, 3); d /= np.linalg.norm(d)
            tip = grow(a, d, int(rng.integers(4, 9)), 2.2)
            for _ in range(2):
                d2 = rng.normal(0, 1, 3); d2 /= np.linalg.norm(d2)
                tip2 = grow(tip, d2, int(rng.integers(3, 6)), 2.0)
                if rng.random() < 0.5:
                    for _ in range(2):
                        d3 = rng.normal(0, 1, 3); d3 /= np.linalg.norm(d3)
                        grow(tip2, d3, int(rng.integers(2, 5)), 1.8)

    xyz = np.asarray(coords, dtype=np.float64)
    parent_idx = np.asarray(parents, dtype=np.int64)
    return xyz, parent_idx, compute_strahler(parent_idx)


# =============================================================================
# main
# =============================================================================
def main():
    global CMAP_FLOOR
    ap = argparse.ArgumentParser(description='Build MorphoMatcher Figure 2')
    ap.add_argument('--demo',  action='store_true', help='用合成骨架測版面')
    ap.add_argument('--swc',   default=swc_default, help='SWC 骨架檔')
    ap.add_argument('--views', default=None,
                    help='standard_views 的 npz (最忠實; 省略則由 SWC 重算)')
    ap.add_argument('--view-order', default='xy,yz,xz',
                    help='npz 中三個 view 對應的平面順序 (預設 xy,yz,xz)')
    ap.add_argument('--cmap', default=PROJ_CMAP,
                    help="投影 colormap (預設 magma_r; 深色面板請搭配 magma)")
    ap.add_argument('--plane-bg', default=PLANE_BG, choices=['none', 'dark'],
                    help="none=零值透明(白底) / dark=深色面板(同現行 Fig.3)")
    ap.add_argument('--skel-cmap', default=SKEL_CMAP,
                    help='3D 骨架的色階 (預設 OrRd, 與投影的冷色系區隔)')
    ap.add_argument('--cmap-floor', type=float, default=CMAP_FLOOR,
                    help='色階下端截斷點 (預設 0.35), 避免低權重細枝在白底上消失')
    ap.add_argument('--layout', default='combined',
                    choices=['cube', 'combined', 'side'],
                    help="combined=上下排(預設) / side=左右排(高度僅一半, "
                         "適合單欄多圖) / cube=僅立方體")
    ap.add_argument('--elev', type=float, default=20)
    ap.add_argument('--azim', type=float, default=-56)
    ap.add_argument('--no-colorbar', action='store_true')
    ap.add_argument('--out',    default='Figure2')
    ap.add_argument('--outdir', default='./Figure')
    args = ap.parse_args()
    CMAP_FLOOR = args.cmap_floor

    # ---- 骨架 ----
    if args.swc:
        print(f'[swc]   load_swc_fast: {args.swc}')
        swc = load_swc_fast(args.swc)
        xyz = swc.xyz.astype(np.float64)
        parent_idx = parent_to_index(swc)
        strahler = compute_strahler(parent_idx)
    elif args.demo:
        print('[swc]   使用合成骨架 (demo)')
        xyz, parent_idx, strahler = make_demo_skeleton()
    else:
        raise SystemExit('請提供 --swc 或使用 --demo')

    n_root = int((parent_idx < 0).sum())
    print(f'[swc]   {len(xyz)} nodes, {n_root} root(s), max Strahler = {strahler.max()}')

    lims = cube_limits(xyz)
    print(f'[bbox]  立方體邊長 {lims[0][1]-lims[0][0]:.1f} µm (等向, 與正方形 views 一致)')

    # ---- 投影 ----
    order = [s.strip() for s in args.view_order.split(',')]
    if sorted(order) != ['xy', 'xz', 'yz']:
        raise SystemExit(f'--view-order 必須是 xy,yz,xz 的某個排列, 收到 {order}')

    if args.views:
        print(f'[views] 讀取 standard views: {args.views}')
        v50, raw_shape = views_from_npz(args.views)
        print(f'[views] 原始 {raw_shape} -> _resize_to_50 -> {v50.shape}  (MAX pooling)')
        planes = {order[k]: v50[k] for k in range(3)}
        print(f'[views] 軸對應 {order[0]}={0}, {order[1]}={1}, {order[2]}={2}'
              '  (若對不上請用 --view-order 調整)')
    else:
        print('[views] 未提供 --views, 由 SWC 現場重算 (MAX projection, 與管線一致)')
        pts, val = interpolate_skeleton(xyz, parent_idx, strahler)
        print(f'[views] 內插後 {len(pts)} 點')
        planes = {p: max_projection(pts, val, p, lims) for p in ('xy', 'yz', 'xz')}

    for name, H in planes.items():
        print(f'[views] {name}: 佔用格數 {int((H>0).sum())}/{H.size}')

    # ---- 繪圖 ----
    print(f'[style] proj_cmap={args.cmap}, skel_cmap={args.skel_cmap}, '
          f'plane_bg={args.plane_bg}, floor={args.cmap_floor}, layout={args.layout}')
    if args.layout == 'side':
        fig = build_figure_side(xyz, parent_idx, strahler, planes, lims,
                                elev=args.elev, azim=args.azim,
                                cmap=args.cmap, plane_bg=args.plane_bg,
                                order=tuple(order), skel_cmap=args.skel_cmap)
    elif args.layout == 'combined':
        fig = build_figure_combined(xyz, parent_idx, strahler, planes, lims,
                                    elev=args.elev, azim=args.azim,
                                    cmap=args.cmap, plane_bg=args.plane_bg,
                                    order=tuple(order), skel_cmap=args.skel_cmap)
    else:
        fig, _ = build_figure(xyz, parent_idx, strahler, planes, lims,
                              elev=args.elev, azim=args.azim,
                              show_colorbar=not args.no_colorbar,
                              cmap=args.cmap, plane_bg=args.plane_bg)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, args.out)
    fig.savefig(f'{stem}.pdf', format='pdf')
    fig.savefig(f'{stem}.png', format='png')
    plt.close(fig)

    print()
    print('✓ 輸出完成')
    print(f'   {stem}.pdf   /   {stem}.png')
    fw, fh = fig.get_size_inches()
    print(f'   物理尺寸 {fw:.1f}" x {fh:.1f}" = {fw*25.4:.0f} x {fh*25.4:.0f} mm')
    print(f'   視角 elev={args.elev}, azim={args.azim}')


if __name__ == '__main__':
    main()
