# =============================================================================
# 抓出 FC / EM 的三視圖並存成 npz
#
#
# 需與 result_analysis_make_figure2_3DProjection.py 及 swc_util.py 同一資料夾
# =============================================================================
# %%
import numpy as np
from pathlib import Path
import importlib

f2 = importlib.import_module('result_analysis_make_figure2_3DProjection')

VIEW_ORDER = ('xy', 'yz', 'xz')   # npz 中三個 view 的軸對應, 對不上就調這裡


def get_views(nid, views_dir=None, swc_dir=None):
    """
    取得某條神經元的三視圖 (3, 50, 50), 值域 0-1。
    優先讀 standard_views 的 npz (與模型輸入一致);
    找不到就用 SWC 現場重算。
    """
    # --- 路徑1: standard_views npz ---
    if views_dir:
        npz = Path(views_dir) / f'{nid}_views.npz'
        if npz.exists():
            v50, raw_shape = f2.views_from_npz(npz)
            print(f'  {nid}: npz {raw_shape} -> {v50.shape}')
            return {VIEW_ORDER[k]: v50[k] for k in range(3)}

    # --- 路徑2: 由 SWC 重算 ---
    if swc_dir:
        for ext in ('.swc', '.SWC'):
            p = Path(swc_dir) / f'{nid}{ext}'
            if p.exists():
                swc  = f2.load_swc_fast(p)
                xyz  = swc.xyz.astype(np.float64)
                pidx = f2.parent_to_index(swc)
                stra = f2.compute_strahler(pidx)
                lims = f2.cube_limits(xyz)
                pts, val = f2.interpolate_skeleton(xyz, pidx, stra)
                print(f'  {nid}: swc {len(xyz)} nodes -> 重算投影')
                return {pl: f2.max_projection(pts, val, pl, lims)
                        for pl in VIEW_ORDER}

    raise FileNotFoundError(f'{nid}: 在 {views_dir} / {swc_dir} 都找不到')


# =============================================================================
# 使用: 改成你的實際 ID 與路徑, 執行後會產生 fig4_views.npz
# =============================================================================
FC_ID = 'TH-F-100083'
EM_ID = '331662710'

FC_VIEWS_DIR = 'data/standard_views/FC'
EM_VIEWS_DIR = 'data/standard_views/EM'
FC_SWC_DIR   = 'data/SWC/FC'
EM_SWC_DIR   = 'data/SWC/EM'

print('取得三視圖:')
fc = get_views(FC_ID, FC_VIEWS_DIR, FC_SWC_DIR)
em = get_views(EM_ID, EM_VIEWS_DIR, EM_SWC_DIR)

out = {}
for tag, d in (('fc', fc), ('em', em)):
    for pl in VIEW_ORDER:
        out[f'{tag}_{pl}'] = d[pl].astype(np.float32)

np.savez_compressed('fig4_views.npz', **out)

print('\n✓ 已存成 fig4_views.npz')
print(f'  FC = {FC_ID},  EM = {EM_ID}')
for k, v in out.items():
    print(f'  {k:8s} shape={v.shape}  非零格 {int((v>0).sum()):4d}/{v.size}  '
          f'max={v.max():.3f}')
# %%
