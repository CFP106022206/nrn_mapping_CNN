# %% 以官方 NBLAST 重算 similarity matrix 用的 316 顆 FC 神經兩兩之間的分數
# 直接沿用 analysis_model_results/run_nblast_official.py (論文 D1/D2 所用) 的 dotprops 建構,
# 計分參數與該腳本的預設值一致:
#   navis + smat.fcwb, dotprops k=5, 重採樣 1 um, normalized=True, scores="mean", 不用 alpha
# 舊的 NBLAST_316.npy 為單向分數, 且 FB 區塊與官方實作不符 (r = 0.06), 不再使用。
#
# 執行 (需要 navis):
#   conda run -n nblast python similarity_matrix/nblast_official_316.py
#   可選參數 --n-cores 16 --k 5 --resample 1.0 --alpha
#
# 輸出 (列/欄順序固定為 ALLN.txt, PN.txt, KC.txt, FB.txt 串接, 與 NBLAST_316.npy 相同):
#   NBLAST_316_official.npy   316x316 矩陣, 對角線 = 1
#   NBLAST_316_official.csv   長表 fc_id, em_id, nblast_official (與 FTmodel_predict.csv
#                             相同慣例: 這裡的 em_id 實際上也是 fc_id)
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'analysis_model_results'))
from run_nblast_official import build_dotprops  # noqa: E402

LPU = ['ALLN', 'PN', 'KC', 'FB']


def load_order():
    ids = []
    for l in LPU:
        d = pd.read_csv(os.path.join(HERE, l + '.txt'), header=None, names=['fc_id'])
        d['fc_id'] = d['fc_id'].astype(str).str.strip().str.replace('﻿', '', regex=False)
        ids += d.loc[d['fc_id'] != '', 'fc_id'].tolist()
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-cores', type=int, default=16)
    # 以下三個預設值與 run_nblast_official.py 相同
    ap.add_argument('--k', type=int, default=5, help='dotprops 的最近鄰數')
    ap.add_argument('--resample', type=float, default=1.0, help='重採樣步長 (um)')
    ap.add_argument('--alpha', action='store_true', help='啟用 NBLAST 的 UseAlpha 加權')
    args = ap.parse_args()

    import navis
    from navis.nbl.smat import smat_fcwb
    navis.set_pbars(hide=True)
    print(f'navis {navis.__version__}, smat.fcwb (alpha={args.alpha}), '
          f'k={args.k}, resample={args.resample} um, n_cores={args.n_cores}')

    order = load_order()
    assert len(order) == 316 and len(set(order)) == 316, '名單應為 316 顆不重複的 FC 神經'
    t0 = time.time()
    dps = build_dotprops(order, 'FC', navis, args.k, args.resample)
    missing = [n for n in order if n not in dps]
    if missing:
        sys.exit(f'缺 SWC: {missing}')
    print(f'dotprops {len(dps)} 顆 ({time.time()-t0:.0f}s)', flush=True)

    nl = navis.NeuronList([dps[n] for n in order])
    m = navis.nblast(nl, nl, scores='mean', smat=smat_fcwb(alpha=args.alpha), normalized=True,
                     use_alpha=args.alpha, progress=False, n_cores=args.n_cores)
    m = m.loc[order, order].to_numpy(dtype=float)

    np.save(os.path.join(HERE, 'NBLAST_316_official.npy'), m)
    fc_grid, em_grid = np.meshgrid(order, order, indexing='ij')
    pd.DataFrame({'fc_id': fc_grid.ravel(), 'em_id': em_grid.ravel(),
                  'nblast_official': m.ravel()}).to_csv(
        os.path.join(HERE, 'NBLAST_316_official.csv'), index=False)

    print(f'完成 ({time.time()-t0:.0f}s)  shape={m.shape}  max|M-M.T|={np.abs(m - m.T).max():.2e}  '
          f'對角線 [{np.diag(m).min():.3f}, {np.diag(m).max():.3f}]  範圍 [{m.min():.3f}, {m.max():.3f}]')


if __name__ == '__main__':
    main()
