# %% 量化 Fig.10 相似度矩陣的分群品質：MorphoMatcher (fine-tune / annotator) vs NBLAST
# 產出 Results/Clustering Neurons in FlyCircuit 一節所引用的所有數字
# 用法: 先產生 METHODS 裡列出的矩陣, 再
#       python similarity_matrix/cluster_quantification.py
#   模型分數: python similarity_matrix/model_predict_316.py [--model-dir ... --prefix ... --fold ...]
#   NBLAST  : conda run -n nblast python similarity_matrix/nblast_official_316.py
import os, itertools
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, 'similarity_matrix')
SWC_FC = os.path.join(ROOT, 'data', 'SWC', 'FC')
LPU = ['ALLN', 'PN', 'KC', 'FB']

# 要比較的相似度矩陣 (列/欄順序皆為 ALLN, PN, KC, FB 串接)
#   模型: model_predict_316.py 算出的有序分數, 未對稱化
#   NBLAST: nblast_official_316.py 以官方實作算出 (與論文 D1/D2 同設定)
# 不再使用的舊檔:
#   FTmodel_predict.csv  來自另一組模型, 與 FineTune_miniLR 各折相關僅 0.69-0.74
#   NBLAST_316.npy       FB 區塊與官方實作幾乎不相關 (r = 0.06), 來源無法追溯
METHODS = {
    'FineTune_miniLR (fold 0)': 'FineTune_miniLR_D1-D6_0_316.npy',
    'Annotator (fold 0)': 'Annotator_D1-D6_0_316.npy',
    'NBLAST (official)': 'NBLAST_316_official.npy',
}
LINKAGES = ('average', 'complete', 'ward')


def _clean(s):
    return s.astype(str).str.strip().str.replace('﻿', '', regex=False)


def load_order():
    """316 顆 FC 神經的順序 = ALLN(45) + PN(100) + KC(71) + FB(100)。"""
    ids, num = [], []
    for l in LPU:
        d = pd.read_csv(os.path.join(F, l + '.txt'), header=None, names=['fc_id'])
        d['fc_id'] = _clean(d['fc_id'])
        d = d[d['fc_id'] != '']
        ids += d['fc_id'].tolist()
        num.append(len(d))
    return ids, num


def norm01(A):
    return (A - A.min()) / (A.max() - A.min())


def load_matrices(order):
    """回傳 {方法: 矩陣}。每個矩陣都先取兩個輸入方向的平均使其對稱, 再 min-max 正規化,
    確保比較條件相同。AUC 與 ARI 不受正規化影響; silhouette 與 within/between 平均值會。"""
    out = {}
    for name, fname in METHODS.items():
        X = np.load(os.path.join(F, fname)).astype(float)
        assert X.shape == (len(order), len(order)), f'{fname} 形狀不符: {X.shape}'
        out[name] = norm01((X + X.T) / 2)
    return out


def swc_laterality(fc_id):
    """以「纜長為質量」計算左右分布，與論文 pre-screening 的定義一致。
    回傳 x<0 側的纜長占比、纜長加權平均 x、soma 的 x。"""
    a = np.loadtxt(os.path.join(SWC_FC, fc_id + '.swc'), comments='#')
    if a.ndim == 1:
        a = a[None, :]
    node, typ = a[:, 0].astype(int), a[:, 1].astype(int)
    x, y, z, par = a[:, 2], a[:, 3], a[:, 4], a[:, 6].astype(int)
    pos = {ni: i for i, ni in enumerate(node)}
    has = par > 0
    ci = np.where(has)[0]
    pi = np.array([pos[q] for q in par[has]])
    seg = np.sqrt((x[ci] - x[pi])**2 + (y[ci] - y[pi])**2 + (z[ci] - z[pi])**2)
    mid_x = (x[ci] + x[pi]) / 2
    soma = np.where(typ == 1)[0]
    return dict(fc_id=fc_id, cable=seg.sum(),
                frac_neg=seg[mid_x < 0].sum() / seg.sum(),
                wmean_x=(seg * mid_x).sum() / seg.sum(),
                soma_x=x[soma[0]] if len(soma) else x[0])


def cut(X, k, method='average'):
    D = 1 - X
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    D[D < 0] = 0
    Z = sch.linkage(squareform(D, checks=False), method=method)
    return sch.fcluster(Z, k, criterion='maxclust'), D


def pair_auc(S, labels):
    """same-label pair 的分數是否高於 different-label pair；對單調變換不變。"""
    off = ~np.eye(len(labels), dtype=bool)
    same = labels[:, None] == labels[None, :]
    w, b = S[same & off], S[~same]
    auc = roc_auc_score(np.r_[np.ones(len(w)), np.zeros(len(b))], np.r_[w, b])
    return w.mean(), b.mean(), auc


def main():
    order, num = load_order()
    mats = load_matrices(order)
    lab = np.concatenate([[i] * k for i, k in enumerate(num)])
    rows = []

    print('=' * 72)
    print('A. 四個 neuropil 的整體分離度 (n=316)')
    for nm, X in mats.items():
        w, b, auc = pair_auc(X, lab)
        _, D = cut(X, 4)
        sil = silhouette_score(D, lab, metric='precomputed')
        rec = dict(analysis='4-neuropil', method=nm, within=w, between=b,
                   contrast=w - b, auc=auc, silhouette=sil)
        for lk in LINKAGES:
            cl, _ = cut(X, 4, lk)
            rec[f'ari_{lk}'] = adjusted_rand_score(lab, cl)
            rec[f'nmi_{lk}'] = normalized_mutual_info_score(lab, cl)
        print(f'  {nm:25s} within={w:.3f} between={b:.3f} AUC={auc:.3f} silhouette={sil:.3f} '
              + ' '.join(f'ARI[{lk[:3]}]={rec[f"ari_{lk}"]:.3f}' for lk in LINKAGES))
        rows.append(rec)

    print('\nB. 各類別兩兩分離度 (AUC: 同類 pair vs 跨類 pair)')
    for nm, X in mats.items():
        print(f'  {nm}:')
        for i, j in itertools.combinations(range(4), 2):
            sel = (lab == i) | (lab == j)
            w, b, auc = pair_auc(X[np.ix_(sel, sel)], lab[sel])
            print(f'    {LPU[i]:5s} vs {LPU[j]:5s}  within={w:.3f} cross={b:.3f} AUC={auc:.3f}')
            rows.append(dict(analysis=f'{LPU[i]}-vs-{LPU[j]}', method=nm,
                             within=w, between=b, contrast=w - b, auc=auc))

    print('\nC. 四類各自的左右腦分布 (SWC 纜長)')
    lat = pd.DataFrame([swc_laterality(f) for f in order])
    lat['group'] = np.array(LPU)[lab]
    lat.to_csv(os.path.join(F, 'laterality_316.csv'), index=False)
    for l in LPU:
        d = lat[lat.group == l]
        print(f'  {l:5s} n={len(d):3d}  x<0 側纜長占比 median={d.frac_neg.median():.3f} '
              f'[{d.frac_neg.min():.3f}, {d.frac_neg.max():.3f}]  '
              f'soma x<0: {int((d.soma_x < 0).sum())}/{len(d)}')

    print('\nD. FB 內部三塊結構 vs 解剖側化 (n=100)')
    fb = slice(216, 316)
    L = lat.iloc[fb].reset_index(drop=True)
    fn = L.frac_neg.values
    anat = np.where(fn > 0.65, 'L(x<0)', np.where(fn < 0.35, 'R(x>0)', 'bilateral'))
    print('  解剖標籤 (只由 SWC 決定, 與相似度無關):',
          pd.Series(anat).value_counts().to_dict())
    for nm, X in mats.items():
        Xf = X[fb, fb]
        cl, _ = cut(Xf, 3)
        ct = pd.crosstab(pd.Series(anat, name='anatomy'), pd.Series(cl, name='cluster'))
        ari = adjusted_rand_score(anat, cl)
        pur = ct.max(0).sum() / len(cl)
        print(f'\n  {nm}: ARI={ari:+.3f}  purity={pur:.3f}')
        print('   ' + ct.to_string().replace('\n', '\n   '))
        rows.append(dict(analysis='FB-3way', method=nm, ari_average=ari, purity=pur))
        # 只取側化明確的神經，做不依賴分群的檢定
        lz = anat != 'bilateral'
        w, b, auc = pair_auc(Xf[np.ix_(lz, lz)], anat[lz])
        print(f'   側化明確的 {lz.sum()} 顆: same-side={w:.3f} opposite={b:.3f} AUC={auc:.3f}')
        rows.append(dict(analysis='FB-same-vs-opposite-side', method=nm,
                         within=w, between=b, contrast=w - b, auc=auc))

    res = pd.DataFrame(rows)
    print('\nE. 總表 (AUC / ARI 不受分數尺度影響)')
    auc_tab = res.pivot(index='analysis', columns='method', values='auc')
    order_rows = ['4-neuropil'] + [f'{LPU[i]}-vs-{LPU[j]}' for i, j in itertools.combinations(range(4), 2)] \
        + ['FB-same-vs-opposite-side']
    print('  AUC:')
    print('   ' + auc_tab.loc[order_rows, list(METHODS)].round(3).to_string().replace('\n', '\n   '))
    ari_tab = res[res.analysis.isin(['4-neuropil', 'FB-3way'])].pivot(
        index='analysis', columns='method', values='ari_average')
    print('  ARI (average linkage):')
    print('   ' + ari_tab[list(METHODS)].round(3).to_string().replace('\n', '\n   '))

    out = os.path.join(F, 'cluster_quantification.csv')
    res.to_csv(out, index=False)
    print(f'\n已寫出: {out}')
    print(f'已寫出: {os.path.join(F, "laterality_316.csv")}')


if __name__ == '__main__':
    main()
