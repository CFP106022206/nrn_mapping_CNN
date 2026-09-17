# %% 用單一 fine-tune 模型重算 similarity matrix 的 316 顆 FC 神經兩兩分數
# 前處理與模型載入直接沿用專案根目錄的 Model_predict.py / swc_util.py:
#   每對神經先 _pad_to_same_size 補到同尺寸, 再 _resize_to_50, /255 後送入 MVCNN_Siamese
# 模型的分類頭是把兩側特徵串接, 分數與輸入順序有關, 因此 316x316 的有序配對全部都算
# (row 神經進 "FC" 輸入, col 神經進 "EM" 輸入); 下游使用時再取 (M + M.T) / 2。
#
# 執行 (需要 tensorflow / keras, 專案的 ming 環境):
#   python similarity_matrix/model_predict_316.py                 # 預設 FineTune_miniLR_D1-D6_ 第 0 折
#   python similarity_matrix/model_predict_316.py --fold 3
#   python similarity_matrix/model_predict_316.py --check-only    # 只做重現性檢查
#
# 重現性檢查: 以同樣流程重算 result/test_label_{prefix}{fold}.csv 的 FC-EM 測試配對,
#            與訓練時存下的 model_pred 比對, 確認前處理與權重載入接對了。
#
# 輸出 (列/欄順序為 ALLN.txt, PN.txt, KC.txt, FB.txt 串接):
#   {prefix}{fold}_316.npy   316x316 有序分數 (未對稱化)
#   {prefix}{fold}_316.csv   長表 fc_id, em_id, model_predict
#                            (與 FTmodel_predict.csv 相同慣例: em_id 實際上也是 fc_id)
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from Model_predict import Config, load_model  # noqa: E402
from swc_util import _load_views_from_npz, _pad_to_same_size, _resize_to_50  # noqa: E402

LPU = ['ALLN', 'PN', 'KC', 'FB']
VIEWS = {'FC': os.path.join(ROOT, 'data', 'standard_views', 'FC'),
         'EM': os.path.join(ROOT, 'data', 'standard_views', 'EM')}


def load_order():
    ids = []
    for l in LPU:
        d = pd.read_csv(os.path.join(HERE, l + '.txt'), header=None, names=['fc_id'])
        d['fc_id'] = d['fc_id'].astype(str).str.strip().str.replace('﻿', '', regex=False)
        ids += d.loc[d['fc_id'] != '', 'fc_id'].tolist()
    return ids


class PairPreprocessor:
    """逐對呼叫 Model_predict.py 用的 _pad_to_same_size + _resize_to_50。

    補齊後的影像只取決於 (神經, 補齊目標尺寸), 以此為 key 快取, 結果與逐對重算相同。
    """

    def __init__(self, out_hw):
        self.out_hw = out_hw
        self.raw = {}
        self.cache = {}

    def views(self, nid, source):
        key = (source, nid)
        if key not in self.raw:
            self.raw[key] = _load_views_from_npz(os.path.join(VIEWS[source], f'{nid}_views.npz'))
        return self.raw[key]

    def pair(self, a, a_src, b, b_src):
        va, vb = self.views(a, a_src), self.views(b, b_src)
        target = max(va.shape[-1], vb.shape[-1])
        ka, kb = (a_src, a, target), (b_src, b, target)
        if ka not in self.cache or kb not in self.cache:
            pa, pb = _pad_to_same_size(va, vb)
            self.cache[ka] = _resize_to_50(pa, self.out_hw)
            self.cache[kb] = _resize_to_50(pb, self.out_hw)
        return self.cache[ka], self.cache[kb]


def predict_pairs(model, prep, pairs, batch):
    """pairs: [(a_id, a_src, b_id, b_src)] -> 分數 (N,)。a 進 "FC" 輸入, b 進 "EM" 輸入。"""
    out = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        a_img, b_img = zip(*(prep.pair(*p) for p in chunk))
        fc = np.transpose(np.stack(a_img), (0, 2, 3, 1)).astype(np.float32) / 255.0
        em = np.transpose(np.stack(b_img), (0, 2, 3, 1)).astype(np.float32) / 255.0
        out[s:s + len(chunk)] = model.predict({'FC': fc, 'EM': em}, batch_size=1024, verbose=0).reshape(-1)
    return out


def check_reproduction(model, prep, prefix, fold, batch):
    ref_path = os.path.join(ROOT, 'result', f'test_label_{prefix}{fold}.csv')
    if not os.path.exists(ref_path):
        print(f'[check] 找不到 {ref_path}, 略過重現性檢查')
        return
    ref = pd.read_csv(ref_path, dtype={'fc_id': str, 'em_id': str})
    pairs = [(r.fc_id, 'FC', r.em_id, 'EM') for r in ref.itertuples()]
    got = predict_pairs(model, prep, pairs, batch)
    diff = np.abs(got - ref['model_pred'].to_numpy(dtype=np.float32))
    r = np.corrcoef(got, ref['model_pred'])[0, 1]
    print(f'[check] 重算 {len(ref)} 組 fold-{fold} 測試配對: max|差|={diff.max():.2e}  '
          f'median|差|={np.median(diff):.2e}  r={r:.6f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='FineTune_miniLR_D1-D6_', help='權重檔名前綴 (不含折號)')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--model-dir', default=os.path.join(ROOT, 'FineTune_Model'))
    ap.add_argument('--batch', type=int, default=20000, help='每批組裝的配對數 (控制記憶體)')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()

    cfg = Config(model_dir=args.model_dir, model_prefix=args.prefix)
    out_hw = tuple(int(x) for x in cfg.out_hw)
    model = load_model(cfg, args.fold, input_size=(*out_hw, 3))
    print(f'模型: {args.model_dir}/{args.prefix}{args.fold}.weights.h5', flush=True)
    prep = PairPreprocessor(out_hw)

    check_reproduction(model, prep, args.prefix, args.fold, args.batch)
    if args.check_only:
        return

    order = load_order()
    assert len(order) == 316 and len(set(order)) == 316, '名單應為 316 顆不重複的 FC 神經'
    t0 = time.time()
    pairs = [(a, 'FC', b, 'FC') for a in order for b in order]
    scores = predict_pairs(model, prep, pairs, args.batch)
    m = scores.reshape(len(order), len(order)).astype(float)

    stem = os.path.join(HERE, f'{args.prefix}{args.fold}_316')
    np.save(stem + '.npy', m)
    pd.DataFrame({'fc_id': [p[0] for p in pairs], 'em_id': [p[2] for p in pairs],
                  'model_predict': scores}).to_csv(stem + '.csv', index=False)
    print(f'完成 {len(pairs)} 組 ({time.time()-t0:.0f}s)  範圍 [{m.min():.4f}, {m.max():.4f}]  '
          f'max|M-M.T|={np.abs(m - m.T).max():.3f}  對角線中位 {np.median(np.diag(m)):.4f}')
    print(f'已寫出: {stem}.npy, {stem}.csv')


if __name__ == '__main__':
    main()
