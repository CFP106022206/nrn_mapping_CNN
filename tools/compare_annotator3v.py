"""比較修正視角切片後重訓的 annotator（Annotator3v）與舊 annotator。

背景：`model.MVCNN_Siamese` 的視角切片有閉包延遲綁定錯誤，舊 annotator 只讀到
三視圖中的第 3 張。`train_annotator_3view.py` 用 `MVCNN_Siamese_3View` 重訓，
除了視角切片以外的設定都與舊 annotator 相同。見 MODEL_PIPELINE_HANDOVER.md §13.0。

每一折做三件事：
1. **權重載入驗證**：用 `MVCNN_Siamese_3View` 載入新權重重新預測，
   與訓練腳本當時存下的 `result/test_label_Annotator3v_D1-D6_{fold}.csv` 比對，應逐筆相同。
2. **確認真的用到三個視角**：新模型 6 個逐視角 BN 的 moving 統計量在三個視角之間應該不同
   （舊模型逐位元相同）。
3. **守門 AUC**：同一批測試配對上，舊 vs 新，配對 bootstrap 算差距的 95 % CI。
   fold 測試集只有約 122 對，單折 CI 很寬，要看多折合併。

在 CPU 上跑，不和 GPU 上的訓練搶資源。
執行：CUDA_VISIBLE_DEVICES="" python3 tools/compare_annotator3v.py [fold ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from data_process_fineTune import make_numpy_from_standard_views  # noqa: E402
from model import MVCNN_Siamese, MVCNN_Siamese_3View  # noqa: E402

OLD = PROJ / "Annotator_Model" / "Annotator_D1-D6_{fold}.weights.h5"
NEW = PROJ / "Annotator_Model" / "Annotator3v_D1-D6_{fold}.weights.h5"
TRAIN_CSV = PROJ / "result" / "test_label_Annotator3v_D1-D6_{fold}.csv"


def view_bn_spread(net) -> float:
    """FC 三個視角的 BN moving_mean/moving_var 之間的最大差（舊模型為 0）。"""
    bns = [l for l in net.layers if l.__class__.__name__ == "BatchNormalization"][:3]
    ms = [np.concatenate([l.get_weights()[2].ravel(), l.get_weights()[3].ravel()]) for l in bns]
    return max(float(np.abs(ms[0] - ms[1]).max()), float(np.abs(ms[0] - ms[2]).max()))


def paired_boot(y, a, b, n=2000, seed=5):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


def main() -> None:
    folds = [int(f) for f in sys.argv[1:]] or [
        f for f in range(10) if Path(str(NEW).format(fold=f)).exists()]
    if not folds:
        print("還沒有任何一折訓練完成（找不到 result/test_label_Annotator3v_*.csv）")
        return

    rows, pooled = [], []
    for fold in folds:
        te = pd.read_csv(PROJ / "train_test_split" / f"test_split_{fold}_D1-D6.csv")
        x, pairs, miss, _ = make_numpy_from_standard_views(
            te[["fc_id", "em_id", "label"]],
            fc_dir=str(PROJ / "data/standard_views/FC"),
            em_dir=str(PROJ / "data/standard_views/EM"))
        inp = {"FC": x[:, 0], "EM": x[:, 1]}
        y = (pairs.label.to_numpy() >= 0.5).astype(int)

        old = MVCNN_Siamese((50, 50, 3))
        old.load_weights(str(OLD).format(fold=fold))
        new = MVCNN_Siamese_3View((50, 50, 3))
        new.load_weights(str(NEW).format(fold=fold))
        po = old.predict(inp, verbose=0).ravel()
        pn = new.predict(inp, verbose=0).ravel()

        # 訓練中途被中斷的折沒有這個 CSV（它在訓練結束時才寫），此時跳過載入驗證
        tcsv = Path(str(TRAIN_CSV).format(fold=fold))
        if tcsv.exists():
            tr = pd.read_csv(tcsv)
            key = lambda d: d.assign(fc_id=d.fc_id.astype(str), em_id=d.em_id.astype("int64"))
            m = key(pairs[["fc_id", "em_id"]]).assign(pn=pn).merge(
                key(tr[["fc_id", "em_id", "model_pred"]]), on=["fc_id", "em_id"], how="left")
            assert m.model_pred.notna().all(), "訓練時的預測對不上測試配對"
            reload_diff = float(np.abs(m.pn - m.model_pred).max())
        else:
            reload_diff = float("nan")

        a_old, a_new = roc_auc_score(y, po), roc_auc_score(y, pn)
        dm, lo, hi, p = paired_boot(y, po, pn)
        rows.append({"fold": fold, "n": len(y), "n_pos": int(y.sum()),
                     "auc_old_1view": a_old, "auc_new_3view": a_new,
                     "diff": dm, "diff_lo": lo, "diff_hi": hi, "p_new_better": p,
                     "bn_view_spread_old": view_bn_spread(old),
                     "bn_view_spread_new": view_bn_spread(new),
                     "reload_max_diff": reload_diff})
        pooled.append(pd.DataFrame({"fold": fold, "y": y, "old": po, "new": pn}))
        print(f"fold {fold}：n={len(y)}（正 {int(y.sum())}）  舊 {a_old:.4f} → 新 {a_new:.4f}  "
              f"差 {dm:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(新較好)={p:.3f}  |  "
              f"BN 三視角差 舊 {view_bn_spread(old):.3g} / 新 {view_bn_spread(new):.3g}  |  "
              f"重載 vs 訓練時預測 最大差 {reload_diff:.3g}")

    res = pd.DataFrame(rows)
    if len(folds) > 1:
        P = pd.concat(pooled)
        y, po, pn = P.y.to_numpy(), P.old.to_numpy(), P.new.to_numpy()
        dm, lo, hi, p = paired_boot(y, po, pn)
        print(f"\n合併 {len(folds)} 折（n={len(y)}）：舊 {roc_auc_score(y, po):.4f} → "
              f"新 {roc_auc_score(y, pn):.4f}  差 {dm:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(新較好)={p:.3f}")
        print(f"逐折平均：舊 {res.auc_old_1view.mean():.4f} → 新 {res.auc_new_3view.mean():.4f}")

    out = PROJ / "result" / "compare_annotator3v.csv"
    res.to_csv(out, index=False)
    print(f"-> {out.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
