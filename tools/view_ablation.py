"""每一張視圖各自貢獻多少鑑別力？順便核對新舊模型的架構差異。

做兩種消融（都在專家測試集上算 AUC）：
  * 去掉一張：把該視角的影像清零，其餘兩張保留。
  * 只留一張：其餘兩張清零。
舊 annotator 只讀得到第 3 張，所以它應該只有「第 3 張」那幾欄會變。

⚠️ 清零是分布外的輸入（模型沒看過全黑的視角），數字只能當相對強弱的參考，
不能當成「移除該視角重新訓練」的結果。

在 CPU 上跑：CUDA_VISIBLE_DEVICES="" python3 tools/view_ablation.py [fold ...]
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


def arch_diff() -> None:
    """核對兩個模型函數的層組成——除了視角切片以外應該完全相同。"""
    a, b = MVCNN_Siamese((50, 50, 3)), MVCNN_Siamese_3View((50, 50, 3))
    ca = pd.Series([l.__class__.__name__ for l in a.layers]).value_counts()
    cb = pd.Series([l.__class__.__name__ for l in b.layers]).value_counts()
    t = pd.concat([ca, cb], axis=1, keys=["MVCNN_Siamese(舊)", "MVCNN_Siamese_3View(新)"]).fillna(0).astype(int)
    print("層組成：")
    print(t.to_string())
    print(f"參數量：舊 {a.count_params():,} / 新 {b.count_params():,}")
    print(f"有沒有 LayerNormalization：舊 {'LayerNormalization' in ca.index} / "
          f"新 {'LayerNormalization' in cb.index}\n")


def main() -> None:
    arch_diff()
    folds = [int(f) for f in sys.argv[1:]] or [
        f for f in range(10) if Path(str(NEW).format(fold=f)).exists()]
    rows = []
    for fold in folds:
        te = pd.read_csv(PROJ / "train_test_split" / f"test_split_{fold}_D1-D6.csv")
        x, pairs, _, _ = make_numpy_from_standard_views(
            te[["fc_id", "em_id", "label"]],
            fc_dir=str(PROJ / "data/standard_views/FC"),
            em_dir=str(PROJ / "data/standard_views/EM"))
        fc, em = x[:, 0], x[:, 1]
        y = (pairs.label.to_numpy() >= 0.5).astype(int)

        def masked(keep: set[int]) -> dict:
            f, e = fc.copy(), em.copy()
            for c in range(3):
                if c not in keep:
                    f[..., c] = 0.0
                    e[..., c] = 0.0
            return {"FC": f, "EM": e}

        for tag, path, builder in (("舊(單視角)", OLD, MVCNN_Siamese),
                                   ("新(三視角)", NEW, MVCNN_Siamese_3View)):
            m = builder((50, 50, 3))
            m.load_weights(str(path).format(fold=fold))
            r = {"fold": fold, "model": tag,
                 "全部": roc_auc_score(y, m.predict(masked({0, 1, 2}), verbose=0).ravel())}
            for c in range(3):
                r[f"去掉第{c+1}張"] = roc_auc_score(
                    y, m.predict(masked({0, 1, 2} - {c}), verbose=0).ravel())
                r[f"只留第{c+1}張"] = roc_auc_score(
                    y, m.predict(masked({c}), verbose=0).ravel())
            rows.append(r)

    res = pd.DataFrame(rows)
    cols = ["fold", "model", "全部"] + [f"去掉第{c}張" for c in (1, 2, 3)] + \
           [f"只留第{c}張" for c in (1, 2, 3)]
    res = res[cols]
    print(res.round(4).to_string(index=False))
    print("\n各折平均：")
    print(res.groupby("model")[cols[2:]].mean().round(4).to_string())
    out = PROJ / "result" / "view_ablation.csv"
    res.to_csv(out, index=False)
    print(f"\n-> {out.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
