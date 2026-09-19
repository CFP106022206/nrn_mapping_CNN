"""Annotator 基線訓練（v2）。

這是 2026-09-18/19 那輪調查之後的乾淨版本，把當時用開關逐一驗證過的設定直接寫死，
不再保留實驗用的參數開關。每一項選擇的依據見 MVCNN_VIEW_BUG_INVESTIGATION.md。

固定的設計決定
--------------
* **`MVCNN_Siamese_3View`**：修正了閉包延遲綁定，三個視角各自讀對應的圖層。
  舊的 `MVCNN_Siamese` 三個視角都只讀第 3 張。
* **`share_view_norm=True`**：trunk 只有**一個**正規化層，所有視角、兩個分支全部共用。
  這是忠於原始 MVCNN（Su et al. 2015）的設計——pooling 之前沒有任何 per-view 參數。
  依據：訓練後三個逐視角 BN 的 moving_mean / moving_var 在視角之間實質相同
  （視角間 std / 整體 std = 0.017），逐視角正規化並沒有在做事。
* **`pool_type="max"`**：配上共用正規化層，模型對視角順序**精確不變**（實測交換後差 0）。
* **`sym_merge=True`**：對稱合併 head，使推論時 f(FC,EM) 與 f(EM,FC) 逐位元相同。
* **augmentation = D4 的 8 個 in-plane 操作（統一施加到三個視角）× FC/EM 交換 = 每對 16 筆**

  為什麼是這一組：三張視圖是**標準腦座標**下固定的 YZ / XZ / XY 投影
  （`standard_draw.py`「使用標準腦座標畫圖，不旋轉」），**不是主軸座標系**。
  所以沒有特徵向量符號歧義、也沒有軸序歧義可言，augmentation 的作用回到單純的正則化。
  又因為模型精確置換不變，跨視角的配置對它不可見（實測：三視角獨立抽操作
  與統一施加，對輸出的擾動幅度比 1.02x），所以只需要最大化**每張圖的 in-plane 變換分布**。
  D4 是方格上無損的完整對稱群（4 旋轉 × 2 翻轉），比 legacy 的 5 個多了
  rot90+flip / rot180+flip / rot270+flip。
* **保留 FC/EM 交換**：推論時對稱（差 0），但**訓練時不對稱**（實測差 5.46e-01）——
  共用的 BN 在圖中被呼叫 6 次，每次各自從自己的輸入算 batch 統計量，
  而 FC 與 EM 的影像分布不同。所以它不是冗餘的。
* **樣本層洗牌**：擴增後必須再洗一次，否則一個 batch 會剛好是同一對的 16 個擴增，
  批內標籤全同會讓 BatchNorm 洩漏標籤（見 `make_dataset`）。
* **lr 1e-5 / 300 epochs / batch 16**：lr 掃描顯示 1e-5 已接近最佳；
  batch 維持 16（固定 lr 下放大 batch 會減少梯度噪音與 BN 統計噪音這兩層隱性正則化）。
* **兩種 checkpoint 都存**：val_loss 最低與 val AUC 最高各一份，零成本對照。

執行：python3 data_process_train_v2.py <fold>
"""

from __future__ import annotations

import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from gpu_config import enable_gpu_memory_growth

enable_gpu_memory_growth()

from data_process_fineTune import make_numpy_from_standard_views  # noqa: E402
from model import MVCNN_Siamese_3View  # noqa: E402

# D4：方格上無損的 8 個 in-plane 操作 (rot90 次數, 是否左右翻)。
# 同一個操作施加到三個視角，不改變視角順序。
D4_OPS = tuple((k, f) for k in range(4) for f in (False, True))
AUG_PER_PAIR = 2 * len(D4_OPS)   # x2 = FC/EM 交換


@dataclass(frozen=True)
class Config:
    fold: int
    split_dir: str = "./train_test_split"
    split_suffix: str = "D1-D6"
    fc_dir: str = "./data/standard_views/FC"
    em_dir: str = "./data/standard_views/EM"
    model_dir: str = "./Baseline_Model"
    result_dir: str = "./result"
    model_name: str = "Baseline"
    seed: int = 42
    val_ratio: float = 0.15
    batch_size: int = 16
    initial_lr: float = 1e-5
    train_epochs: int = 300

    @property
    def stem(self) -> str:
        return f"{self.model_name}_{self.split_suffix}_{self.fold}"


class BinarizedAUC(tf.keras.metrics.AUC):
    """soft label 依 >= 0.5 二值化後再算，與評估時的 roc_auc_score 定義一致。"""

    def __init__(self, *a, num_thresholds=1000, **kw):
        super().__init__(*a, num_thresholds=num_thresholds, **kw)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.cast(y_true >= 0.5, y_pred.dtype), y_pred, sample_weight)


class BinarizedAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.cast(y_true >= 0.5, y_pred.dtype), y_pred, sample_weight)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def apply_d4(img, k: int, flip: bool):
    """先翻再轉，對三個視角一起做（channel 維不動）。"""
    if flip:
        img = tf.image.flip_left_right(img)
    return tf.image.rot90(img, k) if k else img


def make_dataset(x, y, cfg: Config, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 4096), seed=cfg.seed, reshuffle_each_iteration=True)

        def expand(pair, label):
            fc, em = pair[0], pair[1]
            a, b = [], []
            for u, v in ((fc, em), (em, fc)):      # FC/EM 交換：訓練時 BN 不對稱，非冗餘
                for k, flip in D4_OPS:
                    a.append(apply_d4(u, k, flip))
                    b.append(apply_d4(v, k, flip))
            return tf.data.Dataset.from_tensor_slices((
                {"FC": tf.stack(a), "EM": tf.stack(b)},
                tf.stack([label] * AUG_PER_PAIR)))

        ds = ds.flat_map(expand)
        # ⚠️ 必須在 flat_map 之後再洗一次。expand 每對連續吐出 AUG_PER_PAIR 筆，
        # 而 AUG_PER_PAIR == batch_size == 16，只在 pair 層洗牌的話每個 batch 恰好
        # 是同一對的 16 個擴增、批內標籤全同 —— BatchNorm 的 batch 統計量就直接
        # 洩漏標籤。實測後果：訓練時 AUC 0.9987（走捷徑），同一份權重在推論模式下
        # 連訓練集都只有 0.8675，測試 AUC 0.59。buffer 2048 = 128 對，足夠打散。
        ds = ds.shuffle(2048, seed=cfg.seed + 1, reshuffle_each_iteration=True)
        n = len(x) * AUG_PER_PAIR
    else:
        ds = ds.map(lambda p, l: ({"FC": p[0], "EM": p[1]}, l),
                    num_parallel_calls=tf.data.AUTOTUNE)
        n = len(x)
    ds = ds.batch(cfg.batch_size)
    # flat_map 之後 cardinality 未知，Keras 3 會誤判成 one-shot iterator
    ds = ds.apply(tf.data.experimental.assert_cardinality(
        int(math.ceil(n / cfg.batch_size))))
    return ds.prefetch(tf.data.AUTOTUNE)


def report(y_true, y_prob) -> dict:
    yb = (np.asarray(y_true) >= 0.5).astype(int)
    pb = (np.asarray(y_prob).reshape(-1) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(f1_score(yb, pb, pos_label=1)),
            "auc": float(roc_auc_score(yb, np.asarray(y_prob).reshape(-1)))}


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)

    sd = Path(cfg.split_dir)
    tr = pd.read_csv(sd / f"train_split_{cfg.fold}_{cfg.split_suffix}.csv")
    te = pd.read_csv(sd / f"test_split_{cfg.fold}_{cfg.split_suffix}.csv")
    cols = ["fc_id", "em_id", "label"]
    x_all, pair_all, miss_tr, _ = make_numpy_from_standard_views(
        tr[cols], fc_dir=cfg.fc_dir, em_dir=cfg.em_dir)
    x_test, pair_test, miss_te, _ = make_numpy_from_standard_views(
        te[cols], fc_dir=cfg.fc_dir, em_dir=cfg.em_dir)
    print(f"未找到 train {len(miss_tr)} / test {len(miss_te)}")

    x_tr, x_val, p_tr, p_val = train_test_split(
        x_all, pair_all, test_size=cfg.val_ratio, random_state=cfg.seed)
    y_tr = p_tr["label"].to_numpy(np.float32)
    y_val = p_val["label"].to_numpy(np.float32)
    y_test = pair_test["label"].to_numpy(np.float32)
    print(f"Train {len(x_tr)} x{AUG_PER_PAIR} 擴增 | Val {len(x_val)} | Test {len(x_test)}")

    model = MVCNN_Siamese_3View(x_tr.shape[2:], pool_type="max", trunk_norm="bn",
                                share_view_norm=True, sym_merge=True)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg.initial_lr),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
                  metrics=[BinarizedAccuracy(name="bi_acc"), BinarizedAUC(name="auc")])
    print(f"參數量 {model.count_params():,}")

    md = Path(cfg.model_dir)
    cbs = [tf.keras.callbacks.ModelCheckpoint(
               str(md / f"{cfg.stem}_bestloss.weights.h5"), monitor="val_loss",
               save_best_only=True, save_weights_only=True, mode="min", verbose=0),
           tf.keras.callbacks.ModelCheckpoint(
               str(md / f"{cfg.stem}_bestauc.weights.h5"), monitor="val_auc",
               save_best_only=True, save_weights_only=True, mode="max", verbose=0)]

    hist = model.fit(make_dataset(x_tr, y_tr, cfg, True),
                     validation_data=make_dataset(x_val, y_val, cfg, False),
                     epochs=cfg.train_epochs, callbacks=cbs, verbose=2)

    rd = Path(cfg.result_dir)
    with open(rd / f"Train_History_{cfg.stem}.pkl", "wb") as f:
        pickle.dump(hist.history, f)

    # 兩種 checkpoint 各自在測試集上評估
    summary = {}
    for tag in ("bestloss", "bestauc"):
        model.load_weights(str(md / f"{cfg.stem}_{tag}.weights.h5"))
        pred = model.predict(make_dataset(x_test, y_test, cfg, False), verbose=0).ravel()
        summary[tag] = report(y_test, pred)
        out = pair_test[["fc_id", "em_id", "label"]].copy()
        out["model_pred"] = pred
        out.to_csv(rd / f"test_label_{cfg.stem}_{tag}.csv", index=False)
        print(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in summary[tag].items()))
    with open(rd / f"Test_Result_{cfg.stem}.pkl", "wb") as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main(Config(fold=int(sys.argv[1])))
