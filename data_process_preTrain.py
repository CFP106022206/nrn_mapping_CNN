"""Pre-train (pseudo-label) data pipeline + training.

This file intentionally follows the same structure/style as `Data_process_Train.py`,
but differs in I/O:
  - Input pairs/labels come from pseudo-label CSV: ./data/pairs_label/EMxFC_all_high_confidence.csv
  - We train a fresh model from scratch.
  - Default: only train/val split (no test split) since labels are pseudo.
"""

# %%
from __future__ import annotations

import os
import random
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

import keras

keras.config.enable_unsafe_deserialization()  # 关闭 keras safe mode

from model import MVCNN_Siamese
from swc_util import make_numpy_from_standard_views


@dataclass(frozen=True)
class Config:
    seed: int = 3407

    # I/O
    label_csv: str = "./data/pairs_label/EMxFC_all_high_confidence.csv"
    fc_views_dir: str = "./data/standard_views/FC"
    em_views_dir: str = "./data/standard_views/EM"

    save_model_dir: str = "./PreTrain_Model"
    save_result_dir: str = "./result"
    fig_dir: str = "./Figure"

    save_model_name: str = "pre_train_model_by_EMxFC_180K"

    # train
    initial_lr: float = 1e-5
    train_epochs: int = 300
    batch_size: int = 16
    val_ratio: float = 0.1

    scheduler_exp: float = 0.0
    min_lr: float = 1e-7

    out_hw: tuple[int, int] = (50, 50)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_tf_dataset(x, y, batch_size, training, seed):
    """Dataset builder.

    Pre-train stage augmentation policy:
        - keep ONLY swap (exchange FC/EM inputs) to enforce symmetry
        - remove rotation / flip
    """

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(x), 4096),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def augment(pair, label):
        fc = pair[0]
        em = pair[1]

        # two samples: original and swapped
        fc_stack = tf.stack([fc, em], axis=0)
        em_stack = tf.stack([em, fc], axis=0)
        label_stack = tf.stack([label, label], axis=0)

        return tf.data.Dataset.from_tensor_slices(
            ({"FC": fc_stack, "EM": em_stack}, label_stack)
        )

    if training:
        ds = ds.flat_map(augment)
    else:
        ds = ds.map(
            lambda pair, label: ({"FC": pair[0], "EM": pair[1]}, label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def scheduler_factory(cfg: Config):
    def scheduler(epoch, lr):
        if cfg.scheduler_exp <= 0:
            return lr
        total = cfg.train_epochs
        new_lr = lr * ((1 - epoch / total) ** cfg.scheduler_exp)
        return max(new_lr, cfg.min_lr)

    return scheduler


def metrics_report(y_true, y_prob, threshold=0.5):
    y_true_bin = (np.array(y_true) > threshold).astype(int)
    y_pred_bin = (np.array(y_prob).reshape(-1) > threshold).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = f1_score(y_true_bin, y_pred_bin, pos_label=1)

    return {
        "cm": cm,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_pos": float(f1_pos),
    }, y_pred_bin


def _load_pseudo_label_pairs(cfg: Config) -> pd.DataFrame:
    label_csv = Path(cfg.label_csv)
    if not label_csv.exists():
        raise FileNotFoundError(f"Pseudo-label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    required = {"fc_id", "em_id", "label"}
    if not required.issubset(set(df.columns)):
        raise KeyError(f"Pseudo-label CSV must contain {sorted(required)}. got={list(df.columns)}")

    df = df[["fc_id", "em_id", "label"]].copy()
    df["fc_id"] = df["fc_id"].astype(str).str.strip()
    df["em_id"] = df["em_id"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    # avoid merge explosion if duplicated pairs exist
    df = df.drop_duplicates(["fc_id", "em_id"], keep="first")
    return df


def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    Path(cfg.save_model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.save_result_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.fig_dir).mkdir(parents=True, exist_ok=True)

    save_model_name = cfg.save_model_name

    label_df = _load_pseudo_label_pairs(cfg)
    x_all, pair_all, not_found, resolutions = make_numpy_from_standard_views(
        label_df,
        fc_dir=cfg.fc_views_dir,
        em_dir=cfg.em_views_dir,
        out_hw=cfg.out_hw,
    )

    print("Not found pairs:", len(not_found))

    # train/val split only (pseudo-label stage)
    x_train, x_val, pair_train, pair_val = train_test_split(
        x_all, pair_all, test_size=cfg.val_ratio, random_state=cfg.seed
    )
    y_train = pair_train["label"].to_numpy(dtype=np.float32)
    y_val = pair_val["label"].to_numpy(dtype=np.float32)
    print(f"Train {len(x_train)} | Val {len(x_val)}")

    ds_train = make_tf_dataset(x_train, y_train, cfg.batch_size, training=True, seed=cfg.seed)
    ds_val = make_tf_dataset(x_val, y_val, cfg.batch_size, training=False, seed=cfg.seed)

    _, H, W = resolutions
    model = MVCNN_Siamese((H, W, resolutions[0]))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg.initial_lr),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bi_acc")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(cfg.save_model_dir) / f"{save_model_name}.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1,
        )
    ]
    if cfg.scheduler_exp > 0:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler_factory(cfg), verbose=1))

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.train_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    with open(Path(cfg.save_result_dir) / f"PreTrain_History_{save_model_name}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # reload best and report on val
    best = MVCNN_Siamese((H, W, resolutions[0]))
    best.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg.initial_lr),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bi_acc")],
    )
    best.load_weights(Path(cfg.save_model_dir) / f"{save_model_name}.weights.h5")

    y_val_pred = best.predict(ds_val, verbose=0)
    val_report, val_pred_bin = metrics_report(y_val, y_val_pred)
    print("Val:", val_report)

    with open(Path(cfg.save_result_dir) / f"PreTrain_Val_Result_{save_model_name}.pkl", "wb") as f:
        pickle.dump(val_report, f)

    pred_df = pair_val.copy()
    pred_df["model_pred"] = y_val_pred.reshape(-1)
    pred_df["model_pred_binary"] = val_pred_bin
    pred_df.to_csv(Path(cfg.save_result_dir) / f"pretrain_val_pred_{save_model_name}.csv", index=False)


if __name__ == "__main__":
    main(Config())

