'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
5, Iterative self-labeling
6, Transfer Big Model...
'''
# %%
from __future__ import annotations

import os
import random
import pickle
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

import keras
keras.config.enable_unsafe_deserialization() # 关闭keras safe mode

from util import load_pkl
from model import MVCNN_Siamese
from typing import Dict, Tuple, List

@dataclass(frozen=True)
class Config:
    fold: int
    seed: int = 3407

    used_split_suffix: str = "D1-D6"
    split_dir: str = "./train_test_split"

    # ------(Choose 1/2) Annotator model / FineTune model -------
    # use_pretrain_model = False
    # pretrain_model = None
    # save_model_dir: str = "./Annotator_Model"
    # model_name = "Annotator"

    # Finetune model
    use_pretrain_model = True
    pretrain_model: str = "./PreTrain_Model/pre_train_model_by_EMxFC_120K.weights.h5"
    save_model_dir: str = "./FineTune_Model"
    model_name = "FineTune_miniLR"
    # ------------------------------------------------------------

    save_result_dir: str = "./result"
    fig_dir: str = "./Figure"

    initial_lr: float = 1e-8    #Annotator use 1e-5, finetune use 1e-6
    train_epochs: int = 100     #Annotator use 300, finetune use 100
    batch_size: int = 16

    val_ratio: float = 0.15

    scheduler_exp: float = 0.0  # 0 means off
    min_lr: float = 1e-10


def _to_uint8_views(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    if v.dtype == np.uint8:
        return v
    vf = v.astype(np.float32, copy=False)
    vmax = float(np.nanmax(vf)) if vf.size else 0.0
    # 常见：如果是 0~1 浮点，就转 0~255
    if vmax <= 1.0:
        vf = np.round(vf * 255.0)
    vf = np.clip(vf, 0.0, 255.0)
    return vf.astype(np.uint8)


def _ensure_3hw_views(v: np.ndarray) -> np.ndarray:
    """Normalize view array to shape (3,H,W)."""
    v = np.asarray(v)
    if v.ndim != 3:
        raise ValueError(f"Expect 3D views, got shape={v.shape}")

    # standard_draw: (3,H,W)
    if v.shape[0] == 3:
        return v

    # sometimes: (H,W,3)
    if v.shape[-1] == 3:
        return np.transpose(v, (2, 0, 1))

    raise ValueError(f"Cannot interpret views shape as (3,H,W): {v.shape}")


def _pad_to_same_size(fc_views: np.ndarray, em_views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把较小的一侧用黑边补齐到较大的一侧尺寸（居中）。输入/输出都是 (3,H,W)。"""
    fc = _ensure_3hw_views(_to_uint8_views(fc_views))
    em = _ensure_3hw_views(_to_uint8_views(em_views))

    target = max(int(fc.shape[1]), int(em.shape[1]))

    def _pad(v: np.ndarray) -> np.ndarray:
        _, h, w = v.shape
        if h != w:
            raise ValueError(f"Expect square views (H==W). got {(h, w)}")
        pad = target - h
        if pad < 0:
            raise ValueError(f"target smaller than current: current=({h},{w}) target=({target},{target})")
        top = pad // 2
        bottom = pad - top
        left = pad // 2
        right = pad - left
        return np.pad(v, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)

    return _pad(fc), _pad(em)


def _resize_to_50(views: np.ndarray, out_hw: tuple[int, int] = (50, 50)) -> np.ndarray:
    """把 (3,H,W) 下采样到 (3,out_h,out_w)。

    使用“自适应 max pooling”（每个输出像素取对应输入块的最大值），
    对稀疏线条/骨架图更不容易产生断裂。
    """

    v = _ensure_3hw_views(_to_uint8_views(views))

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    h, w = int(v.shape[1]), int(v.shape[2])

    if h == out_h and w == out_w:
        return v

    # 如果出现比目标还小的情况：直接 padding 到目标尺寸（居中补黑边），不做上采样，避免结构被复制/变粗。
    if h < out_h or w < out_w:
        pad_h = max(out_h - h, 0)
        pad_w = max(out_w - w, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        vv = np.pad(v, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)
        # 安全裁切：避免 padding 后仍超出目标
        return vv[:, :out_h, :out_w].astype(np.uint8, copy=False)

    # 自适应 max pooling（向量化）：
    # 仍然是用 y0=(oy*h)//out_h, y1=((oy+1)*h)//out_h 的分箱方式，
    y_starts = (np.arange(out_h, dtype=np.int64) * h) // out_h  # (out_h,)
    x_starts = (np.arange(out_w, dtype=np.int64) * w) // out_w  # (out_w,)

    # 先沿 y 方向做分段 max： (3,H,W) -> (3,out_h,W)
    tmp = np.maximum.reduceat(v, y_starts, axis=1)
    # 再沿 x 方向做分段 max： (3,out_h,W) -> (3,out_h,out_w)
    out = np.maximum.reduceat(tmp, x_starts, axis=2)
    return out.astype(np.uint8, copy=False)


def _load_views_from_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as z:
        if "views" not in z.files:
            raise KeyError(f"Missing key 'views' in {npz_path}. keys={list(z.files)}")
        return z["views"]


def make_numpy_from_standard_views(
    pair_df: pd.DataFrame,
    fc_dir: Path | str = "./data/standard_views/FC",
    em_dir: Path | str = "./data/standard_views/EM",
    out_hw: tuple[int, int] = (50, 50),
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, tuple[int, int, int]]:
    """从 standard_draw 的单神经元 npz（FC/EM 分开）按 pairs 表构建训练数据。

    - 每一对 (fc_id, em_id) 默认从 `FC/{fc_id}_views.npz` 与 `EM/{em_id}_views.npz` 读取 (3,H,W)
    - pair 内先 padding 到同尺寸，再用 adaptive max pooling 下采样到 50x50（小于目标则 padding）

    Returns:
      x: (M,2,out_h,out_w,3) float32 in [0,1]
      found_df: fc_id, em_id, label
      not_found_df: fc_id, em_id, label, reason
      resolutions: (V,H,W) = (3,out_h,out_w)
    """
    fc_dir = Path(fc_dir)
    em_dir = Path(em_dir)

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid out_hw={out_hw}")

    if not fc_dir.exists():
        raise FileNotFoundError(f"FC standard views dir not found: {fc_dir}")
    if not em_dir.exists():
        raise FileNotFoundError(f"EM standard views dir not found: {em_dir}")

    # cache raw views to reduce IO (note: pair-level padding/resizing is still computed per pair)
    fc_cache: dict[str, np.ndarray] = {}
    em_cache: dict[str, np.ndarray] = {}

    x_list: list[np.ndarray] = []
    found_rows: list[tuple[str, str, float]] = []
    not_found_rows: list[tuple[str, str, float, str]] = []

    required_cols = {"fc_id", "em_id", "label"}
    if not required_cols.issubset(set(pair_df.columns)):
        raise KeyError(f"pair_df must contain columns {sorted(required_cols)}. got={list(pair_df.columns)}")

    for row in pair_df.itertuples(index=False):
        fc_id = str(getattr(row, "fc_id")).strip()
        em_id = str(getattr(row, "em_id")).strip()
        label = float(getattr(row, "label"))

        fc_npz = fc_dir / f"{fc_id}_views.npz"
        em_npz = em_dir / f"{em_id}_views.npz"

        if not fc_npz.exists() and not em_npz.exists():
            not_found_rows.append((fc_id, em_id, label, "missing_fc_and_em_npz"))
            continue
        if not fc_npz.exists():
            not_found_rows.append((fc_id, em_id, label, "missing_fc_npz"))
            continue
        if not em_npz.exists():
            not_found_rows.append((fc_id, em_id, label, "missing_em_npz"))
            continue

        try:
            if fc_id in fc_cache:
                fc_v = fc_cache[fc_id]
            else:
                fc_v = _load_views_from_npz(fc_npz)
                fc_cache[fc_id] = fc_v

            if em_id in em_cache:
                em_v = em_cache[em_id]
            else:
                em_v = _load_views_from_npz(em_npz)
                em_cache[em_id] = em_v
        except Exception as e:
            not_found_rows.append((fc_id, em_id, label, f"load_error:{type(e).__name__}:{e}"))
            continue

        try:
            fc_pad, em_pad = _pad_to_same_size(fc_v, em_v)
            fc_50 = _resize_to_50(fc_pad, (out_h, out_w))
            em_50 = _resize_to_50(em_pad, (out_h, out_w))
        except Exception as e:
            not_found_rows.append((fc_id, em_id, label, f"preprocess_error:{type(e).__name__}:{e}"))
            continue

        x_pair = np.empty((2, out_h, out_w, 3), dtype=np.float32)
        x_pair[0] = np.transpose(fc_50, (1, 2, 0))
        x_pair[1] = np.transpose(em_50, (1, 2, 0))
        x_pair /= 255.0

        x_list.append(x_pair)
        found_rows.append((fc_id, em_id, label))

    if not x_list:
        raise RuntimeError("No valid pairs loaded from standard_views. Check directories, ids, and split CSVs.")

    x = np.stack(x_list, axis=0)
    found_df = pd.DataFrame(found_rows, columns=["fc_id", "em_id", "label"])
    not_found_df = pd.DataFrame(not_found_rows, columns=["fc_id", "em_id", "label", "reason"])
    return x, found_df, not_found_df, (3, out_h, out_w)



def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = Path(cfg.split_dir) / f"train_split_{cfg.fold}_{cfg.used_split_suffix}.csv"
    test_csv = Path(cfg.split_dir) / f"test_split_{cfg.fold}_{cfg.used_split_suffix}.csv"
    return pd.read_csv(train_csv), pd.read_csv(test_csv)


def make_tf_dataset(x, y, batch_size, training, seed):
    """
    x shape: (N,2,H,W,3)
    model input:
        {'FC': (N,H,W,3), 'EM': (N,H,W,3)}
    """

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(x), 4096),
            seed=seed,
            reshuffle_each_iteration=True
        )

    def augment(pair, label):

        fc = pair[0]
        em = pair[1]

        fc_list = []
        em_list = []
        label_list = []

        for swap in [False, True]:  # 交换 FC/EM 视角，保正模型对输入有对称性

            if swap:
                fc0, em0 = em, fc
            else:
                fc0, em0 = fc, em
            
            # Rotation
            for rot in range(4):

                fc_r = tf.image.rot90(fc0, rot)
                em_r = tf.image.rot90(em0, rot)

                fc_list.append(fc_r)
                em_list.append(em_r)
                label_list.append(label)
            
            # Flip
            fc_f = tf.image.flip_left_right(fc0)
            em_f = tf.image.flip_left_right(em0)
            fc_list.append(fc_f)
            em_list.append(em_f)
            label_list.append(label)
    
        fc_stack = tf.stack(fc_list)
        em_stack = tf.stack(em_list)
        label_stack = tf.stack(label_list)

        return tf.data.Dataset.from_tensor_slices(
            ({"FC": fc_stack, "EM": em_stack}, label_stack)
        )

    if training:
        ds = ds.flat_map(augment)
    else:
        ds = ds.map(
            lambda pair, label: ({"FC": pair[0], "EM": pair[1]}, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Keras 3 can treat datasets with unknown cardinality (common after `flat_map`)
    # as one-shot iterators across epochs, which may trigger:
    #   OUT_OF_RANGE: End of sequence
    #   UserWarning: Your input ran out of data; interrupting training.
    # We know our augmentation expands each base pair into a fixed number of samples.
    if training:
        aug_per_pair = 2 * (4 + 1)  # swap * (rot90 x4 + flip)
        num_samples = int(len(x)) * aug_per_pair
    else:
        num_samples = int(len(x))

    ds = ds.batch(batch_size)
    num_batches = int(math.ceil(num_samples / float(batch_size))) if batch_size > 0 else 0
    ds = ds.apply(tf.data.experimental.assert_cardinality(num_batches))
    ds = ds.prefetch(tf.data.AUTOTUNE)

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

    # confusion matrix in standard order: [[TN,FP],[FN,TP]]
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = f1_score(y_true_bin, y_pred_bin, pos_label=1)

    return {
        "cm": cm,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_pos": float(f1_pos),
    }, y_pred_bin
# %%

def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    Path(cfg.save_model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.save_result_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.fig_dir).mkdir(parents=True, exist_ok=True)

    save_model_name = f"{cfg.model_name}_{cfg.used_split_suffix}_{cfg.fold}"

    train_df, test_df = load_splits(cfg)

    x_train_all, pair_train, not_found_train, resolutions = make_numpy_from_standard_views(
        train_df[["fc_id", "em_id", "label"]],
        fc_dir="./data/standard_views/FC",
        em_dir="./data/standard_views/EM",
        out_hw=(50, 50),
    )
    x_test, pair_test, not_found_test, _ = make_numpy_from_standard_views(
        test_df[["fc_id", "em_id", "label"]],
        fc_dir="./data/standard_views/FC",
        em_dir="./data/standard_views/EM",
        out_hw=(50, 50),
    )

    print("Not found train:", len(not_found_train), "Not found test:", len(not_found_test))

    # train/val split (on pair rows aligned with found data)
    x_train, x_val, pair_train, pair_val = train_test_split(
        x_train_all, pair_train, test_size=cfg.val_ratio, random_state=cfg.seed
    )
    y_train = pair_train["label"].to_numpy(dtype=np.float32)
    y_val = pair_val["label"].to_numpy(dtype=np.float32)
    y_test = pair_test["label"].to_numpy(dtype=np.float32)

    print(f"Train {len(x_train)} | Val {len(x_val)} | Test {len(x_test)}")

    # build datasets
    ds_train = make_tf_dataset(x_train, y_train, cfg.batch_size, training=True, seed=cfg.seed)
    ds_val = make_tf_dataset(x_val, y_val, cfg.batch_size, training=False, seed=cfg.seed)
    ds_test = make_tf_dataset(x_test, y_test, cfg.batch_size, training=False, seed=cfg.seed)

    # model
    _, H, W = resolutions
    model = MVCNN_Siamese((H, W, resolutions[0]))

    if cfg.use_pretrain_model:
        pretrain_path = Path(cfg.pretrain_model)
        if not pretrain_path.exists():
            raise FileNotFoundError(f"Pretrain model not found: {pretrain_path}")
        
        model.load_weights(pretrain_path)
        print(f"Loaded pretrain model from {cfg.pretrain_model}")

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

    with open(Path(cfg.save_result_dir) / f"Train_History_{save_model_name}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # reload best and eval
    best = MVCNN_Siamese((H, W, resolutions[0]))  # 重新建同结构
    # NOTE: We only call `predict()` below; compiling is unnecessary and can
    # trigger optimizer-state loading warnings when `.weights.h5` contains saved
    # optimizer variables from a different Keras/optimizer implementation.
    best.load_weights(Path(cfg.save_model_dir) / f"{save_model_name}.weights.h5")


    y_val_pred = best.predict(ds_val, verbose=0)
    val_report, val_pred_bin = metrics_report(y_val, y_val_pred)
    print("Val:", val_report)

    y_test_pred = best.predict(ds_test, verbose=0)
    test_report, test_pred_bin = metrics_report(y_test, y_test_pred)
    print("Test:", test_report)

    with open(Path(cfg.save_result_dir) / f"Test_Result_{save_model_name}.pkl", "wb") as f:
        pickle.dump(test_report, f)

    pred_df = pair_test.copy()
    pred_df["model_pred"] = y_test_pred.reshape(-1)
    pred_df["model_pred_binary"] = test_pred_bin
    pred_df.to_csv(Path(cfg.save_result_dir) / f"test_label_{save_model_name}.csv", index=False)


if __name__ == "__main__":
    import sys
    fold = int(sys.argv[1])
    main(Config(fold=fold))



# # %% 查詢特定層輸出情況
# fmap1_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac1').output)
# fmap1_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac1').output)

# fmap2_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac2').output)
# fmap2_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac2').output)

# fmap3_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac3').output)
# fmap3_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac3').output)

# fmap4_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac4').output)
# fmap4_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac4').output)

# fmap1_test_FC = fmap1_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap1_test_EM = fmap1_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap1_val_FC = fmap1_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap1_val_EM = fmap1_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap1_train_FC = fmap1_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap1_train_EM = fmap1_EM.predict({'EM':x_train_EM}, verbose=2)

# fmap2_test_FC = fmap2_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap2_test_EM = fmap2_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap2_val_FC = fmap2_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap2_val_EM = fmap2_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap2_train_FC = fmap2_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap2_train_EM = fmap2_EM.predict({'EM':x_train_EM}, verbose=2)

# fmap3_test_FC = fmap3_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap3_test_EM = fmap3_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap3_val_FC = fmap3_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap3_val_EM = fmap3_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap3_train_FC = fmap3_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap3_train_EM = fmap3_EM.predict({'EM':x_train_EM}, verbose=2)

# fmap4_test_FC = fmap4_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap4_test_EM = fmap4_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap4_val_FC = fmap4_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap4_val_EM = fmap4_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap4_train_FC = fmap4_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap4_train_EM = fmap4_EM.predict({'EM':x_train_EM}, verbose=2)




# def plot_feature(fmap):
#     f_num = fmap.shape[2]
#     while f_num > 0:
#         plt.figure(figsize=(20,5))
#         for i in range(min(4, fmap.shape[2])):
#             plt.subplot(1,4,i+1)
#             plt.imshow(fmap[:,:,-f_num+i], cmap='magma')
#             plt.xticks([])
#             plt.yticks([])
#         plt.show()
#         f_num -= 4


# plot_feature(fmap4_test_FC[3])
# plot_feature(fmap4_test_EM[3])

# %%
