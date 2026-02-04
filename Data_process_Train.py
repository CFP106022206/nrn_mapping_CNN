'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
5, Iterative self-labeling
6, Transfer Big Model...
'''
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

from util import load_pkl
from model import MVCNN_Siamese


@dataclass(frozen=True)
class Config:
    fold: int
    seed: int = 3407

    used_split_suffix: str = "D1-D6"
    split_dir: str = "./train_test_split"
    map_dict_folder: str = "./data/labeled_sn"

    save_model_dir: str = "./Annotator_Model"
    save_result_dir: str = "./result"
    fig_dir: str = "./Figure"

    initial_lr: float = 1e-5
    train_epochs: int = 100
    batch_size: int = 128

    val_ratio: float = 0.15

    scheduler_exp: float = 0.0  # 0 means off
    min_lr: float = 1e-7


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


def build_pair_index_from_folder(folder: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, float]], tuple[int, int, int]]:
    """
    Build dict: key='fc_em' -> (fc_img[3,H,W], em_img[3,H,W], score)
    """
    folder_p = Path(folder)
    pkl_files = sorted([p for p in folder_p.iterdir() if p.suffix == ".pkl"])
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in {folder}")

    data_dict = {}
    resolutions = None

    for pkl_path in pkl_files:
        data_lst = load_pkl(str(pkl_path))
        for item in data_lst:
            fc_id, em_id, score, fc_arr, em_arr = item[0], item[1], item[2], item[3], item[4]
            key = f"{fc_id}_{em_id}"
            data_dict[key] = (fc_arr, em_arr, score)
            if resolutions is None:
                resolutions = fc_arr.shape  # expect (3,H,W) or (views,H,W)

    if resolutions is None:
        raise RuntimeError("Empty mapping data; cannot infer resolutions.")

    # return (views, H, W)
    return data_dict, resolutions


def make_numpy_from_pairs(
    pair_df: pd.DataFrame,
    data_dict: dict[str, tuple[np.ndarray, np.ndarray, float]],
    resolutions: tuple[int, int, int],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Output x: (N,2,H,W,3), pair_df_aligned with score column, not_found df.
    """
    views, H, W = resolutions
    x = np.zeros((len(pair_df), 2, H, W, views), dtype=np.float32)

    found_rows = []
    not_found_rows = []

    for i, row in enumerate(pair_df.itertuples(index=False)):
        fc_id = getattr(row, "fc_id")
        em_id = getattr(row, "em_id")
        label = getattr(row, "label")
        key = f"{fc_id}_{em_id}"
        if key not in data_dict:
            not_found_rows.append((fc_id, em_id, label))
            continue

        fc_arr, em_arr, score = data_dict[key]
        # fc_arr/em_arr: (views,H,W)
        x[i, 0, :, :, :] = np.transpose(fc_arr, (1, 2, 0))
        x[i, 1, :, :, :] = np.transpose(em_arr, (1, 2, 0))
        found_rows.append((fc_id, em_id, label, score))

    # drop not-found zero rows
    mask = np.any(x != 0, axis=(1, 2, 3, 4))
    x = x[mask]

    found_df = pd.DataFrame(found_rows, columns=["fc_id", "em_id", "label", "score"])
    not_found_df = pd.DataFrame(not_found_rows, columns=["fc_id", "em_id", "label"])

    # normalize globally (keep same behavior)
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)

    return x, found_df, not_found_df


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
    """
    x shape: (N,2,H,W,3)  -> model expects dict {'FC':(N,H,W,3), 'EM':(N,H,W,3)}
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=min(len(x), 4096), seed=seed, reshuffle_each_iteration=True)

    def _map(pair, label):
        fc = pair[0]
        em = pair[1]

        if training:
            # 1) symmetry: swap FC/EM with 50% prob
            do_swap = tf.random.uniform(()) < 0.5
            fc, em = tf.cond(do_swap, lambda: (em, fc), lambda: (fc, em))

            # 2) flips
            do_lr = tf.random.uniform(()) < 0.5
            fc = tf.cond(do_lr, lambda: tf.image.flip_left_right(fc), lambda: fc)
            em = tf.cond(do_lr, lambda: tf.image.flip_left_right(em), lambda: em)

            do_ud = tf.random.uniform(()) < 0.5
            fc = tf.cond(do_ud, lambda: tf.image.flip_up_down(fc), lambda: fc)
            em = tf.cond(do_ud, lambda: tf.image.flip_up_down(em), lambda: em)

            # 3) rot90
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            fc = tf.image.rot90(fc, k=k)
            em = tf.image.rot90(em, k=k)

        return {"FC": fc, "EM": em}, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
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


def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    Path(cfg.save_model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.save_result_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.fig_dir).mkdir(parents=True, exist_ok=True)

    save_model_name = f"Annotator_{cfg.used_split_suffix}_{cfg.fold}"

    train_df, test_df = load_splits(cfg)

    # build mapping index once
    data_dict, resolutions = build_pair_index_from_folder(cfg.map_dict_folder)

    # to numpy
    x_train_all, pair_train, _ = make_numpy_from_pairs(train_df[["fc_id","em_id","label"]], data_dict, resolutions)
    x_test, pair_test, _ = make_numpy_from_pairs(test_df[["fc_id","em_id","label"]], data_dict, resolutions)

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

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg.initial_lr),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bi_acc")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(cfg.save_model_dir) / f"{save_model_name}.h5"),
            monitor="val_loss",
            save_best_only=True,
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
    best = tf.keras.models.load_model(Path(cfg.save_model_dir) / f"{save_model_name}.h5")

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
