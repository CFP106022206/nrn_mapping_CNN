'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
5, Iterative self-labeling
6, Transfer Big Model...
'''

'''
這個檔案使用10-fold validation規則
'''
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    mode: int = 1
    cross_validation_num: int = 10
    used_label: str = "soft_label"  # "thres0.5" | "thres0.6" | "soft_label"
    seed: int = 7
    mode2_file_path: str = "./labeled_info/nblast_D2+D5+D6_50as1.csv"
    out_dir: str = "./train_test_split"


LABEL_CSVS: Dict[str, Dict[str, str]] = {
    "thres0.5": {
        "D1": "./labeled_info/D1_20221230.csv",
        "D2": "./labeled_info/D2_20230710.csv",
        "D3": "./labeled_info/D3_20221230.csv",
        "D4": "./labeled_info/D4_20230710.csv",
        "D5": "./labeled_info/D5_20221230.csv",
        "D6": "./labeled_info/D6_20230523.csv",
    },
    "thres0.6": {
        "D1": "./labeled_info/D1_20230113.csv",
        "D2": "./labeled_info/D2_60as1.csv",
        "D3": "./labeled_info/D3_20230113.csv",
        "D4": "./labeled_info/D4_60as1.csv",
        "D5": "./labeled_info/D5_60as1.csv",
        "D6": "./labeled_info/D6_60as1.csv",
    },
    "soft_label": {
        "D1": "./labeled_info/D1_conf.csv",
        "D2": "./labeled_info/D2_conf.csv",
        "D3": "./labeled_info/D3_conf.csv",
        "D4": "./labeled_info/D4_conf.csv",
        "D5": "./labeled_info/D5_conf.csv",
        "D6": "./labeled_info/D6_conf.csv",
    },
}


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def read_label_tables(used_label: str) -> Dict[str, pd.DataFrame]:
    if used_label not in LABEL_CSVS:
        raise ValueError(f"Unknown used_label={used_label}. Options={list(LABEL_CSVS)}")

    tables: Dict[str, pd.DataFrame] = {}
    for name, path in LABEL_CSVS[used_label].items():
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset=["fc_id", "em_id"])
        tables[name] = df
    return tables


def concat_all_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(list(tables.values()), ignore_index=True)
    df = df.drop_duplicates(subset=["fc_id", "em_id"])
    return df


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_split(out_dir: Path, i: int, train_df: pd.DataFrame, test_df: pd.DataFrame, suffix: str = "D1-D6") -> None:
    test_df.to_csv(out_dir / f"test_split_{i}_{suffix}.csv", index=False)
    train_df.to_csv(out_dir / f"train_split_{i}_{suffix}.csv", index=False)


def kfold_indices(df: pd.DataFrame, n_splits: int, seed: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kf.split(df)


def drop_by_pairs(full_df: pd.DataFrame, drop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows in full_df whose (fc_id, em_id) appear in drop_df.
    Avoids merge(score_x/score_y) mess.
    """
    key_cols = ["fc_id", "em_id"]
    drop_keys = set(map(tuple, drop_df[key_cols].to_numpy()))
    mask = ~full_df[key_cols].apply(tuple, axis=1).isin(drop_keys)
    return full_df.loc[mask].reset_index(drop=True)


def align_columns_like(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c in ref.columns]
    return df[cols]


# -----------------------------
# Modes
# -----------------------------
def mode0_holdout(label_all: pd.DataFrame, out_dir: Path, seed: int, test_ratio: float = 0.1) -> None:
    label_all = label_all.sample(frac=1, random_state=seed).reset_index(drop=True)

    test_size = int(len(label_all) * test_ratio)
    test_df = label_all.iloc[:test_size].reset_index(drop=True)
    train_df = label_all.iloc[test_size:].reset_index(drop=True)

    save_split(out_dir, 0, train_df, test_df)


def mode1_kfold(label_all: pd.DataFrame, out_dir: Path, n_splits: int, seed: int) -> None:
    for i, (train_idx, test_idx) in enumerate(kfold_indices(label_all, n_splits, seed)):
        train_df = label_all.iloc[train_idx].reset_index(drop=True)
        test_df = label_all.iloc[test_idx].reset_index(drop=True)
        save_split(out_dir, i, train_df, test_df)


def mode2_custom_test_kfold(
    label_all: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    out_dir: Path,
    n_splits: int,
    seed: int,
    mode2_file_path: str,
) -> None:
    test_table = pd.read_csv(mode2_file_path)
    test_table = align_columns_like(test_table, label_all)

    # 用 label_all 校正 label/score：只保留在 label_all 裡存在的 pairs
    test_table = test_table.merge(label_all, on=["fc_id", "em_id"], how="inner", suffixes=("_in", ""))
    # 最終只保留 label_all 的 score/label 欄位
    keep_cols = list(label_all.columns)
    test_table = test_table[keep_cols].drop_duplicates(subset=["fc_id", "em_id"]).reset_index(drop=True)

    # 分離 D2+D6 與 D5，讓 KFold 更均勻
    d2d6 = pd.concat([tables["D2"], tables["D6"]], ignore_index=True)
    test_table_d2 = test_table.merge(d2d6[["fc_id", "em_id"]], on=["fc_id", "em_id"], how="inner")
    test_table_d5 = test_table.merge(tables["D5"][["fc_id", "em_id"]], on=["fc_id", "em_id"], how="inner")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    d2_splits = [test_table_d2.iloc[idx].reset_index(drop=True) for _, idx in kf.split(test_table_d2)]
    d5_splits = [test_table_d5.iloc[idx].reset_index(drop=True) for _, idx in kf.split(test_table_d5)]

    for i in range(n_splits):
        test_df = pd.concat([d2_splits[i], d5_splits[i]], ignore_index=True).drop_duplicates(subset=["fc_id", "em_id"])
        train_df = drop_by_pairs(label_all, test_df)

        save_split(out_dir, i, train_df, test_df)


def mode3_fc_heavy_test(label_all: pd.DataFrame, out_dir: Path, seed: int, approx_total_test_pairs: int = 100) -> None:

    fc_counts = label_all.groupby("fc_id").size().sort_values(ascending=False)

    test_tables: List[pd.DataFrame] = []
    total = 0

    print("\nTest set's fc_id / Number of pairs")
    for fc_id, cnt in fc_counts.items():
        print(fc_id, int(cnt))
        fc_df = label_all[label_all["fc_id"] == fc_id].copy()

        # 至少要有一個 positive 才納入
        if (fc_df["label"] == 1).any():
            pos_one = fc_df[fc_df["label"] == 1].iloc[:1]

            # shuffle
            fc_df = fc_df.sample(frac=1, random_state=seed).reset_index(drop=True)

            half = max(1, len(fc_df) // 2)
            fc_df = pd.concat([fc_df.iloc[:half], pos_one], ignore_index=True)
            fc_df = fc_df.drop_duplicates(subset=["fc_id", "em_id"]).reset_index(drop=True)

            test_tables.append(fc_df)
            total += len(fc_df)

        if total >= approx_total_test_pairs:
            break

    test_df = pd.concat(test_tables, ignore_index=True).drop_duplicates(subset=["fc_id", "em_id"]).reset_index(drop=True)
    train_df = drop_by_pairs(label_all, test_df)

    save_split(out_dir, 0, train_df, test_df)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    ensure_out_dir(out_dir)

    tables = read_label_tables(cfg.used_label)
    label_all = concat_all_tables(tables)

    if cfg.mode == 0:
        mode0_holdout(label_all, out_dir, cfg.seed)
    elif cfg.mode == 1:
        mode1_kfold(label_all, out_dir, cfg.cross_validation_num, cfg.seed)
    elif cfg.mode == 2:
        mode2_custom_test_kfold(label_all, tables, out_dir, cfg.cross_validation_num, cfg.seed, cfg.mode2_file_path)
    elif cfg.mode == 3:
        mode3_fc_heavy_test(label_all, out_dir, cfg.seed)
    else:
        raise ValueError(f"Unknown mode={cfg.mode}")


if __name__ == "__main__":
    main()

