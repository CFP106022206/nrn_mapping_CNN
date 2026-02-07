# %%
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_merge_conf_csvs(
    label_dir: str | Path = "./labeled_info",
    datasets: list[str] = ["D1", "D2", "D3", "D4", "D5", "D6"],
) -> pd.DataFrame:
    """
    合并所有 D1-D6 的 conf.csv 文件

    预期每个 csv 至少包含：fc_id, em_id, label
    """
    dfs = []
    for dataset in datasets:
        csv_path = Path(label_dir) / f"{dataset}_conf.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["dataset"] = dataset
            dfs.append(df)
            print(f"Loaded {dataset}: {len(df)} rows")
        else:
            print(f"{csv_path} not found, skipping")

    if not dfs:
        raise FileNotFoundError(f"No conf.csv found under: {label_dir}")

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["fc_id", "em_id"])
    merged_df["fc_id"] = merged_df["fc_id"].astype(str)
    merged_df["em_id"] = merged_df["em_id"].astype(str)

    print(f"\nTotal pairs after merge: {len(merged_df)}")
    return merged_df


def load_descriptors(parquet_path: str | Path) -> pd.DataFrame:
    """加载 parquet 文件中的 descriptors"""
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {parquet_path}: {len(df)} neurons")
    return df


def _pick_centroid_cols(df: pd.DataFrame) -> list[str]:
    want = ["cx", "cy", "cz"]
    if all(c in df.columns for c in want):
        return want
    raise ValueError(f"Missing centroid columns. Need {want}, got: {list(df.columns)}")


def _pick_ratio2d_cols(df: pd.DataFrame) -> list[str]:
    """
    你的 ratio 可能出现在不同命名里：
    - (r21, r31)       ✅ 最理想
    - (r2, r3)         ✅ 我们前面建议的统一命名
    - (r11, r21, r31)  ✅ 你旧版 batch_run 用过
    - (r1, r2, r3)     ✅ 也常见
    """
    candidates = [
        ["r21", "r31"],
        ["r2", "r3"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols

    # fallback: 有 3 个 ratio
    if all(c in df.columns for c in ["r11", "r21", "r31"]):
        return ["r21", "r31"]
    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        return ["r2", "r3"]

    raise ValueError(
        "Missing ratio columns for (r21,r31). "
        f"Columns found: {list(df.columns)}"
    )


def compute_centroid_distances(
    merged_pairs: pd.DataFrame,
    fc_desc: pd.DataFrame,
    em_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算每个 pair 的 centroid 距离
    Returns: fc_id, em_id, centroid_distance
    """
    centroid_cols = _pick_centroid_cols(fc_desc)

    fc_centroids = (
        fc_desc.set_index("neuron_id")[centroid_cols]
        .rename(columns={c: f"fc_{c}" for c in centroid_cols})
    )
    em_centroids = (
        em_desc.set_index("neuron_id")[centroid_cols]
        .rename(columns={c: f"em_{c}" for c in centroid_cols})
    )

    # 确保索引与 pair id 是 string
    fc_centroids.index = fc_centroids.index.astype(str)
    em_centroids.index = em_centroids.index.astype(str)

    result = merged_pairs[["fc_id", "em_id"]].copy()
    result["fc_id"] = result["fc_id"].astype(str)
    result["em_id"] = result["em_id"].astype(str)

    result = result.merge(fc_centroids, left_on="fc_id", right_index=True, how="left")
    result = result.merge(em_centroids, left_on="em_id", right_index=True, how="left")

    fc_cols = [f"fc_{c}" for c in centroid_cols]
    em_cols = [f"em_{c}" for c in centroid_cols]

    # 如果有找不到的 id，会出现 NaN；这里先不 silent 变 0，避免误导
    if result[fc_cols + em_cols].isna().any().any():
        missing_fc = result.loc[result[fc_cols].isna().any(axis=1), "fc_id"].nunique()
        missing_em = result.loc[result[em_cols].isna().any(axis=1), "em_id"].nunique()
        print(f"⚠️ centroid merge has missing ids: missing_fc={missing_fc}, missing_em={missing_em}")

    diff2 = 0.0
    for fc, em in zip(fc_cols, em_cols):
        diff2 = diff2 + (result[fc] - result[em]) ** 2
    result["centroid_distance"] = np.sqrt(diff2)

    result = result[["fc_id", "em_id", "centroid_distance"]]
    print(f"✅ Computed centroid distances: {len(result)} pairs")
    return result


def compute_ratio2d_distances(
    merged_pairs: pd.DataFrame,
    fc_desc: pd.DataFrame,
    em_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算每个 pair 的 (r21,r31) 2D 距离（或映射到 r2,r3）
    Returns: fc_id, em_id, ratio2d_distance
    """
    ratio_cols_fc = _pick_ratio2d_cols(fc_desc)
    ratio_cols_em = _pick_ratio2d_cols(em_desc)
    print(f"✅ Using FC ratio cols: {ratio_cols_fc}; EM ratio cols: {ratio_cols_em}")

    fc_ratio = (
        fc_desc.set_index("neuron_id")[ratio_cols_fc]
        .rename(columns={ratio_cols_fc[0]: "fc_r21", ratio_cols_fc[1]: "fc_r31"})
    )
    em_ratio = (
        em_desc.set_index("neuron_id")[ratio_cols_em]
        .rename(columns={ratio_cols_em[0]: "em_r21", ratio_cols_em[1]: "em_r31"})
    )

    fc_ratio.index = fc_ratio.index.astype(str)
    em_ratio.index = em_ratio.index.astype(str)

    result = merged_pairs[["fc_id", "em_id"]].copy()
    result["fc_id"] = result["fc_id"].astype(str)
    result["em_id"] = result["em_id"].astype(str)

    result = result.merge(fc_ratio, left_on="fc_id", right_index=True, how="left")
    result = result.merge(em_ratio, left_on="em_id", right_index=True, how="left")

    if result[["fc_r21", "fc_r31", "em_r21", "em_r31"]].isna().any().any():
        missing_fc = result.loc[result[["fc_r21", "fc_r31"]].isna().any(axis=1), "fc_id"].nunique()
        missing_em = result.loc[result[["em_r21", "em_r31"]].isna().any(axis=1), "em_id"].nunique()
        print(f"⚠️ ratio merge has missing ids: missing_fc={missing_fc}, missing_em={missing_em}")

    dr21 = (result["fc_r21"] - result["em_r21"]) ** 2
    dr31 = (result["fc_r31"] - result["em_r31"]) ** 2
    result["ratio2d_distance"] = np.sqrt(dr21 + dr31)

    result = result[["fc_id", "em_id", "ratio2d_distance"]]
    print(f"✅ Computed ratio2d distances: {len(result)} pairs")
    return result


def plot_distance_histograms(result: pd.DataFrame) -> None:
    """
    绘制 centroid_distance 和 ratio2d_distance 的直方图
    分别显示 label > 0.5 (正样本) 和 label <= 0.5 (负样本)
    """
    positive = result[result["label"] > 0.5]
    negative = result[result["label"] <= 0.5]

    print(f"\n样本分布:")
    print(f"  Positive (label > 0.5): {len(positive)} pairs")
    print(f"  Negative (label <= 0.5): {len(negative)} pairs")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Centroid Distance
    ax = axes[0]
    bins = np.linspace(0, max(float(result["centroid_distance"].max()), 1.0), 50)
    ax.hist(negative["centroid_distance"], bins=bins, alpha=0.6, label=f"Negative (≤0.5): {len(negative)}")
    ax.hist(positive["centroid_distance"], bins=bins, alpha=0.6, label=f"Positive (>0.5): {len(positive)}")
    ax.set_xlabel("Centroid Distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Centroid Distance Distribution", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Ratio2D Distance (r21,r31)
    ax = axes[1]
    bins = np.linspace(0, max(float(result["ratio2d_distance"].max()), 1.0), 50)
    ax.hist(negative["ratio2d_distance"], bins=bins, alpha=0.6, label=f"Negative (≤0.5): {len(negative)}")
    ax.hist(positive["ratio2d_distance"], bins=bins, alpha=0.6, label=f"Positive (>0.5): {len(positive)}")
    ax.set_xlabel("Ratio2D Distance in (r21,r31)")
    ax.set_ylabel("Frequency")
    ax.set_title("Inertia Ratio 2D Distance Distribution", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("./Figure/distance_threshold_histograms.png", dpi=300)
    plt.show()

    print(f"\n📈 Centroid Distance 统计:")
    print(f"  Negative - Mean: {negative['centroid_distance'].mean():.4f}, Std: {negative['centroid_distance'].std():.4f}")
    print(f"  Positive - Mean: {positive['centroid_distance'].mean():.4f}, Std: {positive['centroid_distance'].std():.4f}")

    print(f"\n📈 Ratio2D Distance 统计:")
    print(f"  Negative - Mean: {negative['ratio2d_distance'].mean():.4f}, Std: {negative['ratio2d_distance'].std():.4f}")
    print(f"  Positive - Mean: {positive['ratio2d_distance'].mean():.4f}, Std: {positive['ratio2d_distance'].std():.4f}")


def main():
    print("\n[1/4] 合并 D1-D6 conf.csv 文件...")
    merged_pairs = load_and_merge_conf_csvs()

    print("\n[2/4] 加载 FC descriptors...")
    fc_desc = load_descriptors("./data/descriptors_FC/descriptors_FC.parquet")
    print("FC columns:", list(fc_desc.columns))

    print("\n[2/4] 加载 EM descriptors...")
    em_desc = load_descriptors("./data/descriptors_EM/descriptors_EM.parquet")
    print("EM columns:", list(em_desc.columns))

    print("\n[3/4] 计算 centroid 距离...")
    centroid_dists = compute_centroid_distances(merged_pairs, fc_desc, em_desc)

    print("\n[4/4] 计算 (r21,r31) 2D 距离...")
    ratio2d_dists = compute_ratio2d_distances(merged_pairs, fc_desc, em_desc)

    final_result = merged_pairs[["fc_id", "em_id", "label", "dataset"]].copy()
    final_result = final_result.merge(centroid_dists, on=["fc_id", "em_id"], how="left")
    final_result = final_result.merge(ratio2d_dists, on=["fc_id", "em_id"], how="left")

    final_result = final_result[
        ["fc_id", "em_id", "label", "dataset", "centroid_distance", "ratio2d_distance"]
    ]

    print("\n[5/5] 绘制直方图...")
    plot_distance_histograms(final_result)

    return final_result


if __name__ == "__main__":
    result_df = main()

# %%
