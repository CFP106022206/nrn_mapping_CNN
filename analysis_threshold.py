# %%
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# %%

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
    ax.set_ylabel("Count")
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


        # %%
# 打開centroids_EM.npy
centroids_em = np.load("./data/descriptors_EM/centroids_EM.npy")
# 打開descriptors_EM.parquet
descriptors_em = pd.read_parquet("./data/descriptors_EM/descriptors_EM.parquet")
# %%
@dataclass(frozen=True)
class SourceDesc:
    source: str
    centroids: np.ndarray  # (N,3) float32
    ratios2d: np.ndarray   # (N,2) float32 -> [r21, r31]
    eigvecs: np.ndarray    # (N,3,3) float32

def load_source(out_dir: str | Path, source: str) -> SourceDesc:
    out_dir = Path(out_dir)

    cent_path = out_dir / f"centroids_{source}.npy"
    ratio_path = out_dir / f"eigvals_ratio_{source}.npy"
    eigvecs_path = out_dir / f"eigvecs_{source}.npy"

    if not cent_path.exists():
        raise FileNotFoundError(f"Missing {cent_path}")
    if not ratio_path.exists():
        raise FileNotFoundError(f"Missing {ratio_path}")
    if not eigvecs_path.exists():
        raise FileNotFoundError(f"Missing {eigvecs_path}")

    centroids = np.load(cent_path).astype(np.float32)
    ratios = np.load(ratio_path).astype(np.float32)
    eigvecs = np.load(eigvecs_path).astype(np.float32)

    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"{cent_path} invalid shape: {centroids.shape}")

    if ratios.ndim != 2:
        raise ValueError(f"{ratio_path} invalid shape: {ratios.shape}")

    if eigvecs.ndim != 3 or eigvecs.shape[1:] != (3, 3):
        raise ValueError(f"{eigvecs_path} invalid shape: {eigvecs.shape}")

    # Accept (N,3)=[r11,r21,r31] OR (N,2)=[r21,r31]
    if ratios.shape[1] == 3:
        ratios2d = ratios[:, 1:3]
    elif ratios.shape[1] == 2:
        ratios2d = ratios
    else:
        raise ValueError(f"{ratio_path} expected (N,3) or (N,2), got {ratios.shape}")

    return SourceDesc(source=source, centroids=centroids, ratios2d=ratios2d, eigvecs=eigvecs)


fc = load_source('data/descriptors_FC/', "FC")
em = load_source('data/descriptors_EM/', "EM")
# %% 畫圖
# import matplotlib.pyplot as plt
# # plt.scatter(fc.ratios2d[:,0], fc.ratios2d[:,1], s=1, linewidths=0, label='FC')
# plt.scatter(em.ratios2d[:,0], em.ratios2d[:,1], s=1, linewidths=0, label='EM')
# plt.xlabel("I")
# plt.ylabel("(r31)")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title("r21/r31 (I1 > I2 > I3, r21=I2/I1, r31=I3/I1)")
# plt.legend()
# plt.savefig('Figure/ratio2d_scatter_EM.png', dpi=300)
# plt.show()
# %%  r31越小越接近長條型，找出不同r31的名單出來畫畫看

fc_r31 = fc.ratios2d[:,1]
em_r31 = em.ratios2d[:,1]


def find_closest_indices(data, targets, tolerance=0.05):
    """
    找到接近目标值的索引
    data: 数据数组
    targets: 目标值列表
    tolerance: 容差范围，找到容差范围内的所有值，然后随机选择一个
    
    Returns:
        dict: {target_value: random_index}，如果容差范围内没有找到则返回最接近的索引
    """
    result = {}
    for target in targets:
        # 找到所有在容差范围内的索引
        mask = np.abs(data - target) <= tolerance
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            # 从容差范围内的索引中随机选择一个
            result[target] = np.random.choice(indices)
        else:
            # 如果容差范围内没有找到，就找最接近的一个
            closest_idx = np.argmin(np.abs(data - target))
            result[target] = closest_idx
    
    return result


# 查找并记录符合条件的索引
target_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# 为FC和EM分别查找
fc_indices = find_closest_indices(fc_r31, target_values, tolerance=0.05)
em_indices = find_closest_indices(em_r31, target_values, tolerance=0.05)

# 讀取data/descriptors_FC/descriptors_FC.parquet
df_fc = pd.read_parquet('data/descriptors_FC/descriptors_FC.parquet')
df_em = pd.read_parquet('data/descriptors_EM/descriptors_EM.parquet')
# 提取对应索引的id
fc_ids = {target: df_fc.iloc[idx]['neuron_id'] for target, idx in fc_indices.items()}
em_ids = {target: df_em.iloc[idx]['neuron_id'] for target, idx in em_indices.items()}


# %% 分析已標注資料們的形狀類型、角度
def classify_pairs_orientation_from_merged(
    merged_pairs: pd.DataFrame,
    df_fc: pd.DataFrame,
    df_em: pd.DataFrame,
    fc_src: SourceDesc,
    em_src: SourceDesc,
    *,
    rod_r31_max: float = 0.35,  # 判斷是否是rod-like，需要r21接近1且r31接近0
    rod_gap_min: float = 0.4,       # r21 - r31
    disk_gap_min: float = 0.30,     # 1 - r21，與 candidate_matching.py 一致
    chunk: int = 2_000_000,
) -> tuple[pd.DataFrame, dict]:
    """
    For each pair in `merged_pairs` find the corresponding indices in `df_fc`/`df_em`
    and classify the pair as: 0=other, 1=rod, 2=disk according to the same
    logic used in `filter_pairs_by_orientation_rod_disk`.

    Returns:
        result_df: merged_pairs augmented with columns `fc_idx`, `em_idx`,
                   `orient_type` (uint8) and `orient_angle_deg` (float32, NaN when not enabled)
        groups: dict mapping type -> DataFrame (subset of result_df)
    """
    # build id -> index map from the descriptor tables
    fc_ids = df_fc["neuron_id"].astype(str).to_numpy()
    em_ids = df_em["neuron_id"].astype(str).to_numpy()
    fc_map = {nid: i for i, nid in enumerate(fc_ids)}
    em_map = {nid: i for i, nid in enumerate(em_ids)}

    mp = merged_pairs.copy()
    mp["fc_id_str"] = mp["fc_id"].astype(str)
    mp["em_id_str"] = mp["em_id"].astype(str)

    fc_idx_series = mp["fc_id_str"].map(fc_map)
    em_idx_series = mp["em_id_str"].map(em_map)

    missing_mask = fc_idx_series.isna() | em_idx_series.isna()
    if missing_mask.any():
        n_miss = int(missing_mask.sum())
        print(f"⚠️ {n_miss} pairs missing in descriptors and will be skipped")

    valid_mask = ~missing_mask
    mp_valid = mp[valid_mask].copy().reset_index(drop=True)
    if mp_valid.empty:
        print("No valid pairs to classify")
        mp["orient_type"] = 0
        mp["orient_angle_deg"] = np.nan
        return mp, {0: mp}

    ia = fc_idx_series[valid_mask].to_numpy(dtype=np.int32)
    ib = em_idx_series[valid_mask].to_numpy(dtype=np.int32)

    n = ia.shape[0]
    ang = np.full((n,), np.nan, dtype=np.float32)
    typ = np.zeros((n,), dtype=np.uint8)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ia_s = ia[s:e]
        ib_s = ib[s:e]

        ra = fc_src.ratios2d[ia_s]
        rb = em_src.ratios2d[ib_s]

        r21_a, r31_a = ra[:, 0], ra[:, 1]
        r21_b, r31_b = rb[:, 0], rb[:, 1]

        gap_a = r21_a - r31_a
        gap_b = r21_b - r31_b

        rod_a = (r31_a <= rod_r31_max) & (gap_a >= rod_gap_min)
        rod_b = (r31_b <= rod_r31_max) & (gap_b >= rod_gap_min)

        disk_a = (1.0 - r21_a) >= disk_gap_min
        disk_b = (1.0 - r21_b) >= disk_gap_min

        enable_rod = rod_a & rod_b
        enable_disk = disk_a & disk_b

        # rod: compare v3
        if np.any(enable_rod):
            sel = enable_rod
            va = fc_src.eigvecs[ia_s[sel], :, 2].astype(np.float32, copy=False)
            vb = em_src.eigvecs[ib_s[sel], :, 2].astype(np.float32, copy=False)
            dot = np.abs(np.einsum("ij,ij->i", va, vb)).astype(np.float32)
            na = np.sqrt(np.einsum("ij,ij->i", va, va)).astype(np.float32)
            nb = np.sqrt(np.einsum("ij,ij->i", vb, vb)).astype(np.float32)
            cos = dot / (na * nb + 1e-12)
            cos_clip = np.clip(cos, -1.0, 1.0)
            ang_enable = np.rad2deg(np.arccos(cos_clip)).astype(np.float32)
            idx = np.flatnonzero(enable_rod) + s
            ang[idx] = ang_enable
            typ[idx] = 1

        # disk: compare v1
        if np.any(enable_disk):
            sel = enable_disk
            va = fc_src.eigvecs[ia_s[sel], :, 0].astype(np.float32, copy=False)
            vb = em_src.eigvecs[ib_s[sel], :, 0].astype(np.float32, copy=False)
            dot = np.abs(np.einsum("ij,ij->i", va, vb)).astype(np.float32)
            na = np.sqrt(np.einsum("ij,ij->i", va, va)).astype(np.float32)
            nb = np.sqrt(np.einsum("ij,ij->i", vb, vb)).astype(np.float32)
            cos = dot / (na * nb + 1e-12)
            cos_clip = np.clip(cos, -1.0, 1.0)
            ang_enable = np.rad2deg(np.arccos(cos_clip)).astype(np.float32)
            idx = np.flatnonzero(enable_disk) + s
            ang[idx] = ang_enable
            typ[idx] = 2

        # 既不是rod也不是disk的，比較v3
        not_rod_disk = ~enable_rod & ~enable_disk
        if np.any(not_rod_disk):
            sel = not_rod_disk
            va = fc_src.eigvecs[ia_s[sel], :, 2].astype(np.float32, copy=False)
            vb = em_src.eigvecs[ib_s[sel], :, 2].astype(np.float32, copy=False)
            dot = np.abs(np.einsum("ij,ij->i", va, vb)).astype(np.float32)
            na = np.sqrt(np.einsum("ij,ij->i", va, va)).astype(np.float32)
            nb = np.sqrt(np.einsum("ij,ij->i", vb, vb)).astype(np.float32)
            cos = dot / (na * nb + 1e-12)
            cos_clip = np.clip(cos, -1.0, 1.0)
            ang_enable = np.rad2deg(np.arccos(cos_clip)).astype(np.float32)
            idx = np.flatnonzero(not_rod_disk) + s
            ang[idx] = ang_enable

    # attach results back to mp_valid and then to original mp
    mp_valid["fc_idx"] = ia
    mp_valid["em_idx"] = ib
    mp_valid["orient_type"] = typ
    mp_valid["orient_angle_deg"] = ang

    # for pairs that were missing, keep orient_type=0 and NaN angle
    mp = mp.drop(columns=["fc_id_str", "em_id_str"])
    result = mp.copy()
    # create columns then fill for valid rows
    result["orient_type"] = 0
    result["orient_angle_deg"] = np.nan
    result.loc[valid_mask.values, "orient_type"] = typ
    result.loc[valid_mask.values, "orient_angle_deg"] = ang

    # grouped dict
    groups = {
        0: result[result["orient_type"] == 0].reset_index(drop=True),
        1: result[result["orient_type"] == 1].reset_index(drop=True),
        2: result[result["orient_type"] == 2].reset_index(drop=True),
    }

    print(f"Classified pairs: total={len(merged_pairs)}, valid={len(result) - int(missing_mask.sum())}")
    print(f"  other(0)={len(groups[0])}, rod(1)={len(groups[1])}, disk(2)={len(groups[2])}")

    return result, groups

# 讀取data/descriptors_FC/descriptors_FC.parquet
df_fc = pd.read_parquet('data/descriptors_FC/descriptors_FC.parquet')
df_em = pd.read_parquet('data/descriptors_EM/descriptors_EM.parquet')

fc = load_source('data/descriptors_FC/', "FC")
em = load_source('data/descriptors_EM/', "EM")

result, groups = classify_pairs_orientation_from_merged(merged_pairs, df_fc, df_em, fc, em)
result.to_csv("merged_pairs_with_orientation.csv", index=False)
# %%
