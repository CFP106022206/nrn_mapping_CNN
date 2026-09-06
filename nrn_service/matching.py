"""單顆神經元 vs 一整個資料庫的候選初篩。

離線的 candidate_matching.run_matching 是「整個 FC 資料庫 x 整個 EM 資料庫」，
服務端要的是「一顆 x 一個資料庫」。這裡直接重用 candidate_matching.py 裡的
三段過濾函式，把查詢那顆當成只有一列的 A side，所以篩選邏輯與離線流程完全一致，
candidate_matching.py 本身不需要任何修改。

三段過濾（與離線相同）：
    1. 質心距離 <= centroid_th        （用常駐的 KDTree）
    2. (r21, r31) 2D 距離 <= ratio_th
    3. rod / disk 形狀的方向性過濾（兩側同時落在同一種形狀時才啟用）

最後再按 descriptor 距離取前 K，用來保證單次查詢的延遲上限。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from candidate_matching import (
    filter_pairs_by_orientation_rod_disk,
    filter_pairs_by_ratio2d_distance,
)

from .config import MatchConfig
from .db_index import NeuronDescriptor, SideIndex

CANDIDATE_COLUMNS = ("target_id", "centroid_dist", "ratio_dist", "prefilter_score")


def match_one(
    query: NeuronDescriptor,
    target: SideIndex,
    cfg: MatchConfig,
) -> pd.DataFrame:
    """找出 target 資料庫中與 query 幾何特徵相近的候選。

    回傳欄位：target_id / centroid_dist / ratio_dist / prefilter_score，
    依 prefilter_score 由小到大排序（越小越相近），最多 cfg.top_k_candidates 列。
    沒有候選時回傳空的 DataFrame（欄位仍在）。
    """
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in CANDIDATE_COLUMNS})
    empty["target_id"] = empty["target_id"].astype(str)

    if target.tree is None:
        raise RuntimeError(f"{target.side} 的 KDTree 沒有建立")

    q_cent = np.asarray(query.centroid, dtype=np.float64).reshape(1, 3)
    q_ratio = np.asarray(query.ratios2d, dtype=np.float32).reshape(1, 2)
    q_eigvec = np.asarray(query.eigvecs, dtype=np.float32).reshape(1, 3, 3)

    # --- 1) 質心距離 ---
    hits = target.tree.query_ball_point(q_cent[0], r=float(cfg.centroid_th))
    if len(hits) == 0:
        return empty
    ib = np.asarray(sorted(hits), dtype=np.int32)
    ia = np.zeros_like(ib)

    # --- 2) inertia ratio 距離 ---
    ia2, ib2, d_ratio = filter_pairs_by_ratio2d_distance(
        ia, ib, q_ratio, target.desc.ratios2d, threshold=float(cfg.ratio_th)
    )
    if ib2.size == 0:
        return empty

    # --- 3) 方向性過濾 ---
    ia3, ib3, _ang, _typ = filter_pairs_by_orientation_rod_disk(
        ia2,
        ib2,
        q_ratio,
        target.desc.ratios2d,
        q_eigvec,
        target.desc.eigvecs,
        rod_angle_th_deg=float(cfg.rod_angle_th_deg),
        disk_angle_th_deg=float(cfg.disk_angle_th_deg),
    )
    if ib3.size == 0:
        return empty

    # 第 3 步不回傳距離，重新取一次（純查表，成本可忽略）
    d_cent = np.linalg.norm(
        target.desc.centroids[ib3].astype(np.float64) - q_cent, axis=1
    ).astype(np.float32)
    dr = np.linalg.norm(
        target.desc.ratios2d[ib3].astype(np.float32) - q_ratio, axis=1
    ).astype(np.float32)

    # 兩個距離量綱不同，各自除以自己的門檻再相加，範圍大致落在 [0, 2]
    prefilter = (d_cent / float(cfg.centroid_th)) + (dr / float(cfg.ratio_th))

    df = pd.DataFrame(
        {
            "target_id": target.neuron_ids[ib3],
            "centroid_dist": d_cent,
            "ratio_dist": dr,
            "prefilter_score": prefilter,
        }
    )
    df = df.drop_duplicates("target_id", keep="first")
    df = df.sort_values("prefilter_score", kind="mergesort").reset_index(drop=True)

    if cfg.top_k_candidates > 0 and len(df) > cfg.top_k_candidates:
        df = df.head(cfg.top_k_candidates).reset_index(drop=True)
    return df
