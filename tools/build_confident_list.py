"""用兩個模型的共識，從全庫掃描結果建立高信心配對清單。

做法：取兩個模型 top-N 結果的**聯集**當候選池，對池中每一對**重新計算兩個模型的分數**
（不沿用輸入 CSV 裡的分數，確保兩邊走同一套前處理），再加上與模型無關的幾何指標 IoU，
最後用「兩個模型都夠高 + IoU 夠大」篩選。

    python3 tools/build_confident_list.py \\
        --fc_finetune  result/fc_all_top5.csv \\
        --fc_annotator result/fc_all_top5_annotator_notrunc.csv \\
        --em_finetune  result/em_all_top5.csv \\
        --em_annotator result/em_all_top5_annotator_notrunc.csv \\
        --min_score 0.6 --min_iou 0.11 --out_dir result

輸出（out_dir 底下）：
    confident_fc_to_em.csv         完整欄位，含兩個模型分數、iou、min_score、mutual
    confident_em_to_fc.csv
    confident_fc_to_em_simple.csv  只有 source_id, target_id, score, rank（給前端）
    confident_em_to_fc_simple.csv

注意：
* 輸入 CSV 只用到 source_id / target_id 兩欄，分數一律重算，所以不怕輸入檔是哪個模型跑的。
* `rank` 依 `score`（FineTune 分數）由大到小，同分用 target_id 決定，確保可重現、不跳號。
* `mutual` 表示同一對在兩個方向的清單裡都通過篩選。
* IoU 門檻 0.11 的由來：60 組人工確認的真配對，IoU 最小值就是 0.110（中位數 0.426）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import MVCNN_Siamese  # noqa: E402
from nrn_service.config import ServiceConfig  # noqa: E402
from nrn_service.view_store import ViewStore  # noqa: E402
from swc_util import _pad_to_same_size, _resize_to_50  # noqa: E402

PAIR_COLS = ["source_id", "target_id"]


def load_pairs(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"source_id": str, "target_id": str})
    missing = [c for c in PAIR_COLS if c not in d.columns]
    if missing:
        raise KeyError(f"{path} 缺少欄位 {missing}（需要 source_id / target_id）")
    return d[PAIR_COLS]


def build_union(finetune_csv: str, annotator_csv: str) -> pd.DataFrame:
    a, b = load_pairs(finetune_csv), load_pairs(annotator_csv)
    u = pd.concat([a, b]).drop_duplicates(PAIR_COLS).reset_index(drop=True)
    print(f"  FineTune {len(a):,} 對 | Annotator {len(b):,} 對 | 聯集 {len(u):,} 對", flush=True)
    return u


def score_and_iou(pairs: pd.DataFrame, src_side: str, models: dict, views: dict,
                  chunk: int = 10000) -> pd.DataFrame:
    """對每一對算出兩個模型的分數與 IoU。回傳加了欄位的 DataFrame。"""
    tgt_side = "EM" if src_side == "FC" else "FC"
    S = pairs["source_id"].values
    T = pairs["target_id"].values
    n = len(pairs)
    out = {k: np.full(n, np.nan, np.float32) for k in models}
    iou = np.full(n, np.nan, np.float32)
    ok = np.zeros(n, bool)
    t0 = time.time()

    for s0 in range(0, n, chunk):
        e0 = min(s0 + chunk, n)
        A, B, idx = [], [], []
        for i in range(s0, e0):
            sv_raw, tv_raw = views[src_side].get(str(S[i])), views[tgt_side].get(str(T[i]))
            if sv_raw is None or tv_raw is None:
                continue
            a, b = _pad_to_same_size(sv_raw, tv_raw)
            sv, tv = _resize_to_50(a), _resize_to_50(b)
            iou[i] = ((sv > 0) & (tv > 0)).sum() / ((sv > 0) | (tv > 0)).sum()
            fc_v, em_v = (sv, tv) if src_side == "FC" else (tv, sv)
            A.append(np.transpose(fc_v, (1, 2, 0)))
            B.append(np.transpose(em_v, (1, 2, 0)))
            idx.append(i)
            ok[i] = True
        if not idx:
            continue
        fc_img = np.stack(A).astype(np.float32) / 255.0
        em_img = np.stack(B).astype(np.float32) / 255.0
        for name, m in models.items():
            p = m.predict({"FC": fc_img, "EM": em_img}, verbose=0, batch_size=512).reshape(-1)
            out[name][idx] = p
        print(f"    [{e0}/{n}] {time.time() - t0:.0f}s", flush=True)

    res = pairs.copy()
    for name in models:
        res[name] = out[name]
    res["iou"] = iou
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"  ⚠️ {n_bad} 對取不到三視圖，已剔除", flush=True)
    return res[ok].reset_index(drop=True)


def finalize(df: pd.DataFrame, min_score: float, min_iou: float) -> pd.DataFrame:
    keep = df[(df.finetune_score > min_score) & (df.annotator_score > min_score)
              & (df.iou >= min_iou)].copy()
    keep["min_score"] = keep[["finetune_score", "annotator_score"]].min(axis=1)
    # rank 依 finetune_score（= simple 檔的 score 欄）遞減，同分用 target_id，確保可重現
    keep = keep.sort_values(["source_id", "finetune_score", "target_id"],
                            ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    keep["rank"] = keep.groupby("source_id").cumcount() + 1
    return keep


def main() -> int:
    cfg = ServiceConfig()
    ap = argparse.ArgumentParser(description="Build the two-model consensus confident list.")
    ap.add_argument("--fc_finetune", required=True)
    ap.add_argument("--fc_annotator", required=True)
    ap.add_argument("--em_finetune", required=True)
    ap.add_argument("--em_annotator", required=True)
    ap.add_argument("--finetune_weights", default=cfg.model.weights)
    ap.add_argument("--annotator_weights", default="./Annotator_Model/Annotator_D1-D6_0.weights.h5")
    ap.add_argument("--min_score", type=float, default=0.6, help="兩個模型都必須 > 這個值")
    ap.add_argument("--min_iou", type=float, default=0.11, help="IoU 下限（真配對實測最小值 0.110）")
    ap.add_argument("--out_dir", default="result")
    args = ap.parse_args()

    views = {s: ViewStore(s, store_dir=cfg.paths.view_store_dir,
                          npz_dir=cfg.paths.views_dir(s)) for s in ("FC", "EM")}
    models = {}
    for name, w in (("finetune_score", args.finetune_weights),
                    ("annotator_score", args.annotator_weights)):
        if not Path(w).exists():
            ap.error(f"找不到權重檔：{w}")
        m = MVCNN_Siamese(cfg.model.input_size)
        m.load_weights(w)
        models[name] = m
    print(f"[model] finetune={args.finetune_weights}\n[model] annotator={args.annotator_weights}",
          flush=True)
    print(f"[filter] 兩模型都 > {args.min_score}，IoU >= {args.min_iou}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full: dict[str, pd.DataFrame] = {}
    for direction, src_side, ft, an in (("fc_to_em", "FC", args.fc_finetune, args.fc_annotator),
                                        ("em_to_fc", "EM", args.em_finetune, args.em_annotator)):
        print(f"\n[{direction}]", flush=True)
        u = build_union(ft, an)
        scored = score_and_iou(u, src_side, models, views)
        full[direction] = finalize(scored, args.min_score, args.min_iou)
        print(f"  通過篩選 {len(full[direction]):,} 對，"
              f"涵蓋 {full[direction].source_id.nunique():,} 顆 {src_side}", flush=True)

    # mutual：同一對（fc, em）在兩個方向都通過
    fc2em = full["fc_to_em"]
    em2fc = full["em_to_fc"]
    set_a = set(map(tuple, fc2em[["source_id", "target_id"]].values))
    set_b = {(t, s) for s, t in map(tuple, em2fc[["source_id", "target_id"]].values)}
    mutual = set_a & set_b
    fc2em["mutual"] = [tuple(x) in mutual for x in fc2em[["source_id", "target_id"]].values]
    em2fc["mutual"] = [(t, s) in mutual for s, t in em2fc[["source_id", "target_id"]].values]
    print(f"\n雙向互選 {len(mutual):,} 對", flush=True)

    for direction, d in (("fc_to_em", fc2em), ("em_to_fc", em2fc)):
        cols = ["source_id", "target_id", "finetune_score", "annotator_score",
                "iou", "min_score", "rank", "mutual"]
        p1 = out_dir / f"confident_{direction}.csv"
        d[cols].to_csv(p1, index=False)
        p2 = out_dir / f"confident_{direction}_simple.csv"
        (d[["source_id", "target_id", "finetune_score", "rank"]]
         .rename(columns={"finetune_score": "score"})).to_csv(p2, index=False)
        print(f"  {p1}  ({len(d):,} 對)\n  {p2}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
