"""把每顆 FC 的**整個候選池**分數 dump 出來（不只 top-5）。

per-EM 去偏需要每顆 EM 的分數基線，而 `result/fc_all_top5*.csv` 只存前 5 名：
只有當某顆 EM 擠進某人的前五時才看得到它的分數，樣本被嚴重截斷，估不出基線。

`NeuronMatchService.query()` 其實已經算完整個池，只是 `table` 被 `top_n` 截斷，
`full_table` 保留全部（`nrn_service/service.py:474`）。所以這裡只要拿 `full_table`，
不需要改動 `match_cli.py` 或服務本身。

輸出：`scores/pool_scores_{model_key}.parquet`
      欄位 fc_id / em_id / score

執行：
    python3 analysis_hubness_debias/dump_pool_scores.py --model finetune
    python3 analysis_hubness_debias/dump_pool_scores.py --model annotator
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

SCORES = ROOT / "scores"

MODELS = {
    "finetune": ("./FineTune_Model/FineTune_miniLR_D1-D6_0.weights.h5",
                 "FineTune_miniLR_D1-D6_0"),
    "annotator": ("./Annotator_Model/Annotator_D1-D6_0.weights.h5",
                  "Annotator_D1-D6_0"),
}

# 掃描對象：外部型別驗證可評估的那批 FC。比 winnable 的 1 276 顆多，
# 好處是可以用「與評估集不相交的 FC」另外估一版基線當對照（見 debias.py）。
FC_LIST = PROJECT / "analysis_external_validation" / "results" / "pair_labels.csv"


def fc_ids() -> list[str]:
    d = pd.read_csv(FC_LIST, usecols=["fc_id"])
    return sorted(d.fc_id.unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 顆，用來試跑")
    ap.add_argument("--flush-every", type=int, default=200)
    args = ap.parse_args()

    from nrn_service.config import ServiceConfig
    from nrn_service.service import NeuronMatchService

    weights, model_id = MODELS[args.model]
    cfg = ServiceConfig()
    cfg = dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, weights=weights, model_id=model_id)
    )
    svc = NeuronMatchService(cfg)

    ids = fc_ids()
    if args.limit:
        ids = ids[: args.limit]
    print(f"[dump] model={model_id}  FC={len(ids)}", flush=True)

    SCORES.mkdir(parents=True, exist_ok=True)
    out = SCORES / f"pool_scores_{args.model}.parquet"
    tmp_dir = SCORES / f"_parts_{args.model}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    buf: list[pd.DataFrame] = []
    parts: list[Path] = []
    n_pairs = 0
    n_err = 0
    t0 = time.time()

    def flush(tag: int) -> None:
        nonlocal buf
        if not buf:
            return
        p = tmp_dir / f"part_{tag:05d}.parquet"
        pd.concat(buf, ignore_index=True).to_parquet(p, index=False)
        parts.append(p)
        buf = []

    for k, fc in enumerate(ids, start=1):
        try:
            r = svc.query(neuron_id=fc, query_side="FC", target_side="EM", top_n=0)
        except Exception as e:
            n_err += 1
            print(f"  ! {fc}: {type(e).__name__}: {e}", flush=True)
            continue
        t = r.full_table
        if t.empty:
            continue
        buf.append(
            pd.DataFrame(
                {
                    "fc_id": t.source_id.astype(str),
                    "em_id": t.target_id.astype("int64"),
                    "score": t.similarity_score.astype("float32"),
                }
            )
        )
        n_pairs += len(t)
        if k % args.flush_every == 0:
            flush(k)
            el = time.time() - t0
            print(f"  [{k}/{len(ids)}] pairs={n_pairs:,} "
                  f"{el:.0f}s  {n_pairs/max(el,1e-9):,.0f} pair/s", flush=True)
    flush(len(ids) + 1)

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["fc_id"] = df.fc_id.astype("category")
    df.to_parquet(out, index=False)
    for p in parts:
        p.unlink()
    tmp_dir.rmdir()

    print(f"[dump] -> {out}  {len(df):,} 對  "
          f"FC={df.fc_id.nunique()}  EM={df.em_id.nunique()}  "
          f"錯誤 {n_err} 顆  耗時 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
