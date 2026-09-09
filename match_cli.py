"""命令列介面：跑比對並輸出 CSV。不需要安裝任何 web 套件。

單筆查詢
--------
    # 上傳一個新的 SWC，指定它是 FC 側，去 EM 資料庫找
    python3 match_cli.py --swc /path/to/neuron.swc --query_side FC --out result.csv

    # 反方向
    python3 match_cli.py --swc /path/to/body.swc --query_side EM --out result.csv

    # 不上傳檔案，直接用資料庫裡已有的神經元當查詢對象
    python3 match_cli.py --neuron_id Trh-F-000040 --query_side FC --out result.csv

批次掃描
--------
    # 掃描 curated 資料庫裡「全部」的 FC 神經元，各找 top 5 EM
    # python3 match_cli.py --batch_side FC --top_n 5 --out result/fc_all_top5_annotator.csv > logs/batch_fc_top5.log 2>&1 &
    # python3 match_cli.py --batch_side EM --top_n 5 --out result/em_all_top5_annotator.csv > logs/batch_em_top5.log 2>&1 &

    python3 match_cli.py --batch_side FC --top_n 5 --top_k 0 --out result/fc_all_top5_notrunc.csv > logs/batch_fc_notrunc.log 2>&1 &
    python3 match_cli.py --batch_side EM --top_n 5 --top_k 0 --out result/em_all_top5_notrunc.csv > logs/batch_em_notrunc.log 2>&1 &
    # top_k=0 代表不截斷候選，否則預設只取前 2000 個候選（線上查詢的延遲護欄），
    
    # 先試 200 顆確認沒問題再跑全部
    python3 match_cli.py --batch_side FC --limit 200 --out trial.csv

    # 取消候選截斷（--top_k 0）。config 的 top_k_candidates 是線上查詢的延遲護欄，
    # 離線全庫掃描沒有延遲壓力，但它會改變輸出：候選被截掉時，第 2001 名之後的
    # 會遞補進來打分，所以截斷與否的結果不同（EM->FC 約 53% 的 source 會受影響）。
    # 用這個參數跑對照，不要去改 config —— 改了忘記還原，線上服務就沒有延遲上限了。
    python3 match_cli.py --batch_side FC --top_n 5 --top_k 0 --out result/fc_all_top5_notrunc.csv
    python3 match_cli.py --batch_side EM --top_n 5 --top_k 0 --out result/em_all_top5_notrunc.csv

    # 掃描一整個目錄的新 SWC（會走完整計算路徑，並在 user_data/ 留紀錄）
    python3 match_cli.py --batch_dir /path/to/swc_folder --query_side FC --out result.csv

    ⚠️ 不要用 shell 迴圈逐檔呼叫這支程式：每次啟動都要重新載入模型（約 3 秒），
       28612 顆會變成 24 小時。批次模式在同一個 process 裡跑完，全庫約 85 分鐘。

輸出 CSV 欄位：source_id, target_id, similarity_score, rank
批次模式另外輸出一份 report CSV（預設是主檔名加 _report），欄位：
    source_id, status, n_candidates, n_scored, top1_score, elapsed_seconds, note
    status = ok / no_candidate / low_score / error
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import pandas as pd

from nrn_service.config import RESULT_COLUMNS, SIDES, ServiceConfig
from nrn_service.service import NeuronMatchService
from nrn_service.validation import ValidationError

REPORT_COLUMNS = (
    "source_id", "status", "n_candidates", "n_scored",
    "top1_score", "elapsed_seconds", "note",
)


def _report_path(out_path: str) -> Path:
    p = Path(out_path)
    return p.with_name(f"{p.stem}_report{p.suffix or '.csv'}")


def run_batch(
    svc: NeuronMatchService,
    *,
    batch_side: str | None,
    batch_dir: str | None,
    query_side: str | None,
    target_side: str | None,
    top_n: int,
    limit: int,
    min_score: float,
    out_path: str,
) -> int:
    """對一整批神經元跑比對，邊跑邊寫檔（中途中斷不會全部白做）。"""
    if batch_side:
        side = batch_side
        items = [(nid, None) for nid in svc.db[side].neuron_ids.tolist()]
        source_desc = f"curated {side} 資料庫"
    else:
        if not query_side:
            print("[error] --batch_dir 必須同時指定 --query_side", file=sys.stderr)
            return 2
        side = query_side
        files = sorted(Path(batch_dir).glob("*.swc"))
        if not files:
            print(f"[error] {batch_dir} 底下找不到任何 .swc", file=sys.stderr)
            return 2
        items = [(p.stem, p) for p in files]
        source_desc = f"目錄 {batch_dir}"

    if limit > 0:
        items = items[:limit]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = _report_path(out_path)
    # 從頭開始寫，先落下表頭
    pd.DataFrame(columns=list(RESULT_COLUMNS)).to_csv(out, index=False)
    pd.DataFrame(columns=list(REPORT_COLUMNS)).to_csv(rep, index=False)

    print(
        f"[batch] 掃描 {source_desc}，共 {len(items)} 顆，"
        f"{side} -> {target_side or ('EM' if side == 'FC' else 'FC')}，每顆取 top {top_n}",
        flush=True,
    )
    if min_score > 0:
        print(f"[batch] 只保留分數 >= {min_score} 的結果", flush=True)

    rows: list[pd.DataFrame] = []
    reports: list[dict] = []
    counts = {"ok": 0, "no_candidate": 0, "low_score": 0, "error": 0}
    n_pairs = 0
    t_start = time.time()

    def flush() -> None:
        nonlocal rows, reports
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(out, mode="a", header=False, index=False)
            rows = []
        if reports:
            pd.DataFrame(reports, columns=list(REPORT_COLUMNS)).to_csv(
                rep, mode="a", header=False, index=False
            )
            reports = []

    for k, (nid, path) in enumerate(items, start=1):
        t0 = time.perf_counter()
        try:
            r = svc.query(
                swc_path=str(path) if path is not None else None,
                neuron_id=None if path is not None else nid,
                query_side=side,
                target_side=target_side,
                top_n=top_n,
            )
        except Exception as e:  # 單顆失敗不中斷整批
            counts["error"] += 1
            reports.append({
                "source_id": nid, "status": "error", "n_candidates": 0, "n_scored": 0,
                "top1_score": "", "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "note": f"{type(e).__name__}: {e}",
            })
            continue

        elapsed = round(time.perf_counter() - t0, 3)
        if r.is_empty:
            counts["no_candidate"] += 1
            reports.append({
                "source_id": r.source_id, "status": "no_candidate",
                "n_candidates": r.n_candidates, "n_scored": r.n_scored,
                "top1_score": "", "elapsed_seconds": elapsed,
                "note": "; ".join(r.warnings),
            })
            continue

        table = r.table
        top1 = float(table.iloc[0]["similarity_score"])
        if min_score > 0:
            table = table[table["similarity_score"] >= min_score]

        if table.empty:
            counts["low_score"] += 1
            status = "low_score"
        else:
            counts["ok"] += 1
            status = "ok"
            rows.append(table)
            n_pairs += len(table)

        reports.append({
            "source_id": r.source_id, "status": status,
            "n_candidates": r.n_candidates, "n_scored": r.n_scored,
            "top1_score": round(top1, 6), "elapsed_seconds": elapsed,
            "note": "; ".join(r.warnings),
        })

        if k % 200 == 0 or k == len(items):
            flush()
            done = time.time() - t_start
            eta = done / k * (len(items) - k)
            print(
                f"  [{k}/{len(items)}] ok={counts['ok']} 無候選={counts['no_candidate']} "
                f"低分={counts['low_score']} 失敗={counts['error']}  "
                f"{done / k * 1000:.0f} ms/顆  剩餘約 {eta / 60:.0f} 分",
                flush=True,
            )

    flush()
    total = time.time() - t_start
    print(
        f"\n[batch] 完成，耗時 {total / 60:.1f} 分（{total / max(len(items), 1) * 1000:.0f} ms/顆）"
        f"\n  有結果       {counts['ok']:6d} 顆  -> {n_pairs} 對"
        f"\n  初篩無候選   {counts['no_candidate']:6d} 顆"
        f"\n  分數未達門檻 {counts['low_score']:6d} 顆"
        f"\n  失敗         {counts['error']:6d} 顆"
        f"\n  結果 CSV: {out}"
        f"\n  報表 CSV: {rep}",
        flush=True,
    )
    return 0 if counts["error"] == 0 else 1


def main() -> int:
    cfg = ServiceConfig()
    ap = argparse.ArgumentParser(description="Find the most similar neurons in the other database.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--swc", help="要查詢的 SWC 檔案路徑")
    src.add_argument("--neuron_id", help="改用 curated 資料庫裡既有的神經元 id")
    src.add_argument("--batch_side", choices=list(SIDES),
                     help="批次：掃描 curated 資料庫中該側的全部神經元")
    src.add_argument("--batch_dir",
                     help="批次：掃描一整個目錄的 .swc（需搭配 --query_side）")

    ap.add_argument("--query_side", choices=list(SIDES), default=None,
                    help="查詢的這顆神經元屬於哪一側（--batch_side 模式不需要）")
    ap.add_argument("--target_side", choices=list(SIDES), default=None,
                    help="要搜尋哪個資料庫，預設是另一側")
    ap.add_argument("--top_n", type=int, default=cfg.default_top_n,
                    help=f"輸出幾對（預設 {cfg.default_top_n}）")
    ap.add_argument("--out", default="result.csv", help="輸出 CSV 路徑")
    ap.add_argument("--mirror", action="store_true",
                    help="左右腦鏡像（尚未實作，見 NeuronMatchService.query 說明）")
    ap.add_argument("--limit", type=int, default=0,
                    help="批次模式：只跑前 N 顆，用來先試跑（0 = 全部）")
    ap.add_argument("--min_score", type=float, default=0.0,
                    help="批次模式：只保留分數 >= 此值的結果（0 = 不過濾，與既有 pipeline 一致）")
    ap.add_argument("--top_k", type=int, default=None,
                    help=f"覆蓋候選上限 MatchConfig.top_k_candidates（0 = 不截斷）。"
                         f"預設沿用 config 的 {cfg.match.top_k_candidates}")
    args = ap.parse_args()

    if not (args.batch_side or args.query_side):
        ap.error("--swc / --neuron_id 模式必須指定 --query_side")

    # 候選上限是「線上單次查詢的延遲護欄」，離線全庫掃描沒有延遲壓力，
    # 但它會改變輸出：候選被截掉時，第 top_k+1 名之後的遞補進來打分。
    # 所以這裡用命令列覆蓋，而不是改 config —— config 的線上預設值保持不動。
    if args.top_k is not None:
        if args.top_k < 0:
            ap.error("--top_k 不能是負數（0 = 不截斷）")
        cfg = dataclasses.replace(
            cfg, match=dataclasses.replace(cfg.match, top_k_candidates=args.top_k)
        )

    svc = NeuronMatchService(cfg)
    top_k = cfg.match.top_k_candidates
    print(
        f"[service] 啟動 {svc.startup_timings['total']:.1f}s  "
        f"({svc.db.summary()})  "
        f"rod={cfg.match.rod_angle_th_deg:.0f}° disk={cfg.match.disk_angle_th_deg:.0f}° "
        f"top_k={'不截斷' if top_k == 0 else top_k}",
        flush=True,
    )

    if args.batch_side or args.batch_dir:
        return run_batch(
            svc,
            batch_side=args.batch_side,
            batch_dir=args.batch_dir,
            query_side=args.query_side,
            target_side=args.target_side,
            top_n=args.top_n,
            limit=args.limit,
            min_score=args.min_score,
            out_path=args.out,
        )

    try:
        result = svc.query(
            swc_path=args.swc,
            neuron_id=args.neuron_id,
            query_side=args.query_side,
            target_side=args.target_side,
            top_n=args.top_n,
            mirror=args.mirror,
        )
    except ValidationError as e:
        print(f"[error] 上傳檔案無法處理: {e}", file=sys.stderr)
        return 2
    except NotImplementedError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    for w in result.warnings:
        print(f"[warn] {w}", flush=True)

    print(
        f"[service] source={result.source_id} {result.query_side} -> {result.target_side}"
        f"  候選 {result.n_candidates} / 打分 {result.n_scored}"
        f"  ({result.timings.get('total', 0):.2f}s, {result.resolution})",
        flush=True,
    )

    if result.is_empty:
        print("[service] 找不到任何候選，沒有輸出結果", file=sys.stderr)
        return 1

    path = svc.write_csv(result, args.out)
    print(result.table.to_string(index=False))
    print(f"[service] 已寫入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
