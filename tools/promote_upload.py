"""把人工確認過的上傳檔並入 curated 資料庫。

這是**唯一**會寫入 data/ 的入口。服務本身永遠不會動 curated 資料庫，
所以未經確認的上傳檔不可能出現在任何人的候選名單裡。

    # 看有哪些待確認的上傳
    python3 tools/promote_upload.py --list

    # 檢視單一筆
    python3 tools/promote_upload.py --show <upload_id>

    # 確認並並入（會複製 SWC 與三視圖到 data/）
    python3 tools/promote_upload.py --approve <upload_id> [--neuron_id 指定入庫用的 id]

    # 拒絕
    python3 tools/promote_upload.py --reject <upload_id> --note "理由"

approve 之後 descriptor 還沒進資料庫（descriptor 存成位置對齊的 .npy，
不能單筆 append），要再跑一次：

    python3 swc_descriptor_batch.py --input ./data/SWC/<SIDE> --out ./data/descriptors_<SIDE> --source <SIDE>
    python3 tools/pack_views.py --side <SIDE>
    python3 tools/build_curated_index.py --side <SIDE>

然後重啟服務（或重新建立 NeuronMatchService）讓 KDTree 與 view store 生效。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrn_service.config import ServiceConfig  # noqa: E402
from nrn_service.upload_store import UploadStore  # noqa: E402
from nrn_service.validation import ValidationConfig, sanitize_neuron_id  # noqa: E402


def cmd_list(store: UploadStore, status: str | None) -> None:
    df = store.list_uploads(status=status, limit=200)
    if df.empty:
        print("(沒有紀錄)")
        return
    cols = ["upload_id", "neuron_id", "resolved_id", "side", "n_nodes", "status", "in_curated", "created_at"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


def cmd_show(store: UploadStore, upload_id: str) -> None:
    rec = store.get(upload_id)
    if not rec:
        print(f"找不到 upload_id={upload_id}")
        return
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    d = store.upload_dir(upload_id)
    print(f"\n檔案（{d}）:")
    for p in sorted(d.glob("*")):
        print(f"  {p.name:20s} {p.stat().st_size:>10,} bytes")
    res = store.result_path(upload_id)
    if res.exists():
        print(f"\n結果 ({res}):")
        print(res.read_text().strip())


def cmd_approve(store: UploadStore, cfg: ServiceConfig, upload_id: str, neuron_id: str | None) -> int:
    rec = store.get(upload_id)
    if not rec:
        print(f"找不到 upload_id={upload_id}", file=sys.stderr)
        return 2
    if rec["in_curated"]:
        print(f"{upload_id} 的內容已經在 curated 資料庫中（{rec['curated_id']}），不需要並入")
        return 0

    side = str(rec["side"])
    target_id = sanitize_neuron_id(neuron_id or str(rec["resolved_id"]), ValidationConfig())

    swc_src = store.swc_path(upload_id)
    views_src = store.views_path(upload_id)
    if not swc_src.exists():
        print(f"找不到原始 SWC: {swc_src}", file=sys.stderr)
        return 2

    swc_dst = cfg.paths.swc_dir(side) / f"{target_id}.swc"
    views_dst = cfg.paths.views_dir(side) / f"{target_id}_views.npz"

    if swc_dst.exists():
        print(f"data/ 中已經有同名的神經元: {swc_dst}。請改用 --neuron_id 指定別的 id", file=sys.stderr)
        return 2

    swc_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(swc_src, swc_dst)
    print(f"[promote] SWC   -> {swc_dst}")

    if views_src.exists():
        with np.load(views_src, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files}
        payload["nid"] = np.str_(target_id)
        views_dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(views_dst, **payload)
        print(f"[promote] views -> {views_dst}")
    else:
        print("[promote] 這筆沒有存三視圖，稍後重跑 standard_draw 補畫")

    store.set_status(upload_id, "promoted", f"promoted as {side}/{target_id}")
    print(
        f"\n[promote] 完成。接著要重建索引才會真的被搜尋到：\n"
        f"  python3 swc_descriptor_batch.py --input ./data/SWC/{side} "
        f"--out ./data/descriptors_{side} --source {side}\n"
        f"  python3 tools/pack_views.py --side {side}\n"
        f"  python3 tools/build_curated_index.py --side {side}\n"
        f"然後重啟服務。"
    )
    return 0


def cmd_reject(store: UploadStore, upload_id: str, note: str) -> int:
    if not store.get(upload_id):
        print(f"找不到 upload_id={upload_id}", file=sys.stderr)
        return 2
    store.set_status(upload_id, "rejected", note)
    print(f"[reject] {upload_id} 已標記為 rejected")
    return 0


def main() -> int:
    cfg = ServiceConfig()
    ap = argparse.ArgumentParser(description="Review and promote user uploads into the curated DB.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="列出上傳紀錄")
    g.add_argument("--show", metavar="UPLOAD_ID")
    g.add_argument("--approve", metavar="UPLOAD_ID")
    g.add_argument("--reject", metavar="UPLOAD_ID")
    ap.add_argument("--status", default=None, help="搭配 --list 過濾狀態")
    ap.add_argument("--neuron_id", default=None, help="搭配 --approve，指定入庫時使用的 id")
    ap.add_argument("--note", default="", help="搭配 --reject 的理由")
    args = ap.parse_args()

    store = UploadStore(cfg.paths.user_data_root)
    if args.list:
        cmd_list(store, args.status)
        return 0
    if args.show:
        cmd_show(store, args.show)
        return 0
    if args.approve:
        return cmd_approve(store, cfg, args.approve, args.neuron_id)
    return cmd_reject(store, args.reject, args.note)


if __name__ == "__main__":
    raise SystemExit(main())
