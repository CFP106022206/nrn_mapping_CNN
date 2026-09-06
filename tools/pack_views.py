"""把 data/standard_views/{FC,EM} 打包成常駐用的 view store。

    python3 tools/pack_views.py                 # 兩側都打包
    python3 tools/pack_views.py --side EM       # 只打包單側

打包後服務啟動時用 memmap 開，取圖是 O(1) slice，不再逐檔讀 npz。

⚠️ 資料庫的三視圖有變動（補畫、重畫、刪除）之後一定要重跑這支。
   服務啟動時會做便宜的檢查（數檔案數 + 目錄 mtime）並在過期時發出警告，
   但抓不到「檔案數不變、原地覆寫」的情況 —— 那種狀況打包檔會安靜地回傳舊圖。
   要完全確認請跑：

       python3 tools/pack_views.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrn_service.config import SIDES, ServiceConfig  # noqa: E402
from nrn_service.view_store import pack_side, verify_side  # noqa: E402


def main() -> None:
    cfg = ServiceConfig()
    ap = argparse.ArgumentParser(description="Pack standard views into a memmap store.")
    ap.add_argument("--side", choices=list(SIDES), default=None, help="只打包單側，預設兩側都做")
    ap.add_argument("--views_root", default=str(cfg.paths.views_root))
    ap.add_argument("--out_dir", default=str(cfg.paths.view_store_dir))
    ap.add_argument("--check", action="store_true",
                    help="不重新打包，只逐檔比對打包檔與原始 npz 是否一致（FC 約 2 分鐘）")
    args = ap.parse_args()

    sides = [args.side] if args.side else list(SIDES)

    if args.check:
        all_ok = True
        for side in sides:
            print(f"[verify] {side} ...", flush=True)
            all_ok &= verify_side(Path(args.views_root) / side, args.out_dir, side)
        raise SystemExit(0 if all_ok else 1)

    for side in sides:
        print(f"[pack] {side} ...", flush=True)
        pack_side(
            Path(args.views_root) / side,
            args.out_dir,
            side,
            render_version=cfg.render.render_version,
        )


if __name__ == "__main__":
    main()
