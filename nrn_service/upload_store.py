"""使用者上傳檔的暫存區。

設計原則（依討論定案）：
  * curated 資料庫（data/）在服務執行期間完全唯讀。
  * 上傳的檔案一律放在 user_data/，**不會**進入任何人的候選名單 ——
    也就是說別人查詢時絕對比對不到未確認的上傳檔。
  * 要讓一顆上傳的神經元變成「可被搜尋的目標」，必須經過人工確認，
    走 tools/promote_upload.py 明確 approve，那支才會寫進 data/。
  * 目錄用內容雜湊命名，純粹是為了「同一個檔案不要重算兩次」以及節省空間，
    重用範圍限在同一筆 upload 紀錄內，不做跨使用者的結果共享。

目錄結構：
    user_data/
      uploads.sqlite
      <sha256 前16碼>/
        meta.json          原始檔名、side、雜湊、節點數、bbox、狀態
        neuron.swc         原始上傳檔，不做任何修改
        descriptor.npz     centroid / ratios2d / eigvecs
        views.npz          三視圖（與 standard_views 同格式，含 render_version）
        result_full.csv    全部候選的排名結果（快取來源）
        result.csv         回給使用者的前 n 筆
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

UPLOAD_ID_LEN = 16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    upload_id      TEXT PRIMARY KEY,
    sha256         TEXT NOT NULL,
    orig_filename  TEXT,
    neuron_id      TEXT,
    resolved_id    TEXT,
    side           TEXT NOT NULL,
    n_nodes        INTEGER,
    bbox           TEXT,
    status         TEXT NOT NULL,
    render_version TEXT,
    in_curated     INTEGER DEFAULT 0,
    curated_id     TEXT,
    note           TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uploads_sha ON uploads(sha256);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);

CREATE TABLE IF NOT EXISTS results (
    upload_id      TEXT NOT NULL,
    target_side    TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    render_version TEXT NOT NULL,
    n_candidates   INTEGER,
    n_scored       INTEGER,
    csv_path       TEXT,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (upload_id, target_side, model_id, render_version)
);
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class UploadRecord:
    upload_id: str
    sha256: str
    orig_filename: str
    neuron_id: str
    resolved_id: str
    side: str
    n_nodes: int
    bbox: dict
    status: str
    render_version: str
    in_curated: bool = False
    curated_id: str | None = None
    note: str = ""

    @property
    def dir_name(self) -> str:
        return self.upload_id


class UploadStore:
    """user_data/ 的讀寫入口。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "uploads.sqlite"

        # 長駐單一連線：每次操作重開連線並設 PRAGMA 要 150~200 ms，
        # 對「一次查詢只花 1 秒」的服務來說是最大的固定成本。
        # WAL + synchronous=NORMAL 讓寫入不必每次 fsync，同時仍可多執行緒共用。
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---------- 路徑 ----------

    def upload_dir(self, upload_id: str) -> Path:
        return self.root / str(upload_id)

    def swc_path(self, upload_id: str) -> Path:
        return self.upload_dir(upload_id) / "neuron.swc"

    def descriptor_path(self, upload_id: str) -> Path:
        return self.upload_dir(upload_id) / "descriptor.npz"

    def views_path(self, upload_id: str) -> Path:
        return self.upload_dir(upload_id) / "views.npz"

    def result_full_path(self, upload_id: str) -> Path:
        return self.upload_dir(upload_id) / "result_full.csv"

    def result_path(self, upload_id: str) -> Path:
        return self.upload_dir(upload_id) / "result.csv"

    # ---------- 寫入 ----------

    @staticmethod
    def make_upload_id(sha256: str) -> str:
        return str(sha256)[:UPLOAD_ID_LEN]

    def save_raw(self, data: bytes, sha256: str) -> Path:
        """把原始上傳檔落地（同內容已存在就不重寫）。"""
        upload_id = self.make_upload_id(sha256)
        d = self.upload_dir(upload_id)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "neuron.swc"
        if not p.exists():
            p.write_bytes(data)
        return p

    def upsert(self, rec: UploadRecord) -> None:
        d = asdict(rec)
        d["bbox"] = json.dumps(rec.bbox)
        d["in_curated"] = int(rec.in_curated)
        d["created_at"] = _now()
        d["updated_at"] = d["created_at"]
        with self._conn() as c:
            existing = c.execute(
                "SELECT created_at FROM uploads WHERE upload_id=?", (rec.upload_id,)
            ).fetchone()
            if existing:
                d["created_at"] = existing["created_at"]
            c.execute(
                """
                INSERT INTO uploads (upload_id, sha256, orig_filename, neuron_id, resolved_id,
                                     side, n_nodes, bbox, status, render_version,
                                     in_curated, curated_id, note, created_at, updated_at)
                VALUES (:upload_id, :sha256, :orig_filename, :neuron_id, :resolved_id,
                        :side, :n_nodes, :bbox, :status, :render_version,
                        :in_curated, :curated_id, :note, :created_at, :updated_at)
                ON CONFLICT(upload_id) DO UPDATE SET
                    orig_filename=excluded.orig_filename,
                    neuron_id=excluded.neuron_id,
                    resolved_id=excluded.resolved_id,
                    side=excluded.side,
                    n_nodes=excluded.n_nodes,
                    bbox=excluded.bbox,
                    status=excluded.status,
                    render_version=excluded.render_version,
                    in_curated=excluded.in_curated,
                    curated_id=excluded.curated_id,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                d,
            )
        # meta.json 讓人可以不開 sqlite 就看懂這個目錄是什麼
        meta = dict(d)
        meta["bbox"] = rec.bbox
        (self.upload_dir(rec.upload_id)).mkdir(parents=True, exist_ok=True)
        (self.upload_dir(rec.upload_id) / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def set_status(self, upload_id: str, status: str, note: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE uploads SET status=?, note=?, updated_at=? WHERE upload_id=?",
                (status, note, _now(), str(upload_id)),
            )

    def save_descriptor(self, upload_id: str, centroid, ratios2d, eigvecs) -> Path:
        p = self.descriptor_path(upload_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            centroid=np.asarray(centroid, dtype=np.float32),
            ratios2d=np.asarray(ratios2d, dtype=np.float32),
            eigvecs=np.asarray(eigvecs, dtype=np.float32),
        )
        return p

    def load_descriptor(self, upload_id: str):
        p = self.descriptor_path(upload_id)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            return z["centroid"], z["ratios2d"], z["eigvecs"]

    def save_views(self, upload_id: str, neuron_id: str, views: np.ndarray, render_version: str,
                   scale_um_per_px: float, normalize: str) -> Path:
        p = self.views_path(upload_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        v = np.asarray(views, dtype=np.uint8)
        np.savez_compressed(
            p,
            nid=np.str_(str(neuron_id)),
            views=v,
            grid_size=np.int32(v.shape[1]),
            scale_um_per_px=np.float32(scale_um_per_px),
            normalize=np.str_(str(normalize)),
            render_version=np.str_(str(render_version)),
        )
        return p

    def load_views(self, upload_id: str, render_version: str) -> np.ndarray | None:
        """讀回快取的三視圖；render_version 對不上就當作沒有（強制重畫）。"""
        p = self.views_path(upload_id)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            if "render_version" in z.files and str(z["render_version"]) != str(render_version):
                return None
            return z["views"]

    def save_result(
        self,
        upload_id: str,
        full_df: pd.DataFrame,
        *,
        target_side: str,
        model_id: str,
        render_version: str,
        n_candidates: int,
    ) -> Path:
        p = self.result_full_path(upload_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(p, index=False)
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO results (upload_id, target_side, model_id, render_version,
                                     n_candidates, n_scored, csv_path, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(upload_id, target_side, model_id, render_version) DO UPDATE SET
                    n_candidates=excluded.n_candidates,
                    n_scored=excluded.n_scored,
                    csv_path=excluded.csv_path,
                    created_at=excluded.created_at
                """,
                (
                    str(upload_id), str(target_side), str(model_id), str(render_version),
                    int(n_candidates), int(len(full_df)), str(p), _now(),
                ),
            )
        return p

    def load_cached_result(
        self, upload_id: str, *, target_side: str, model_id: str, render_version: str
    ) -> pd.DataFrame | None:
        """只有 model 和 render 版本都一致才重用，避免端出用舊參數算的分數。"""
        with self._conn() as c:
            row = c.execute(
                """SELECT csv_path FROM results
                   WHERE upload_id=? AND target_side=? AND model_id=? AND render_version=?""",
                (str(upload_id), str(target_side), str(model_id), str(render_version)),
            ).fetchone()
        if not row:
            return None
        p = Path(row["csv_path"])
        if not p.exists():
            return None
        return pd.read_csv(p)

    # ---------- 讀取 ----------

    def get(self, upload_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM uploads WHERE upload_id=?", (str(upload_id),)
            ).fetchone()
        return dict(row) if row else None

    def find_by_sha256(self, sha256: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM uploads WHERE sha256=? ORDER BY created_at LIMIT 1", (str(sha256),)
            ).fetchone()
        return dict(row) if row else None

    def list_uploads(self, status: str | None = None, limit: int = 100) -> pd.DataFrame:
        q = "SELECT * FROM uploads"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY created_at DESC LIMIT ?"
        args = args + (int(limit),)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q, args).fetchall()]
        return pd.DataFrame(rows)
