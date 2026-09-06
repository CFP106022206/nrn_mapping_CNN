"""三視圖的常駐儲存。

為什麼需要這層：三視圖原本一顆一個 .npz（壓縮），冷快取下讀一個要 ~8 ms。
單次查詢平均有 ~900 個候選，光讀檔就 7 秒，比模型 predict（0.13 s）慢 50 倍。

整個 view DB 解壓後 EM 約 66 MB、FC 約 145 MB，可以整包放進記憶體，
所以這裡把所有圖打包成一個連續的 uint8 buffer + 偏移表，用 memmap 開，
取一張圖就是一次 slice。

打包檔（由 tools/pack_views.py 產生）：
    views_<side>.bin        所有圖串接起來的 uint8
    views_<side>_meta.npz   ids / offsets / h / render_version

圖是 (3, H, H) 正方形（standard_draw 會把三個 view 統一成同一個 grid），
所以每顆只需要記一個 H 和一個 offset。
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np

from swc_util import _load_views_from_npz


class ViewStore:
    """單側（FC 或 EM）的三視圖存取。

    優先用打包好的 memmap；沒有打包檔時自動退回逐檔讀 npz，
    功能一樣，只是慢。
    """

    def __init__(
        self,
        side: str,
        *,
        store_dir: Path | str | None = None,
        npz_dir: Path | str | None = None,
        render_version: str | None = None,
        check_stale: bool = True,
    ) -> None:
        self.side = str(side).upper()
        self.npz_dir = Path(npz_dir) if npz_dir is not None else None
        self._buf: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._h: np.ndarray | None = None
        self._index: dict[str, int] = {}
        self.render_version: str | None = None
        self.packed = False
        self.packed_at: float = 0.0
        self.packed_count: int = 0
        self.stale_warning: str | None = None

        if store_dir is not None:
            bin_path = Path(store_dir) / f"views_{self.side}.bin"
            meta_path = Path(store_dir) / f"views_{self.side}_meta.npz"
            if bin_path.exists() and meta_path.exists():
                self._load_packed(bin_path, meta_path)

        if not self.packed and self.npz_dir is None:
            raise FileNotFoundError(
                f"{self.side}: 既沒有打包檔（{store_dir}）也沒有給 npz_dir，無法讀取三視圖"
            )

        if (
            self.packed
            and render_version is not None
            and self.render_version is not None
            and self.render_version != render_version
        ):
            raise ValueError(
                f"{self.side} view store 的 render_version={self.render_version!r} "
                f"與目前設定 {render_version!r} 不符。請重跑 tools/pack_views.py"
            )

        if self.packed and self.npz_dir is not None and check_stale:
            self._check_stale()

    def _check_stale(self) -> None:
        """偵測「三視圖已經更新但忘了重跑 pack_views」。

        只做便宜的檢查（數檔案數 + 目錄 mtime，約 7 ms），不逐檔比對內容。
        會抓到：新增、刪除三視圖。
        抓不到：檔案數不變、原地覆寫既有檔案的情況 ——
                那種情況打包檔會安靜地回傳舊圖。要完全確認請跑
                `python3 tools/pack_views.py --check`。
        """
        try:
            n_now = sum(1 for e in os.scandir(self.npz_dir) if e.name.endswith("_views.npz"))
            dir_mtime = os.stat(self.npz_dir).st_mtime
        except OSError:
            return

        msgs = []
        if self.packed_count and n_now != self.packed_count:
            msgs.append(f"打包時 {self.packed_count} 顆，現在目錄裡有 {n_now} 顆")
        if self.packed_at and dir_mtime > self.packed_at:
            msgs.append("三視圖目錄在打包之後有被改過")

        if msgs:
            self.stale_warning = (
                f"{self.side} view store 可能已過期（{'；'.join(msgs)}）。"
                f"沒被打包到的神經元會自動退回讀 npz（正確但較慢）；"
                f"若是既有的圖被重畫過，打包檔會回傳舊圖。建議重跑："
                f" python3 tools/pack_views.py --side {self.side}"
            )
            warnings.warn(self.stale_warning, RuntimeWarning, stacklevel=3)

    def _load_packed(self, bin_path: Path, meta_path: Path) -> None:
        with np.load(meta_path, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            self._offsets = z["offsets"].astype(np.int64)
            self._h = z["h"].astype(np.int64)
            if "render_version" in z.files:
                self.render_version = str(z["render_version"])
            if "packed_at" in z.files:
                self.packed_at = float(z["packed_at"])
            if "packed_count" in z.files:
                self.packed_count = int(z["packed_count"])
        self._index = {k: i for i, k in enumerate(ids)}
        self._buf = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.packed = True

    def __len__(self) -> int:
        if self.packed:
            return len(self._index)
        return len(list(self.npz_dir.glob("*_views.npz"))) if self.npz_dir else 0

    def __contains__(self, neuron_id: str) -> bool:
        if self.packed and str(neuron_id) in self._index:
            return True
        if self.npz_dir is not None:
            return (self.npz_dir / f"{neuron_id}_views.npz").exists()
        return False

    def ids(self) -> list[str]:
        if self.packed:
            return list(self._index.keys())
        if self.npz_dir is None:
            return []
        return [p.name[: -len("_views.npz")] for p in sorted(self.npz_dir.glob("*_views.npz"))]

    def get(self, neuron_id: str) -> np.ndarray | None:
        """回傳 (3, H, H) uint8，找不到回 None。回傳的是唯讀 view，不要就地修改。"""
        nid = str(neuron_id)
        if self.packed:
            i = self._index.get(nid)
            if i is not None:
                h = int(self._h[i])
                off = int(self._offsets[i])
                return self._buf[off : off + 3 * h * h].reshape(3, h, h)
        if self.npz_dir is not None:
            p = self.npz_dir / f"{nid}_views.npz"
            if p.exists():
                return _load_views_from_npz(p)
        return None

    def get_many(self, neuron_ids: list[str]) -> tuple[list[str], list[np.ndarray], list[str]]:
        """批次取圖。回傳 (找到的 id, 對應的圖, 找不到的 id)。"""
        found_ids: list[str] = []
        found_views: list[np.ndarray] = []
        missing: list[str] = []
        for nid in neuron_ids:
            v = self.get(nid)
            if v is None:
                missing.append(nid)
            else:
                found_ids.append(nid)
                found_views.append(v)
        return found_ids, found_views, missing


def pack_side(
    npz_dir: Path | str,
    out_dir: Path | str,
    side: str,
    *,
    render_version: str = "",
    verbose: bool = True,
) -> tuple[Path, Path]:
    """把一整個目錄的 *_views.npz 打包成 bin + meta。"""
    npz_dir = Path(npz_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    side = str(side).upper()

    paths = sorted(npz_dir.glob("*_views.npz"))
    if not paths:
        raise FileNotFoundError(f"{npz_dir} 底下找不到任何 *_views.npz")

    ids: list[str] = []
    hs: list[int] = []
    offsets: list[int] = []
    chunks: list[np.ndarray] = []
    off = 0
    skipped: list[str] = []

    for k, p in enumerate(paths, start=1):
        nid = p.name[: -len("_views.npz")]
        v = _load_views_from_npz(p)
        v = np.ascontiguousarray(v, dtype=np.uint8)
        if v.ndim != 3 or v.shape[0] != 3 or v.shape[1] != v.shape[2]:
            skipped.append(f"{nid}:shape{v.shape}")
            continue
        h = int(v.shape[1])
        ids.append(nid)
        hs.append(h)
        offsets.append(off)
        chunks.append(v.reshape(-1))
        off += 3 * h * h
        if verbose and (k % 5000 == 0 or k == len(paths)):
            print(f"  [{k}/{len(paths)}] packed, {off / 1e6:.1f} MB", flush=True)

    bin_path = out_dir / f"views_{side}.bin"
    meta_path = out_dir / f"views_{side}_meta.npz"

    buf = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.uint8)
    buf.tofile(bin_path)
    np.savez(
        meta_path,
        ids=np.asarray(ids, dtype="U64"),
        offsets=np.asarray(offsets, dtype=np.int64),
        h=np.asarray(hs, dtype=np.int64),
        render_version=np.str_(render_version),
        # 給 ViewStore 用來偵測「三視圖更新了但忘記重跑打包」
        packed_at=np.float64(time.time()),
        packed_count=np.int64(len(ids)),
    )

    if verbose:
        print(f"[pack] {side}: {len(ids)} neurons, {off / 1e6:.1f} MB -> {bin_path}")
        if skipped:
            print(f"[pack] {side}: 跳過 {len(skipped)} 個形狀異常的檔案: {skipped[:5]}")
    return bin_path, meta_path


def verify_side(npz_dir: Path | str, store_dir: Path | str, side: str, *, verbose: bool = True) -> bool:
    """逐檔比對打包檔與原始 npz，確認完全一致。

    startup 時的便宜檢查抓不到「原地覆寫既有檔案」的情況，這支可以。
    整個 FC 約 2 分鐘。
    """
    npz_dir = Path(npz_dir)
    side = str(side).upper()
    store = ViewStore(side, store_dir=store_dir, npz_dir=npz_dir, check_stale=False)
    if not store.packed:
        print(f"[verify] {side}: 沒有打包檔")
        return False

    paths = sorted(npz_dir.glob("*_views.npz"))
    disk_ids = {p.name[: -len("_views.npz")] for p in paths}
    packed_ids = set(store.ids())

    missing = disk_ids - packed_ids      # 磁碟上有、打包檔沒有 -> 會退回讀 npz
    orphan = packed_ids - disk_ids       # 打包檔有、磁碟上已刪
    mismatch: list[str] = []

    for k, p in enumerate(paths, start=1):
        nid = p.name[: -len("_views.npz")]
        if nid not in packed_ids:
            continue
        if not np.array_equal(store.get(nid), _load_views_from_npz(p)):
            mismatch.append(nid)
        if verbose and (k % 5000 == 0 or k == len(paths)):
            print(f"  [{k}/{len(paths)}] checked", flush=True)

    ok = not (missing or orphan or mismatch)
    print(
        f"[verify] {side}: 打包 {len(packed_ids)} 顆 / 磁碟 {len(disk_ids)} 顆"
        f"  未打包 {len(missing)}  已刪除但仍在打包檔 {len(orphan)}  內容不符 {len(mismatch)}"
    )
    for label, lst in (("未打包", missing), ("孤兒", orphan), ("內容不符", mismatch)):
        if lst:
            print(f"    {label} 範例: {sorted(lst)[:5]}")
    print(f"[verify] {side}: {'一致 ✓' if ok else '需要重跑 tools/pack_views.py ✗'}")
    return ok
