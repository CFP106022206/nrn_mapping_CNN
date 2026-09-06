"""神經元比對服務的主體。

一次查詢的流程：
    1. 解析查詢對象
         a. 上傳檔 -> 算 sha256 -> 依序比對：上傳暫存區 / curated 內容雜湊 /
            curated 檔名。命中就直接用現成的 descriptor 與三視圖。
         b. 指定 neuron_id -> 直接查 curated 資料庫。
    2. 沒命中就走完整計算：驗證 -> descriptor -> 三視圖。
    3. 對 target 側資料庫做候選初篩（質心 / inertia ratio / 方向性）。
    4. 取候選的三視圖（view store，O(1)），逐對前處理後送模型打分。
    5. 依分數排序、給 rank、取前 n 筆，輸出 CSV。

服務執行期間 data/ 只讀不寫；所有新產生的東西都寫在 user_data/。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from standard_draw import render_single
from swc_descriptor import (
    compute_segment_weighted_centroid_and_inertia_eigen,
    normalized_inertia_ratios,
)

from .config import RESULT_COLUMNS, ServiceConfig, normalize_side, other_side
from .db_index import CuratedDB, NeuronDescriptor
from .matching import match_one
from .scoring import ScoringModel
from .upload_store import UploadRecord, UploadStore, sha256_bytes
from .validation import (
    ValidationError,
    load_swc_or_raise,
    sanitize_neuron_id,
    validate_file_size,
    validate_swc,
    validate_views,
)
from .view_store import ViewStore


@dataclass
class QueryResult:
    """一次查詢的完整結果。"""

    source_id: str
    query_side: str
    target_side: str
    table: pd.DataFrame            # 前 n 筆，欄位 = RESULT_COLUMNS
    full_table: pd.DataFrame       # 全部打過分的候選（含 rank）
    csv_path: Path | None = None
    upload_id: str | None = None
    resolution: str = ""           # 查詢對象是怎麼解析出來的
    n_candidates: int = 0
    n_scored: int = 0
    n_missing_views: int = 0
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.table.empty


def _empty_result_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": pd.Series(dtype=str),
            "target_id": pd.Series(dtype=str),
            "similarity_score": pd.Series(dtype=float),
            "rank": pd.Series(dtype=int),
        }
    )


class NeuronMatchService:
    """常駐服務物件。建構成本高（載模型 + 建 KDTree），整個程序共用一個。"""

    def __init__(self, cfg: ServiceConfig | None = None, *, load_model: bool = True) -> None:
        self.cfg = cfg or ServiceConfig()
        t0 = time.perf_counter()

        self.db = CuratedDB(self.cfg.paths)
        t_db = time.perf_counter()

        self.views: dict[str, ViewStore] = {
            side: ViewStore(
                side,
                store_dir=self.cfg.paths.view_store_dir,
                npz_dir=self.cfg.paths.views_dir(side),
                render_version=None,   # 舊的打包檔沒有版本字串時不強制擋，只在 get 時比對內容
            )
            for side in self.db.sides
        }
        t_views = time.perf_counter()

        self.uploads = UploadStore(self.cfg.paths.user_data_root)

        self.model: ScoringModel | None = None
        if load_model:
            self.model = ScoringModel(self.cfg.model, self.cfg.render, warmup=True)
        t_model = time.perf_counter()

        self.startup_timings = {
            "curated_db": t_db - t0,
            "view_store": t_views - t_db,
            "model": t_model - t_views,
            "total": t_model - t0,
        }

    # ------------------------------------------------------------------
    # descriptor / 三視圖 計算
    # ------------------------------------------------------------------

    def _compute_descriptor(self, swc_path: Path, neuron_id: str) -> NeuronDescriptor:
        swc = load_swc_or_raise(swc_path, neuron_id)
        centroid, eigvals, eigvecs = compute_segment_weighted_centroid_and_inertia_eigen(swc)
        ratios = normalized_inertia_ratios(eigvals)      # [1, r21, r31]
        return NeuronDescriptor(
            neuron_id=neuron_id,
            centroid=centroid.astype(np.float32),
            ratios2d=ratios[1:3].astype(np.float32),
            eigvecs=eigvecs.astype(np.float32),
        )

    def _render_views(self, swc_path: Path) -> np.ndarray:
        """畫三視圖。

        參數一律取自 config，不依賴 standard_draw 的函式預設值 ——
        尺度不一致不會報錯，只會讓分數失去意義。
        render_single 需要「目錄 + id」的形式，所以這裡用檔案所在目錄與 stem。
        """
        r = self.cfg.render
        _nid, views, _grid = render_single(
            swc_path.stem,
            swc_path=swc_path.parent,
            cache={},
            scale_um_per_px=r.scale_um_per_px,
            norm=r.normalize,
        )
        return views

    # ------------------------------------------------------------------
    # 查詢對象解析
    # ------------------------------------------------------------------

    def _resolve_from_curated(
        self, side: str, neuron_id: str
    ) -> tuple[NeuronDescriptor, np.ndarray] | None:
        idx = self.db[side]
        desc = idx.get(neuron_id)
        if desc is None:
            return None
        views = self.views[side].get(neuron_id)
        if views is None:
            return None
        return desc, views

    def _prepare_upload(
        self,
        data: bytes,
        orig_filename: str,
        query_side: str,
    ) -> tuple[str, NeuronDescriptor, np.ndarray, str, list[str]]:
        """處理上傳檔，回傳 (upload_id, descriptor, views, resolution, warnings)。"""
        cfg = self.cfg
        warnings: list[str] = []

        validate_file_size(len(data), cfg.validation)
        name_id = sanitize_neuron_id(orig_filename, cfg.validation)
        sha = sha256_bytes(data)
        upload_id = self.uploads.make_upload_id(sha)

        # --- 這顆其實已經在 curated 資料庫裡嗎？（先比內容，再比檔名）---
        curated_hit = self.db.find_by_sha256(sha)
        resolved_id = name_id
        curated_id: str | None = None
        resolution: str

        if curated_hit is not None:
            hit_side, hit_id = curated_hit
            curated_id = hit_id
            if hit_side != normalize_side(query_side):
                warnings.append(
                    f"這個檔案的內容與 curated 資料庫 {hit_side} 側的 {hit_id} 完全相同，"
                    f"但 query_side 傳的是 {query_side}。服務仍照 query_side={query_side} "
                    f"執行（三視圖餵進模型的 {query_side} 輸入），請確認 side 是否選對"
                )
            resolved_id = hit_id
            if hit_id != name_id:
                warnings.append(f"上傳檔名為 {name_id}，內容比對後對應到資料庫中的 {hit_id}")
            resolution = f"curated_by_sha256:{hit_side}/{hit_id}"
        else:
            idx = self.db[query_side]
            if name_id in idx:
                db_sha = idx.sha256_of(name_id)
                if db_sha is not None and db_sha != sha:
                    # 撞名但內容不同：當成新的神經元，換一個不會撞的 id
                    resolved_id = f"{name_id}__{sha[:8]}"
                    warnings.append(
                        f"檔名 {name_id} 與資料庫中的神經元同名，但內容不同"
                        f"（sha256 不符）。已視為新的神經元，輸出使用 id {resolved_id}"
                    )
                    resolution = "new_upload_name_conflict"
                elif db_sha is None:
                    warnings.append(
                        f"檔名 {name_id} 在資料庫中存在，但沒有 sha256 索引可比對內容，"
                        f"仍以新上傳的檔案計算"
                    )
                    resolution = "new_upload_unverified_name"
                else:
                    # 名稱與內容都一致
                    curated_id = name_id
                    resolution = f"curated_by_name:{name_id}"
            else:
                resolution = "new_upload"

        # --- 落地 ---
        swc_path = self.uploads.save_raw(data, sha)

        swc = load_swc_or_raise(swc_path, str(orig_filename))
        report = validate_swc(swc, cfg.validation)
        warnings.extend(report.warnings)
        rec = UploadRecord(
            upload_id=upload_id,
            sha256=sha,
            orig_filename=str(orig_filename),
            neuron_id=name_id,
            resolved_id=resolved_id,
            side=normalize_side(query_side),
            n_nodes=report.n_nodes,
            bbox={"min": list(report.bbox_min), "max": list(report.bbox_max)},
            status="received",
            render_version=cfg.render.render_version,
            in_curated=curated_id is not None,
            curated_id=curated_id,
            note="; ".join(warnings),
        )
        self.uploads.upsert(rec)

        if not report.ok:
            self.uploads.set_status(upload_id, "rejected", "; ".join(report.errors))
            report.raise_if_failed()

        # --- 已在 curated 資料庫：直接用現成的 descriptor 與三視圖 ---
        if curated_id is not None:
            hit_side = curated_hit[0] if curated_hit else normalize_side(query_side)
            got = self._resolve_from_curated(hit_side, curated_id)
            if got is not None:
                desc, views = got
                self.uploads.set_status(upload_id, "resolved_curated", "; ".join(warnings))
                return upload_id, desc, views, resolution, warnings
            warnings.append(
                f"{curated_id} 在 curated 資料庫中，但取不到 descriptor 或三視圖，改為重新計算"
            )

        # --- 冷路徑：算 descriptor + 畫三視圖（有快取就重用）---
        cached = self.uploads.load_descriptor(upload_id)
        if cached is not None:
            centroid, ratios2d, eigvecs = cached
            desc = NeuronDescriptor(resolved_id, centroid, ratios2d, eigvecs)
        else:
            desc = self._compute_descriptor(swc_path, resolved_id)
            self.uploads.save_descriptor(upload_id, desc.centroid, desc.ratios2d, desc.eigvecs)

        views = self.uploads.load_views(upload_id, cfg.render.render_version)
        if views is None:
            views = self._render_views(swc_path)
            self.uploads.save_views(
                upload_id,
                resolved_id,
                views,
                cfg.render.render_version,
                cfg.render.scale_um_per_px,
                cfg.render.normalize,
            )
        return upload_id, desc, views, resolution, warnings

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        swc_path: str | Path | None = None,
        swc_bytes: bytes | None = None,
        filename: str | None = None,
        neuron_id: str | None = None,
        query_side: str,
        target_side: str | None = None,
        top_n: int | None = None,
        mirror: bool = False,
    ) -> QueryResult:
        """查詢與某顆神經元最相似的目標。

        參數
        ----
        swc_path / swc_bytes+filename : 上傳的 SWC（擇一）。
        neuron_id : 直接指定 curated 資料庫裡的神經元（不上傳檔案時用）。
        query_side  : 查詢這顆屬於哪一側，"FC" 或 "EM"。由前端提供。
        target_side : 要搜尋哪一個資料庫。預設是另一側；目前只有 FC / EM 兩個選擇。
        top_n : 輸出幾對，預設 5。
        mirror : **左右腦鏡像**（把 x 座標取負之後再算 descriptor 與三視圖）。

            為什麼需要這個開關：EM 資料庫只覆蓋單側腦（質心 x 落在 -268~68），
            FC 兩側都有（-435~430）。實測 500 顆隨機 FC 神經元，有 117 顆（23%）
            在 EM 側找不到任何候選；把 x 取負之後只剩 19 顆兩個方向都找不到。
            也就是說多數「查無結果」其實是左右腦不對稱造成的。

            目前**尚未實作**（依討論先擱置，涉及資料庫資料本身的問題）。
            要啟用的話需要：在 _prepare_upload 算 descriptor 與畫圖之前，
            先把 swc.xyz[:, 0] 取負；並決定鏡像後的結果要不要標註在輸出裡。
            傳 True 會直接丟 NotImplementedError，不會安靜地給出錯誤結果。
        """
        if mirror:
            raise NotImplementedError(
                "mirror（左右腦鏡像）尚未實作。詳見 NeuronMatchService.query 的說明。"
            )

        cfg = self.cfg
        q_side = normalize_side(query_side)
        t_side = normalize_side(target_side) if target_side else other_side(q_side)
        n = int(top_n if top_n is not None else cfg.default_top_n)
        timings: dict[str, float] = {}
        warnings: list[str] = []

        # --- 1) 解析查詢對象 ---
        t0 = time.perf_counter()
        upload_id: str | None = None

        if swc_bytes is not None or swc_path is not None:
            if swc_bytes is None:
                p = Path(swc_path)  # type: ignore[arg-type]
                if not p.exists():
                    raise FileNotFoundError(f"找不到 SWC: {p}")
                swc_bytes = p.read_bytes()
                filename = filename or p.name
            if not filename:
                raise ValidationError("上傳檔案必須提供檔名（neuron id 由檔名決定）")
            upload_id, desc, q_views, resolution, w = self._prepare_upload(
                swc_bytes, filename, q_side
            )
            warnings.extend(w)
        elif neuron_id:
            got = self._resolve_from_curated(q_side, neuron_id)
            if got is None:
                raise ValidationError(
                    f"{q_side} 資料庫中找不到 {neuron_id}（或缺少三視圖）"
                )
            desc, q_views = got
            resolution = f"curated_by_id:{neuron_id}"
        else:
            raise ValidationError("必須提供 swc_path / swc_bytes 或 neuron_id 其中之一")

        source_id = desc.neuron_id
        validate_views(q_views, source_id)
        timings["resolve"] = time.perf_counter() - t0

        # --- 快取：同一份檔案 + 同一個模型 + 同一組畫圖參數 ---
        if upload_id and cfg.reuse_cached_result:
            cached = self.uploads.load_cached_result(
                upload_id,
                target_side=t_side,
                model_id=cfg.model.model_id,
                render_version=cfg.render.render_version,
            )
            if cached is not None and not cached.empty:
                timings["total"] = time.perf_counter() - t0
                return self._finish(
                    cached, source_id, q_side, t_side, n, upload_id,
                    resolution + "+cached_result", len(cached), len(cached), 0,
                    warnings, timings,
                )

        # --- 2) 候選初篩 ---
        t1 = time.perf_counter()
        cand = match_one(desc, self.db[t_side], cfg.match)
        timings["match"] = time.perf_counter() - t1

        if cand.empty:
            warnings.append(
                f"在 {t_side} 資料庫中找不到任何幾何特徵相近的候選"
                f"（質心門檻 {cfg.match.centroid_th} um、ratio 門檻 {cfg.match.ratio_th}）"
            )
            timings["total"] = time.perf_counter() - t0
            if upload_id:
                self.uploads.set_status(upload_id, "scored", "; ".join(warnings))
            return QueryResult(
                source_id=source_id, query_side=q_side, target_side=t_side,
                table=_empty_result_table(), full_table=_empty_result_table(),
                upload_id=upload_id, resolution=resolution, warnings=warnings, timings=timings,
            )

        # --- 3) 取候選三視圖 ---
        t2 = time.perf_counter()
        target_ids = cand["target_id"].astype(str).tolist()
        found_ids, found_views, missing = self.views[t_side].get_many(target_ids)
        # 全黑的圖必須剔除：模型對空白輸入會給高分（實測 0.678）
        keep_ids: list[str] = []
        keep_views: list[np.ndarray] = []
        for nid, v in zip(found_ids, found_views):
            if v.size and int(v.max()) > 0:
                keep_ids.append(nid)
                keep_views.append(v)
            else:
                missing.append(nid)
        if missing:
            warnings.append(f"{len(missing)} 個候選沒有可用的三視圖，已略過")
        timings["fetch_views"] = time.perf_counter() - t2

        if not keep_ids:
            timings["total"] = time.perf_counter() - t0
            return QueryResult(
                source_id=source_id, query_side=q_side, target_side=t_side,
                table=_empty_result_table(), full_table=_empty_result_table(),
                upload_id=upload_id, resolution=resolution,
                n_candidates=len(cand), n_missing_views=len(missing),
                warnings=warnings, timings=timings,
            )

        # --- 4) 打分 ---
        if self.model is None:
            raise RuntimeError("服務是以 load_model=False 建立的，無法打分")
        t3 = time.perf_counter()
        scores = self.model.score(q_views, keep_views, query_side=q_side)
        timings.update(self.model.last_timings)
        timings["score_total"] = time.perf_counter() - t3

        full = pd.DataFrame(
            {
                "source_id": source_id,
                "target_id": keep_ids,
                "similarity_score": scores.astype(np.float32),
            }
        )
        # 分數高的排前面；同分時用 target_id 保證順序穩定可重現
        full = full.sort_values(
            ["similarity_score", "target_id"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        full["rank"] = np.arange(1, len(full) + 1, dtype=np.int64)
        full = full[list(RESULT_COLUMNS)]

        if upload_id:
            self.uploads.save_result(
                upload_id, full,
                target_side=t_side,
                model_id=cfg.model.model_id,
                render_version=cfg.render.render_version,
                n_candidates=len(cand),
            )
            self.uploads.set_status(upload_id, "scored", "; ".join(warnings))

        timings["total"] = time.perf_counter() - t0
        return self._finish(
            full, source_id, q_side, t_side, n, upload_id, resolution,
            len(cand), len(keep_ids), len(missing), warnings, timings,
        )

    def _finish(
        self, full: pd.DataFrame, source_id: str, q_side: str, t_side: str, n: int,
        upload_id: str | None, resolution: str, n_candidates: int, n_scored: int,
        n_missing: int, warnings: list[str], timings: dict[str, float],
    ) -> QueryResult:
        full = full[list(RESULT_COLUMNS)].copy()
        top = full.head(n).reset_index(drop=True) if n > 0 else full.copy()
        return QueryResult(
            source_id=source_id, query_side=q_side, target_side=t_side,
            table=top, full_table=full, upload_id=upload_id, resolution=resolution,
            n_candidates=n_candidates, n_scored=n_scored, n_missing_views=n_missing,
            warnings=warnings, timings=timings,
        )

    # ------------------------------------------------------------------

    def write_csv(self, result: QueryResult, out_path: str | Path) -> Path:
        """輸出最終 CSV：source_id / target_id / similarity_score / rank。"""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        result.table.to_csv(p, index=False)
        result.csv_path = p
        if result.upload_id:
            # 同時在上傳暫存區留一份，方便日後人工檢視
            result.table.to_csv(self.uploads.result_path(result.upload_id), index=False)
        return p
