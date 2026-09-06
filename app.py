"""Web 服務層（FastAPI）。

安裝：
    pip install fastapi uvicorn python-multipart

啟動：
    uvicorn app:app --host 0.0.0.0 --port 8000

    ⚠️ 只能用單一 worker（uvicorn 預設就是），不要開 --workers N：
       每個 worker 都會各自載入模型與整個資料庫索引，記憶體會乘上 N。
       要擴充吞吐量請在前面放一層 queue，不要靠多 worker。

端點：
    GET  /health          服務狀態
    GET  /sides           可選的資料庫（目前只有 FC / EM）
    POST /match           上傳 SWC，回傳 CSV
    POST /match/by_id     用資料庫中既有的 neuron id 查詢，回傳 CSV

/match 的參數：
    file         : 上傳的 .swc（必填）
    query_side   : "FC" 或 "EM"，這顆神經元屬於哪一側（必填，由前端提供）
    target_side  : 要搜尋哪個資料庫，預設是另一側
    top_n        : 回傳幾對，預設 5
    mirror       : 左右腦鏡像，目前尚未實作，傳 true 會回 501

回傳：text/csv，欄位 source_id, target_id, similarity_score, rank
"""

from __future__ import annotations

import io
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from nrn_service.config import SIDES, ServiceConfig
from nrn_service.service import NeuronMatchService, QueryResult
from nrn_service.validation import ValidationError

cfg = ServiceConfig()
app = FastAPI(title="Neuron Matching Service", version="1.0")

# 模型與資料庫索引在啟動時載入一次。
# 千萬不要改成每個 request 才載入：光是 import keras + 建圖 + 載權重就要 40 秒。
service: NeuronMatchService | None = None


@app.on_event("startup")
def _startup() -> None:
    global service
    service = NeuronMatchService(cfg)
    print(f"[startup] {service.db.summary()}  {service.startup_timings}", flush=True)


def _require_service() -> NeuronMatchService:
    if service is None:
        raise HTTPException(status_code=503, detail="服務尚未完成啟動")
    return service


def _csv_response(result: QueryResult, svc: NeuronMatchService) -> StreamingResponse:
    if result.upload_id:
        # 在上傳暫存區留一份，方便日後人工檢視與確認
        result.table.to_csv(svc.uploads.result_path(result.upload_id), index=False)
    buf = io.StringIO()
    result.table.to_csv(buf, index=False)
    buf.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="{result.source_id}_matches.csv"',
        "X-Source-Id": result.source_id,
        "X-Query-Side": result.query_side,
        "X-Target-Side": result.target_side,
        "X-Candidates": str(result.n_candidates),
        "X-Scored": str(result.n_scored),
        "X-Resolution": result.resolution,
        "X-Elapsed-Seconds": f"{result.timings.get('total', 0):.3f}",
    }
    if result.upload_id:
        headers["X-Upload-Id"] = result.upload_id
    if result.warnings:
        # ⚠️ HTTP header 只能放 latin-1，警告訊息是中文，直接塞會讓整個回應爆掉
        #    （UnicodeEncodeError）。所以用百分比編碼，前端收到後用
        #    decodeURIComponent(res.headers.get("X-Warnings")) 還原。
        headers["X-Warnings"] = quote(" | ".join(w.replace("\n", " ") for w in result.warnings))
    # 讓瀏覽器的 JavaScript 讀得到這些自訂 header（預設只看得到少數幾個）
    headers["Access-Control-Expose-Headers"] = ", ".join(headers.keys())
    return StreamingResponse(buf, media_type="text/csv", headers=headers)


@app.get("/health")
def health() -> dict:
    svc = _require_service()
    return {
        "status": "ok",
        "database": {s: len(idx) for s, idx in svc.db.sides.items()},
        "model": cfg.model.model_id,
        "render_version": cfg.render.render_version,
        "startup_seconds": {k: round(v, 2) for k, v in svc.startup_timings.items()},
    }


@app.get("/sides")
def sides() -> dict:
    """前端用來填 query_side / target_side 的選項。"""
    svc = _require_service()
    return {"sides": list(SIDES), "sizes": {s: len(idx) for s, idx in svc.db.sides.items()}}


@app.post("/match")
async def match(
    file: UploadFile = File(..., description="要查詢的 SWC 檔"),
    query_side: str = Form(..., description="這顆神經元屬於哪一側：FC 或 EM"),
    target_side: Optional[str] = Form(None, description="要搜尋哪個資料庫，預設為另一側"),
    top_n: int = Form(cfg.default_top_n),
    mirror: bool = Form(False),
):
    svc = _require_service()
    data = await file.read()
    try:
        result = svc.query(
            swc_bytes=data,
            filename=file.filename or "",
            query_side=query_side,
            target_side=target_side,
            top_n=top_n,
            mirror=mirror,
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.is_empty:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "在目標資料庫中找不到任何幾何特徵相近的候選",
                "source_id": result.source_id,
                "query_side": result.query_side,
                "target_side": result.target_side,
                "warnings": result.warnings,
            },
        )
    return _csv_response(result, svc)


@app.post("/match/by_id")
def match_by_id(
    neuron_id: str = Form(...),
    query_side: str = Form(...),
    target_side: Optional[str] = Form(None),
    top_n: int = Form(cfg.default_top_n),
    mirror: bool = Form(False),
):
    svc = _require_service()
    try:
        result = svc.query(
            neuron_id=neuron_id,
            query_side=query_side,
            target_side=target_side,
            top_n=top_n,
            mirror=mirror,
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.is_empty:
        return JSONResponse(
            status_code=404,
            content={"detail": "找不到候選", "source_id": result.source_id},
        )
    return _csv_response(result, svc)
