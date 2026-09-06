"""用 Siamese CNN 對 (query, candidate) 打相似度分數。

前處理必須和訓練時完全一致，所以一律走 swc_util 裡的那三個函式：
    _pad_to_same_size  把一對圖補零到同一個大小（以「這一對裡較大的那張」為準）
    _resize_to_50      下採樣到 50x50（max pooling；比 50 小的會補零）
    /255.0             正規化

注意 _pad_to_same_size 是「成對」做的：同一顆神經元搭配不同的對象，
補零後的大小會不同，所以不能預先算好每顆的 50x50 快取，必須逐對處理。
好在這一步只要 0.38 ms/對，不是瓶頸。

模型有兩個具名輸入 "FC" 和 "EM"，查詢側的圖餵進自己那一側。
（實測 swap 前後分數相關性 0.998，方向影響很小，但還是照語意餵。）
"""

from __future__ import annotations

import time

import numpy as np

from model import MVCNN_Siamese
from swc_util import _pad_to_same_size, _resize_to_50

from .config import ModelConfig, RenderConfig, normalize_side


def make_pair_batch(
    query_views: np.ndarray,
    target_views: list[np.ndarray],
    out_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """把一顆 query 對上 N 個 target，做成 (N,H,W,3) 的兩組輸入。"""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    n = len(target_views)
    q = np.empty((n, out_h, out_w, 3), dtype=np.float32)
    t = np.empty((n, out_h, out_w, 3), dtype=np.float32)

    for i, tv in enumerate(target_views):
        q_pad, t_pad = _pad_to_same_size(query_views, tv)
        q[i] = np.transpose(_resize_to_50(q_pad, (out_h, out_w)), (1, 2, 0))
        t[i] = np.transpose(_resize_to_50(t_pad, (out_h, out_w)), (1, 2, 0))

    q /= 255.0
    t /= 255.0
    return q, t


class ScoringModel:
    """常駐的打分模型。

    ⚠️ 一定要在服務啟動時就建好並暖機，不要每個 request 才載入：
       實測 import keras 33.5 s、建圖 + 載權重 7.5 s、第一次 predict（XLA 編譯）8.2 s，
       暖機之後才是 0.13 s / 186 對。
    """

    def __init__(self, cfg: ModelConfig, render: RenderConfig, *, warmup: bool = True) -> None:
        self.cfg = cfg
        self.render = render
        self.model = MVCNN_Siamese(cfg.input_size)
        self.model.load_weights(cfg.weights)
        self.load_seconds = 0.0
        self.warmup_seconds = 0.0
        self.last_timings: dict[str, float] = {}
        if warmup:
            self.warmup()

    def warmup(self) -> None:
        """跑一次 dummy batch，把 XLA 編譯的成本挪到啟動時。"""
        t0 = time.perf_counter()
        h, w, c = self.cfg.input_size
        # 用正式推論時的 batch 形狀暖機，否則第一次真正的查詢還是要重 trace 一次
        dummy = np.zeros((int(self.cfg.batch_size), h, w, c), dtype=np.float32)
        self.model.predict(
            {"FC": dummy, "EM": dummy}, verbose=0, batch_size=int(self.cfg.batch_size)
        )
        self.warmup_seconds = time.perf_counter() - t0

    def score(
        self,
        query_views: np.ndarray,
        target_views: list[np.ndarray],
        *,
        query_side: str,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """回傳 (N,) float32 相似度分數，順序與 target_views 相同。"""
        if not target_views:
            return np.empty((0,), dtype=np.float32)

        q_side = normalize_side(query_side)
        t_side = "EM" if q_side == "FC" else "FC"

        t0 = time.perf_counter()
        q, t = make_pair_batch(query_views, target_views, self.render.out_hw)
        t1 = time.perf_counter()

        bs = int(batch_size or self.cfg.batch_size)
        n = q.shape[0]

        # ⚠️ 每個不同的 N 都會讓 keras 重新 trace 一次 predict 函式，
        #    成本 0.8~2.0 秒。因為每次查詢的候選數都不一樣，等於每次查詢
        #    都在付這筆編譯費（實測 198 對第一次 1.13 s、第二次 0.058 s）。
        #    把 batch 補齊到 bs 的倍數，模型就永遠只看到同一種形狀，
        #    trace 一次之後所有查詢都吃快取。補上去的是全零列，
        #    推論時 BatchNormalization 用的是 moving statistics，
        #    每一列彼此獨立，所以不會影響真實資料的分數。
        pad = (-n) % bs
        if pad:
            q = np.concatenate([q, np.zeros((pad,) + q.shape[1:], dtype=np.float32)])
            t = np.concatenate([t, np.zeros((pad,) + t.shape[1:], dtype=np.float32)])

        inputs = {q_side: q, t_side: t}
        pred = self.model.predict(inputs, verbose=0, batch_size=bs)
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)[:n]
        # 讓呼叫端能分辨成本是花在前處理還是模型上
        self.last_timings = {
            "preprocess": t1 - t0,
            "predict": time.perf_counter() - t1,
        }
        return pred
