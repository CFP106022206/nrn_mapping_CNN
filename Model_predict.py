from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

from util import load_pkl


@dataclass(frozen=True)
class Config:
    model_dir: str = "./Annotator_Model"
    model_prefix: str = "Annotator_D1-D6_"   # without fold number
    out_dir: str = "./preTrain_label"

    unlabel_dir: str = "./data/statistical_results/pre_train_map"
    # if you have multiple dirs, you can add them in list below

    # memory control: max number of pairs per chunk
    chunk_pairs: int = 50000

    # inference
    threshold: float = 0.5

    # normalization: keep consistent with your training (optional)
    do_minmax_norm: bool = False  # set True if needed


def list_pkl_files(folder: str) -> List[Path]:
    p = Path(folder)
    return sorted([x for x in p.iterdir() if x.suffix == ".pkl"])


def iter_pairs_from_pkls(pkl_files: List[Path]) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
    """
    Yields: (fc_id, em_id, fc_img(H,W,3), em_img(H,W,3))
    Assumes data[3] and data[4] are (3,H,W) and need transpose to (H,W,3)
    """
    for pkl in pkl_files:
        data_lst = load_pkl(str(pkl))
        for data in data_lst:
            fc_id, em_id = data[0], data[1]
            fc_img = np.transpose(data[3], (1, 2, 0))
            em_img = np.transpose(data[4], (1, 2, 0))
            yield fc_id, em_id, fc_img, em_img


def batched_pairs(
    pair_iter: Iterable[Tuple[str, str, np.ndarray, np.ndarray]],
    chunk_pairs: int,
) -> Iterator[Tuple[List[str], List[str], np.ndarray, np.ndarray]]:
    fc_ids: List[str] = []
    em_ids: List[str] = []
    fc_imgs: List[np.ndarray] = []
    em_imgs: List[np.ndarray] = []

    for fc_id, em_id, fc_img, em_img in pair_iter:
        fc_ids.append(fc_id)
        em_ids.append(em_id)
        fc_imgs.append(fc_img)
        em_imgs.append(em_img)

        if len(fc_ids) >= chunk_pairs:
            yield fc_ids, em_ids, np.asarray(fc_imgs), np.asarray(em_imgs)
            fc_ids, em_ids, fc_imgs, em_imgs = [], [], [], []

    if fc_ids:
        yield fc_ids, em_ids, np.asarray(fc_imgs), np.asarray(em_imgs)


def maybe_minmax_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx > mn:
        return (x - mn) / (mx - mn)
    return x


def load_models(cfg: Config, model_ids: List[int]):
    models = []
    for mid in model_ids:
        path = Path(cfg.model_dir) / f"{cfg.model_prefix}{mid}.h5"
        models.append(load_model(path))
    return models


def predict_ensemble(models, fc_img: np.ndarray, em_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean, std) with shape (N,1)
    """
    preds = []
    for m in models:
        p = m.predict({"FC": fc_img, "EM": em_img}, verbose=0)
        preds.append(p)
    pred_np = np.asarray(preds)  # (M,N,1)
    return pred_np.mean(axis=0), pred_np.std(axis=0)


def main() -> None:
    cfg = Config()

    # model selection: default single model id from argv[1]
    single_model = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    model_ids = [single_model]  # keep as list so you can expand to ensemble easily

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    pkl_files = list_pkl_files(cfg.unlabel_dir)
    print(f"Found {len(pkl_files)} pkl files in {cfg.unlabel_dir}")

    models = load_models(cfg, model_ids)
    print("Used models:", model_ids)

    # iterate + chunk by PAIRS (not by number of pkl files)
    pair_iter = iter_pairs_from_pkls(pkl_files)
    chunk_iter = batched_pairs(pair_iter, cfg.chunk_pairs)

    start = time.time()
    for chunk_idx, (fc_ids, em_ids, fc_img, em_img) in enumerate(chunk_iter):
        print(f"\nChunk {chunk_idx}: {len(fc_ids)} pairs")

        if cfg.do_minmax_norm:
            fc_img = maybe_minmax_norm(fc_img)
            em_img = maybe_minmax_norm(em_img)

        st = time.time()
        mean_pred, std_pred = predict_ensemble(models, fc_img, em_img)
        print("Predict time:", time.time() - st, "s")

        bin_pred = (mean_pred.reshape(-1) > cfg.threshold).astype(int)

        out_df = pd.DataFrame(
            {
                "fc_id": fc_ids,
                "em_id": em_ids,
                "model_predict": mean_pred.reshape(-1),
                "binary_label": bin_pred,
                "pred_std": std_pred.reshape(-1),
            }
        )

        # stable filename
        out_name = f"{cfg.model_prefix}{single_model}_chunk{chunk_idx}.csv"
        out_path = Path(cfg.out_dir) / out_name
        out_df.to_csv(out_path, index=False)
        print("Saved:", out_path)

    print("\nProgram Completed. Total time:", time.time() - start, "s")


if __name__ == "__main__":
    main()
