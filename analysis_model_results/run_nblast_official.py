#!/usr/bin/env python
"""用官方 NBLAST 實作重算 D1 (D5) 與 D2 (D2+D6) 的已標註配對分數。

這支程式**獨立於 s01-s13 的分析流程**, 只做一件事: 拿 navis (nat 的官方 Python
移植) 與 Costa et al. 2016 訓練的官方計分矩陣 smat.fcwb, 對 labeled_info 裡的
898 組人工標註配對重算 NBLAST 分數, 作為專案內既有兩套不一致分數的仲裁基準。

為什麼需要它
------------
labeled_info 底下有兩套彼此不符的 NBLAST 分數:
  舊版 (單向): D2p_nblast_score.csv 及其子集 nblast_D2+D6_50as1.csv 等, 五個檔
               案彼此完全相同
  新版 (含 inverse, 腦科中心提供): nblast_all_list_D2_D5_include_inverse_label.csv
               與其簡化版 nblast_all_list_D2_D5_label.csv (論文 fig8/fig9 使用)
兩者 Pearson 0.93, 平均絕對差 0.084, 沒有任何一組完全相同。用官方實作重算可以
判斷哪一套接近標準 NBLAST, 或兩者都不是。

方法 (依 Costa, Manton, Ostrovsky, Prohaska & Jefferis 2016, Neuron)
--------------------------------------------------------------------
  1. 骨架重採樣到 1 um, 以 k=5 最近鄰的主方向作為每點的切向量 (dotprops)
  2. 對 query 的每個點找 target 最近點, 由 (距離, |切向量內積|) 查 smat.fcwb
     取得 log2 odds, 加總
  3. normalized = S(A,B)/S(A,A); mean = 兩個方向的平均

執行 (venv 路徑用參數傳入, 不寫死):
    conda run -n nblast python analysis_dense_vs_projection/run_nblast_official.py

輸出: results/nblast_official.csv   898 組人工標註配對的官方 NBLAST 分數
      欄位 fc_id, em_id, conf (專家信心 0-1), group, nblast_official

另可用 --emit-figure-csv 一併寫出 labeled_info/nblast_official_mean.csv,
欄位改為 result_analysis_make_figure.py 期望的 fc_id, em_id, "similarity score",
label, 供該程式的 nblast 模式直接讀取。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"

LABEL_DIR = ROOT / "labeled_info"

# 每個 sub dataset 的人工標註配對來源。這些檔案的 `label` 欄是專家信心 (0-1),
# 不是二值標籤; 二值化留給下游 (專家信心 >= 0.5 為真), 與 result_analysis_make_figure.py
# 的 compute_all_metrics() 一致。
PAIR_SOURCES = {
    "D1_projection": [LABEL_DIR / "D5_conf.csv"],
    "D2_dense": [LABEL_DIR / "D2_conf.csv", LABEL_DIR / "D6_conf.csv"],
}
SWC_DIR = {"FC": ROOT / "data" / "SWC" / "FC", "EM": ROOT / "data" / "SWC" / "EM"}


def load_pairs() -> pd.DataFrame:
    """898 組人工標註配對, 帶專家信心 (soft label)。"""
    rows = []
    for grp, files in PAIR_SOURCES.items():
        for f in files:
            if not f.exists():
                sys.exit(f"找不到標註檔: {f}")
            d = pd.read_csv(f)[["fc_id", "em_id", "label"]].rename(
                columns={"label": "conf"})
            d["group"] = grp
            d["source_file"] = f.name
            rows.append(d)
    p = pd.concat(rows, ignore_index=True)
    p["fc_id"] = p["fc_id"].astype(str)
    p["em_id"] = p["em_id"].astype(str)
    return p.drop_duplicates(["fc_id", "em_id", "group"]).reset_index(drop=True)


def build_dotprops(ids, source, navis, k, resample):
    base = SWC_DIR[source]
    out, t0 = {}, time.time()
    for i, n in enumerate(ids, 1):
        p = base / f"{n}.swc"
        if not p.exists():
            print(f"  ! 缺 SWC: {source}/{n}")
            continue
        nrn = navis.read_swc(str(p))
        dp = navis.make_dotprops(nrn, k=k, resample=resample)
        dp.id = n
        out[n] = dp
        if i % 50 == 0:
            print(f"  {source} dotprops {i}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="dotprops 的最近鄰數")
    ap.add_argument("--resample", type=float, default=1.0, help="重採樣步長 (um)")
    ap.add_argument("--alpha", action="store_true", help="啟用 NBLAST 的 UseAlpha 加權")
    ap.add_argument("--emit-figure-csv", action="store_true",
                    help="另外寫出 labeled_info/nblast_official_mean.csv "
                         "(result_analysis_make_figure.py 的 nblast 模式讀這個)")
    args = ap.parse_args()

    try:
        import navis
        from navis.nbl.smat import smat_fcwb
    except ImportError:
        sys.exit("需要 navis。請用安裝了 navis 的直譯器執行, 例如:\n"
                 "  /path/to/nblast_env/bin/python run_nblast_official.py")
    navis.set_pbars(hide=True)
    print(f"navis {navis.__version__}, smat.fcwb (alpha={args.alpha}), "
          f"k={args.k}, resample={args.resample} um\n")

    pairs = load_pairs()
    print(f"標註配對: {len(pairs)} 組")
    print(pairs.groupby("group").size().to_string(), "\n")

    fc_ids = sorted(pairs.fc_id.unique())
    em_ids = sorted(pairs.em_id.unique())
    print(f"需要 dotprops: FC {len(fc_ids)}, EM {len(em_ids)}")
    fc = build_dotprops(fc_ids, "FC", navis, args.k, args.resample)
    em = build_dotprops(em_ids, "EM", navis, args.k, args.resample)

    smat = smat_fcwb(alpha=args.alpha)
    recs, t0 = [], time.time()
    # 只算標註配對: 依 fc 分組, 每次只對它自己的候選計分
    for i, (f, sub) in enumerate(pairs.groupby("fc_id"), 1):
        if f not in fc:
            continue
        # 同一顆 FC 可能同時出現在兩組, em_id 需去重 (navis 要求 target ID 唯一)
        tg = [em[e] for e in dict.fromkeys(sub.em_id) if e in em]
        if not tg:
            continue
        m = navis.nblast(navis.NeuronList([fc[f]]), navis.NeuronList(tg),
                         scores="mean", smat=smat, normalized=True,
                         use_alpha=args.alpha, progress=False, n_cores=1)
        for e, v in zip([t.id for t in tg], np.asarray(m)[0]):
            recs.append({"fc_id": f, "em_id": e, "nblast_official": float(v)})
        if i % 40 == 0:
            print(f"  nblast {i}/{pairs.fc_id.nunique()}  ({time.time()-t0:.0f}s)", flush=True)

    sc = pd.DataFrame(recs).drop_duplicates(["fc_id", "em_id"])
    out = pairs.merge(sc, on=["fc_id", "em_id"], how="left")
    out.to_csv(OUT / "nblast_official.csv", index=False)
    if args.emit_figure_csv:
        fig = (out[["fc_id", "em_id", "nblast_official", "conf"]]
               .rename(columns={"nblast_official": "similarity score", "conf": "label"}))
        fig.to_csv(LABEL_DIR / "nblast_official_mean.csv", index=False)
        print(f"另寫出 {LABEL_DIR / 'nblast_official_mean.csv'}")
    print(f"\n算出 {out.nblast_official.notna().sum()}/{len(out)} 組")
    print(out.groupby("group")["nblast_official"]
             .describe(percentiles=[.25, .5, .75]).round(4).to_string())
    return out


if __name__ == "__main__":
    main()
