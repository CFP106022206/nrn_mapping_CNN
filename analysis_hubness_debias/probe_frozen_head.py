"""階段 2：凍結 trunk 的探針——表徵裡還有沒有沒被榨出來的訊號？

問題
----
去偏實驗（README）證明差距不是可後處理的 per-EM 偏差，而是真實的排序品質差異
（池內 AUC 0.861 vs 0.898）。那麼接下來唯一該問的是：
**凍結 trunk、只重訓 head，能不能超過現有的池內 AUC？**

- 能 → 表徵有空間，現有 head 沒榨乾 → 階段 3 的排序微調值得做。
- 不能 → 瓶頸是表徵（50×50×3 下採樣的三視圖投影），換損失函數救不了。

作法
----
1. 從 train 半邊的 FC 抽池內三元組（正例 = 型別相符、負例 = 確定不同型）。
2. 用凍結 trunk 算這些配對的特徵並快取。
3. 用 InfoNCE 重訓一個 head（架構與原本相同）。
4. 在 eval 半邊的 FC 上跑完整的池，比較池內 AUC / rank-1。

⚠️ 前處理是**成對相依**的（`swc_util._pad_to_same_size` 以「這一對裡較大的那張」
為準補零，`nrn_service/scoring.py:8-10` 有記載），所以特徵**不能** per-neuron 快取，
只能針對抽樣到的配對算。

執行：python3 analysis_hubness_debias/probe_frozen_head.py --model annotator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "analysis_external_validation"))

RESULTS = ROOT / "results"

MODELS = {
    "finetune": "./FineTune_Model/FineTune_miniLR_D1-D6_0.weights.h5",
    "annotator": "./Annotator_Model/Annotator_D1-D6_0.weights.h5",
}

N_POS = 8        # 每顆 FC 抽幾個池內正例
N_NEG = 64       # 每顆 FC 抽幾個池內負例（訓練時每步再抽 K 個）
K = 32           # InfoNCE 的負例數
TAU = 0.07


def load_labeled_pool(model: str) -> pd.DataFrame:
    from debias import build_judge, label_pairs

    df = pd.read_parquet(ROOT / "scores" / f"pool_scores_{model}.parquet")
    df["fc_id"] = df.fc_id.astype(str)
    fc, judge = build_judge()
    return label_pairs(df, fc, judge)


def split_fc(df: pd.DataFrame, seed: int) -> tuple[set[str], set[str]]:
    """對 winnable FC 依族群分層切一半。"""
    ext = PROJECT / "analysis_external_validation" / "results" / "pair_labels.csv"
    fam = pd.read_csv(ext, usecols=["fc_id", "family"]).drop_duplicates("fc_id")
    fam = dict(zip(fam.fc_id, fam.family))
    win = sorted(df[df.type_label == 1.0].fc_id.unique())
    rng = np.random.default_rng(seed)
    tr, ev = [], []
    for f in sorted({fam.get(x, "?") for x in win}):
        grp = np.array([x for x in win if fam.get(x, "?") == f])
        idx = rng.permutation(len(grp))
        h = len(grp) // 2
        tr += list(grp[idx[:h]])
        ev += list(grp[idx[h:]])
    return set(tr), set(ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="annotator")
    ap.add_argument("--init", choices=["scratch", "existing"], default="scratch",
                    help="head 的起點：scratch 測表徵帶多少訊號，"
                         "existing 測排序目標能否改進現有 head")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import tensorflow as tf
    from model import MVCNN_Siamese
    from nrn_service.config import ServiceConfig
    from nrn_service.view_store import ViewStore
    from swc_util import _pad_to_same_size, _resize_to_50

    from debias import pool_auc

    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_labeled_pool(args.model)
    tr_fc, ev_fc = split_fc(df, args.seed)
    print(f"[probe] train FC {len(tr_fc)} / eval FC {len(ev_fc)}")

    # ---------- 取樣訓練配對 ----------
    rng = np.random.default_rng(args.seed)
    rows = []
    for fc, g in df[df.fc_id.isin(tr_fc)].groupby("fc_id", observed=True):
        pos = g[g.type_label == 1.0].em_id.to_numpy()
        neg = g[g.type_label == 0.0].em_id.to_numpy()
        if len(pos) == 0 or len(neg) < K:
            continue
        p = rng.choice(pos, min(N_POS, len(pos)), replace=False)
        n = rng.choice(neg, min(N_NEG, len(neg)), replace=False)
        for e in p:
            rows.append((fc, int(e), 1))
        for e in n:
            rows.append((fc, int(e), 0))
    samp = pd.DataFrame(rows, columns=["fc_id", "em_id", "is_pos"]).drop_duplicates()
    print(f"[probe] 訓練配對 {len(samp):,}（FC {samp.fc_id.nunique()}）")

    # ---------- 建 trunk 特徵抽取器 ----------
    net = MVCNN_Siamese((50, 50, 3))
    net.load_weights(MODELS[args.model])
    cat = next(l for l in net.layers if l.__class__.__name__ == "Concatenate")
    trunk = tf.keras.Model(net.input, cat.output)      # (None, 18432)
    trunk.trainable = False

    cfg = ServiceConfig()
    vs = {
        s: ViewStore(s, store_dir=cfg.paths.view_store_dir,
                     npz_dir=cfg.paths.views_dir(s), render_version=None)
        for s in ("FC", "EM")
    }
    vcache: dict[tuple[str, str], np.ndarray] = {}

    def views(side: str, nid: str) -> np.ndarray | None:
        k = (side, nid)
        if k not in vcache:
            vcache[k] = vs[side].get(nid)
        return vcache[k]

    def make_batch(pairs: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
        q = np.empty((len(pairs), 50, 50, 3), dtype=np.float32)
        t = np.empty_like(q)
        for i, (fc, em) in enumerate(pairs):
            a, b = _pad_to_same_size(views("FC", fc), views("EM", str(em)))
            q[i] = np.transpose(_resize_to_50(a, (50, 50)), (1, 2, 0))
            t[i] = np.transpose(_resize_to_50(b, (50, 50)), (1, 2, 0))
        return q / 255.0, t / 255.0

    def trunk_feats(pairs: list[tuple[str, int]], bs: int = 512) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), bs):
            q, t = make_batch(pairs[i : i + bs])
            out.append(trunk.predict({"FC": q, "EM": t}, verbose=0).astype("float16"))
        return np.concatenate(out)

    pairs = list(zip(samp.fc_id, samp.em_id))
    print("[probe] 計算訓練特徵…", flush=True)
    F = trunk_feats(pairs)
    print(f"[probe] 特徵 {F.shape}  {F.nbytes/1e9:.2f} GB")

    idx = {p: i for i, p in enumerate(pairs)}
    by_fc = {
        fc: (
            [idx[(fc, e)] for e in g[g.is_pos == 1].em_id],
            [idx[(fc, e)] for e in g[g.is_pos == 0].em_id],
        )
        for fc, g in samp.groupby("fc_id", observed=True)
    }
    by_fc = {k: v for k, v in by_fc.items() if v[0] and len(v[1]) >= K}
    fcs = sorted(by_fc)
    print(f"[probe] 可用 FC {len(fcs)}")

    # ---------- 新 head（架構同原本）----------
    # 兩種起點問的是不同的問題：
    #   scratch  凍結特徵「帶有多少」可用訊號
    #   existing 排序目標能不能「改進現有」head —— 對階段 3 的 go/no-go 更直接相關
    inp = tf.keras.Input(shape=(F.shape[1],))
    x = tf.keras.layers.Dropout(0.3)(inp)
    x = tf.keras.layers.Dense(256)(x)
    # ⚠️ momentum 預設 0.99：本探針的步數不多（數千步），moving_mean/var 會嚴重滯後，
    #    訓練用 batch 統計量、推論用幾乎沒更新的 moving average，兩者對不上
    #    → smoke test（15 步）實測池內 AUC 掉到 0.33（低於隨機，系統性反向）。
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Activation("gelu")(x)
    logit = tf.keras.layers.Dense(1)(x)
    head = tf.keras.Model(inp, logit)

    if args.init == "existing":
        # 原本的 head 是 Dense(256) -> BN -> gelu -> Dense(1, sigmoid)。
        # 權重形狀與這裡完全相同，只差最後的 sigmoid（探針要 logit），直接搬。
        src = [l for l in net.layers if l.__class__.__name__ in
               ("Dense", "BatchNormalization")][-3:]
        dst = [l for l in head.layers if l.__class__.__name__ in
               ("Dense", "BatchNormalization")]
        assert len(src) == len(dst) == 3, (len(src), len(dst))
        for a, b in zip(src, dst):
            b.set_weights(a.get_weights())
        print("[probe] head 由現有權重初始化")

    lr = 1e-3 if args.init == "scratch" else 1e-4
    opt = tf.keras.optimizers.AdamW(learning_rate=lr)
    print(f"[probe] init={args.init}  lr={lr}")

    Ft = tf.constant(F.astype("float32"))

    @tf.function
    def step(pi, ni):
        with tf.GradientTape() as tape:
            sp = head(tf.gather(Ft, pi), training=True)              # (B,1)
            sn = head(tf.gather(Ft, tf.reshape(ni, [-1])), training=True)
            sn = tf.reshape(sn, [tf.shape(pi)[0], K])                 # (B,K)
            logits = tf.concat([sp, sn], axis=1) / TAU
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros(tf.shape(pi)[0], tf.int32), logits=logits
                )
            )
        g = tape.gradient(loss, head.trainable_weights)
        opt.apply_gradients(zip(g, head.trainable_weights))
        return loss

    # 以「錨點 =（FC, 某個正例）」為單位迭代，而不是以 FC 為單位：
    # 前者每 epoch 約 40 步，後者只有 5 步——步數太少時 BN 的 moving 統計量學不起來。
    anchors = [(f, p) for f in fcs for p in by_fc[f][0]]
    B = 128
    print(f"[probe] 訓練 head… 錨點 {len(anchors)}，每 epoch {len(anchors)//B + 1} 步",
          flush=True)
    for ep in range(args.epochs):
        rng.shuffle(anchors)
        losses = []
        for i in range(0, len(anchors), B):
            chunk = anchors[i : i + B]
            pi = np.array([p for _, p in chunk], dtype=np.int32)
            ni = np.array([rng.choice(by_fc[f][1], K, replace=False) for f, _ in chunk],
                          dtype=np.int32)
            losses.append(float(step(tf.constant(pi), tf.constant(ni))))
        if ep < 3 or (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:3d}  loss {np.mean(losses):.4f}", flush=True)

    # ---------- 評估：eval 半邊的完整池 ----------
    ev = df[df.fc_id.isin(ev_fc)].copy()
    print(f"[probe] 評估 {ev.fc_id.nunique()} 顆 FC、{len(ev):,} 對，重算 trunk…", flush=True)
    ev_pairs = list(zip(ev.fc_id.astype(str), ev.em_id.astype(int)))
    scores = []
    BS = 4096
    for i in range(0, len(ev_pairs), BS):
        f = trunk_feats(ev_pairs[i : i + BS], bs=1024).astype("float32")
        scores.append(head.predict(f, verbose=0, batch_size=2048).reshape(-1))
        if (i // BS) % 50 == 0:
            print(f"    {i:,}/{len(ev_pairs):,}", flush=True)
    ev["probe"] = np.concatenate(scores)

    rows = []
    for key, name in (("score", "原 head（基線）"), ("probe", "重訓 head（探針）")):
        d = ev.sort_values(key, ascending=False)
        g = d.groupby("fc_id", observed=True)
        t1 = g.head(1)
        t1 = t1[t1.type_label.notna()]
        d = d.copy()
        d["_r"] = g.cumcount() + 1
        corr = d[d.type_label == 1.0].groupby("fc_id", observed=True)._r.min()
        aucs = [a for a in (pool_auc(x, key) for _, x in d.groupby("fc_id", observed=True))
                if a is not None]
        rows.append({
            "key": name, "n_fc": d.fc_id.nunique(),
            "rank1_hit_pct": round(100 * (t1.type_label == 1.0).mean(), 2),
            "MRR": round(float((1 / corr).mean()), 4),
            "median_rank_of_correct": int(corr.median()),
            "p@5_pct": round(100 * (corr <= 5).mean(), 2),
            "pool_auc": round(float(np.mean(aucs)), 4),
        })
    # 保存 eval 分數與 head 權重：EM 側捷徑的分層檢查需要它們，
    # 而且重跑一次評估要 12 分鐘，不值得為了分析再算一遍。
    tag = f"{args.model}_{args.init}"
    ev[["fc_id", "em_id", "type_label", "score", "probe"]].to_parquet(
        RESULTS / f"probe_scores_{tag}.parquet", index=False
    )
    head.save_weights(RESULTS / f"probe_head_{tag}.weights.h5")
    samp.assign(seen=1).to_parquet(RESULTS / f"probe_trainsample_{tag}.parquet", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"probe_{tag}.csv", index=False)
    print(f"\n=== 凍結 trunk 探針（{args.model}）===")
    print(out.to_string(index=False))
    print(f"\n-> results/probe_{tag}.csv（含分數 parquet 與 head 權重）")


if __name__ == "__main__":
    main()
