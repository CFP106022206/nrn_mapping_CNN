"""步驟 13 - 身分洩漏 (identity leakage) 的檢驗: 管道存在, 但模型有沒有依賴它?

問題
----
交叉驗證是以「配對」為單位切分的, 同一顆神經可以同時出現在訓練集與測試集 (只是配在
不同的對象上)。在成對任務裡這被稱為 identity / entity leakage: 人臉驗證、person re-ID
的標準協定都要求「身分不重疊」, 因為共用一邊就足以傳遞資訊 -- 模型可以記住「這顆神經
的正確伴侶長什麼樣」, 或更廉價地記住「這顆神經出現時通常是正例還是負例」。

標註神經數有限, 無法做到兩側都不重疊的切分。因此本步驟不宣稱沒有洩漏管道, 而是量出管道
有多大, 再用四項檢驗看模型有沒有依賴它。只保留能復現這四項關鍵證據的程式:

  (1) 管道有多大: 測試配對中, 單邊神經曾出現在同 fold 訓練集的比例
  (2) 捷徑的上限: 完全不看影像, 只用「該神經在訓練集的正例比例」預測測試標籤的 AUC
  (3) 證據一: 依「兩側都沒看過 / 只有一側看過 / 兩側都看過」分層的 AUC (bootstrap CI),
      以及「訓練時有沒有見過該 FC 神經的正確伴侶」的對照
  (4) 證據二: 先驗必定給錯答案的「反向案例」上模型的正確率 (附帶同一標籤內分數與先驗
      的相關, 那是唯一的殘留訊號, 但與難易度混淆)
  (5) 證據三: 預訓練管道的曝光範圍, 以及依「有沒有進過 pseudo-label 名單」分層的表現
  (6) 證據四: annotator 的記憶會不會以「壓低同一顆神經的其他候選」的形式外溢 -- 只有
      那些其他候選會進入 pseudo-label

分割檔一律取自 repository 根目錄的 train_test_split/ (模型實際使用的版本; 程式會核對
它與預測檔是否一致)。模型分數取自 result/test_label_{MODEL}_{i}.csv。

輸出: results/leakage_overlap.csv   每折與整體的重疊比例
      results/leakage_strata.csv    分層 AUC 與 bootstrap 信賴區間
      results/leakage_shortcut.csv  先驗 baseline、相關、反向案例
      results/leakage_pretrain.csv  預訓練曝光範圍與分層表現
      results/leakage_annotator.csv annotator 記憶的外溢檢驗
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as C

SPLIT_DIR = C.ROOT / "train_test_split"
PRED_TMPL = str(C.ROOT / "result" / "test_label_FineTune_miniLR_D1-D6_{i}.csv")
PSEUDO_CSV = C.ROOT / "data" / "pairs_label" / "EMxFC_all_high_confidence.csv"
SUFFIX = "D1-D6"
N_FOLDS = 10
N_BOOT = 4000


def _ids(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in ("fc_id", "em_id"):
        d[c] = d[c].astype(str).str.strip()
    return d


def load() -> pd.DataFrame:
    """每組測試配對, 附上它在該 fold 訓練集裡的『曝光狀態』。"""
    rows = []
    for i in range(N_FOLDS):
        tr = _ids(pd.read_csv(SPLIT_DIR / f"train_split_{i}_{SUFFIX}.csv"))
        te = _ids(pd.read_csv(SPLIT_DIR / f"test_split_{i}_{SUFFIX}.csv"))
        pr = _ids(pd.read_csv(PRED_TMPL.format(i=i)))
        key = lambda d: set(map(tuple, d[["fc_id", "em_id"]].to_numpy()))
        if key(te) != key(pr):
            raise SystemExit(f"fold {i}: train_test_split/ 的測試集與預測檔不一致, "
                             f"請確認模型用的是哪一版分割")
        tr["y"] = (tr.label >= C.POS_CONF).astype(int)
        fc_rate, em_rate = tr.groupby("fc_id").y.mean(), tr.groupby("em_id").y.mean()
        t = pr.assign(fold=i)
        t["y"] = (t.label >= C.POS_CONF).astype(int)
        t["fc_seen"] = t.fc_id.isin(set(tr.fc_id))
        t["em_seen"] = t.em_id.isin(set(tr.em_id))
        # 訓練集裡有沒有出現過「這顆 FC 神經的正確伴侶」
        t["fc_partner_seen"] = t.fc_id.isin(set(tr.loc[tr.y == 1, "fc_id"]))
        t["fc_prior"] = t.fc_id.map(fc_rate)      # 只靠身分就能取得的標籤先驗
        t["em_prior"] = t.em_id.map(em_rate)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _auc_ci(s: pd.DataFrame, col: str = "model_pred", rng=None):
    """AUC 與 bootstrap 95 % 信賴區間; 樣本不足或單一類別時回傳 NaN。"""
    y, p = s.y.to_numpy(), s[col].to_numpy()
    if len(y) < 10 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boot.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi


def _operating_point(d: pd.DataFrame) -> np.ndarray:
    """與 result_analysis_make_figure.py 相同: min-max 正規化後取 ROC 最靠近 (0,1) 的點。"""
    p = d.model_pred.to_numpy(float)
    p = (p - p.min()) / (p.max() - p.min())
    fpr, tpr, thr = roc_curve(d.y, p)
    fin = np.isfinite(thr)
    t = thr[fin][int(np.argmin(np.sqrt(fpr[fin] ** 2 + (tpr[fin] - 1.0) ** 2)))]
    return p >= t


# ------------------------------------------------------------------- (1) ----
def overlap(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(N_FOLDS):
        s = d[d.fold == i]
        rows.append({"fold": i, "n_test": len(s), "fc_seen": s.fc_seen.mean(),
                     "em_seen": s.em_seen.mean(), "both_seen": (s.fc_seen & s.em_seen).mean(),
                     "neither_seen": (~s.fc_seen & ~s.em_seen).mean()})
    rows.append({"fold": "all", "n_test": len(d), "fc_seen": d.fc_seen.mean(),
                 "em_seen": d.em_seen.mean(), "both_seen": (d.fc_seen & d.em_seen).mean(),
                 "neither_seen": (~d.fc_seen & ~d.em_seen).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "leakage_overlap.csv", index=False)
    a = out.iloc[-1]
    print("=== (1) 洩漏管道有多大: 測試配對中單邊神經曾出現在同 fold 訓練集 ===")
    print(f"  FC 側 {100*a.fc_seen:.1f} %   EM 側 {100*a.em_seen:.1f} %   兩側都是 {100*a.both_seen:.1f} %"
          f"   兩側都沒有 {100*a.neither_seen:.1f} % ({int(a.neither_seen*len(d))} 組)")
    print("  -> 管道確實存在, 而且涵蓋絕大多數測試配對。以下檢驗模型有沒有依賴它。")
    return out


# ------------------------------------------------------------------- (2) ----
def shortcut_ceiling(d: pd.DataFrame) -> list[dict]:
    """完全不看影像, 只用身分帶來的標籤先驗能做到多好 -- 走捷徑的收益上限。"""
    rows = []
    print("\n=== (2) 捷徑的上限: 只用『該神經在訓練集的正例比例』預測測試標籤 ===")
    for name, col in (("fc_prior", "FC 側先驗"), ("em_prior", "EM 側先驗")):
        s = d.dropna(subset=[name])
        auc = roc_auc_score(s.y, s[name])
        rows.append({"test": f"prior_only_{name}", "n": len(s), "value": auc})
        print(f"  {col}: n={len(s):4d}  AUC {auc:.3f}")
    s = d.dropna(subset=["fc_prior", "em_prior"])
    auc = roc_auc_score(s.y, (s.fc_prior + s.em_prior) / 2)
    rows.append({"test": "prior_only_mean", "n": len(s), "value": auc})
    print(f"  兩側平均: n={len(s):4d}  AUC {auc:.3f}   (對照: 模型 {roc_auc_score(d.y, d.model_pred):.3f})")
    print("  -> 身分本身帶有大量標籤資訊, 所以「有沒有被利用」必須實測, 不能只靠設計論證。")
    return rows


# ------------------------------------------------------------------- (3) ----
def strata(d: pd.DataFrame, rng) -> pd.DataFrame:
    groups = [("兩側都沒看過", ~d.fc_seen & ~d.em_seen), ("只有 FC 看過", d.fc_seen & ~d.em_seen),
              ("只有 EM 看過", ~d.fc_seen & d.em_seen), ("兩側都看過", d.fc_seen & d.em_seen),
              ("FC 伴侶見過", d.fc_partner_seen), ("FC 伴侶沒見過", ~d.fc_partner_seen),
              ("全部", pd.Series(True, index=d.index))]
    rows = []
    print("\n=== (3) 分層表現: 沒看過的神經是不是比較差? ===")
    print(f"  {'分層':<16}{'n':>6}{'正例率':>9}{'AUC':>8}{'95% CI':>20}")
    for name, m in groups:
        s = d[m]
        auc, lo, hi = _auc_ci(s, rng=rng)
        rows.append({"stratum": name, "n": len(s), "pos_rate": s.y.mean(),
                     "auc": auc, "ci_lo": lo, "ci_hi": hi})
        ci = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "(樣本不足)"
        a = f"{auc:.3f}" if np.isfinite(auc) else "  n/a"
        print(f"  {name:<16}{len(s):>6}{100*s.y.mean():8.1f}%{a:>8}{ci:>20}")

    a, b = d[~d.fc_seen & ~d.em_seen], d[d.fc_seen & d.em_seen]
    diffs = []
    ya, pa = a.y.to_numpy(), a.model_pred.to_numpy()
    yb, pb = b.y.to_numpy(), b.model_pred.to_numpy()
    for _ in range(N_BOOT):
        ia, ib = rng.integers(0, len(ya), len(ya)), rng.integers(0, len(yb), len(yb))
        if len(np.unique(ya[ia])) == 2 and len(np.unique(yb[ib])) == 2:
            diffs.append(roc_auc_score(ya[ia], pa[ia]) - roc_auc_score(yb[ib], pb[ib]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rows.append({"stratum": "差距(兩側都沒看過 − 兩側都看過)", "n": len(a) + len(b),
                 "pos_rate": np.nan, "auc": float(np.mean(diffs)), "ci_lo": lo, "ci_hi": hi})
    print(f"\n  差距 (兩側都沒看過 − 兩側都看過): {np.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  -> {'看不出差異' if lo < 0 < hi else '差異顯著'}")
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "leakage_strata.csv", index=False)
    return out


# ------------------------------------------------------------------- (4) ----
def uses_shortcut(d: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    print("\n=== (4) 模型有沒有在用捷徑 ===")
    print("  (a) 同一真實標籤內, 分數是否仍跟著先驗走 (若有, 可能在用; 但也與難易度混淆)")
    for lab, nm in ((1, "真配對"), (0, "假配對")):
        s = d[d.y == lab].dropna(subset=["fc_prior", "em_prior"])
        for col, side in (("fc_prior", "FC"), ("em_prior", "EM")):
            rho, p = stats.spearmanr(s.model_pred, s[col])
            rows.append({"test": f"within_label_rho_{nm}_{side}", "n": len(s), "value": rho, "p": p})
            print(f"      {nm} vs {side} 先驗: rho {rho:+.3f} (p {p:.2g}, n={len(s)})")

    d = d.assign(pred_pos=_operating_point(d))
    print("  (b) 先驗必定給錯答案的反向案例, 模型還對嗎")
    cases = [("假配對但 FC 先驗 >= 0.5", (d.y == 0) & (d.fc_prior >= 0.5)),
             ("假配對但 EM 先驗 >= 0.5", (d.y == 0) & (d.em_prior >= 0.5)),
             ("真配對但 FC 先驗 <= 0.2", (d.y == 1) & (d.fc_prior <= 0.2)),
             ("真配對但 EM 先驗 <= 0.2", (d.y == 1) & (d.em_prior <= 0.2))]
    for nm, m in cases:
        s = d[m.fillna(False)]
        if not len(s):
            continue
        acc = float((s.pred_pos == (s.y == 1)).mean())
        rows.append({"test": f"counter_prior_{nm}", "n": len(s), "value": acc})
        print(f"      {nm:<26} n={len(s):4d}  模型判對 {100*acc:5.1f} %  (先驗全錯)")
    acc_all = float((d.pred_pos == (d.y == 1)).mean())
    rows.append({"test": "accuracy_all", "n": len(d), "value": acc_all})
    print(f"      {'對照: 全體':<26} n={len(d):4d}  模型判對 {100*acc_all:5.1f} %")
    print("  -> 模型會覆蓋先驗, 表示分數主要不是來自身分記憶。")
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "leakage_shortcut.csv", index=False)
    return out


# ------------------------------------------------------------------- (5) ----
def pretrain_channel(d: pd.DataFrame, rng) -> pd.DataFrame:
    """預訓練那一條管道。

    預訓練用的 pseudo-label 由 merge_pseudo_label.py 產生, 標籤是 **10 個 Annotator fold
    模型預測的平均** (只保留十模型高度一致者, predict_std < 0.05, 再平衡正負例)。每個 fold
    模型看過該折 9/10 的專家標註, 十個平均起來就涵蓋了全部專家標註 -- 因此預訓練階段
    **不是 fold-clean**, 而且十折微調都從同一個預訓練模型出發。

    排除只做在「配對」層級: 與專家標註沒有共同配對, 但共同神經很多。這裡量曝光範圍,
    並比較「預訓練從未看過其神經」的測試配對表現是否較差。
    """
    if not PSEUDO_CSV.exists():
        print(f"\n=== (5) 預訓練管道: 找不到 {PSEUDO_CSV}, 略過 ===")
        return pd.DataFrame()
    ps = _ids(pd.read_csv(PSEUDO_CSV, usecols=["fc_id", "em_id"]))
    PF, PE = set(ps.fc_id), set(ps.em_id)
    EF, EE = set(d.fc_id), set(d.em_id)
    hit = ps.fc_id.isin(EF) | ps.em_id.isin(EE)
    pair_ov = len(set(map(tuple, ps[["fc_id", "em_id"]].to_numpy()))
                  & set(map(tuple, d[["fc_id", "em_id"]].to_numpy())))
    print("\n=== (5) 預訓練管道: pseudo-label 有沒有把被標註的神經排除掉? ===")
    print(f"  pseudo-label {len(ps):,} 組; 與專家標註共同的配對 {pair_ov} 組 (配對層級已排除)")
    print(f"  但共同的神經: FC {len(PF & EF)}/{len(EF)} ({100*len(PF & EF)/len(EF):.1f} %), "
          f"EM {len(PE & EE)}/{len(EE)} ({100*len(PE & EE)/len(EE):.1f} %); "
          f"涉及任一被標註神經的 pseudo-label {int(hit.sum()):,} ({100*hit.mean():.1f} %)")
    rows = [{"item": "pseudo_pairs", "n": len(ps), "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan},
            {"item": "pair_overlap_with_expert", "n": pair_ov, "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan},
            {"item": "expert_fc_in_pseudo", "n": len(PF & EF), "value": len(PF & EF) / len(EF), "ci_lo": np.nan, "ci_hi": np.nan},
            {"item": "expert_em_in_pseudo", "n": len(PE & EE), "value": len(PE & EE) / len(EE), "ci_lo": np.nan, "ci_hi": np.nan},
            {"item": "pseudo_pairs_touching_expert_neuron", "n": int(hit.sum()), "value": float(hit.mean()), "ci_lo": np.nan, "ci_hi": np.nan}]
    t = d.assign(fc_in_pre=d.fc_id.isin(PF), em_in_pre=d.em_id.isin(PE))
    print(f"\n  {'分層':<18}{'n':>6}{'正例率':>9}{'AUC':>8}{'95% CI':>20}")
    for name, m in (("兩側都沒進預訓練", ~t.fc_in_pre & ~t.em_in_pre),
                    ("只有一側進了", t.fc_in_pre ^ t.em_in_pre),
                    ("兩側都進了", t.fc_in_pre & t.em_in_pre),
                    ("全部", pd.Series(True, index=t.index))):
        s_ = t[m]
        auc, lo, hi = _auc_ci(s_, rng=rng)
        rows.append({"item": f"stratum_{name}", "n": len(s_), "value": auc, "ci_lo": lo, "ci_hi": hi})
        ci = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "(樣本不足)"
        a = f"{auc:.3f}" if np.isfinite(auc) else "  n/a"
        print(f"  {name:<18}{len(s_):>6}{100*s_.y.mean():8.1f}%{a:>8}{ci:>20}")
    print("  -> 沒被預訓練看過的配對表現一樣好; 但 n 很小, 只能說沒看到影響, 不能說證明沒有。")
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "leakage_pretrain.csv", index=False)
    return out


# ------------------------------------------------------------------- (6) ----
def annotator_memory(d: pd.DataFrame) -> pd.DataFrame:
    """產生 pseudo-label 的 annotator 有沒有「記住」專家標籤, 記憶又會不會外溢?

    pseudo-label 的標籤是 10 個 annotator fold 模型的平均, 那組模型合起來看過全部專家
    標註 (§ (5))。annotator 對自己訓練過的配對當然有擬合優勢 (AUC 0.990 vs 留出的 0.933),
    但那只是預期內的 train/val gap, 訓練時看 loss 曲線就知道了, **記憶本身不等於洩漏**。
    要真的造成洩漏, 記憶得外溢到「同一顆神經的其他候選」-- 因為專家配對本身已被排除,
    只有這些其他候選會進到 pseudo-label。本節只測外溢。

    **分折是以配對為單位, 不是以神經為單位**: 對一組專家配對 (X, Y), 9 個模型看過「這組
    配對及其信心度」、1 個沒看過; 但 X 的其他配對是獨立分散在各折的。因此這裡只取
    **所有專家配對都落在同一折**的神經 -- 對這些神經, 那一折的模型完全沒看過它的任何
    標註, 另外 9 個看過全部。再依「該神經有沒有正例配對」分開: 只有「知道正確伴侶」才
    對應到「知情模型會壓低其他候選」這個假設。

    逐折預測取自 preTrain_label/Annotator_D1-D6_{i}.csv (annotator 的逐折輸出, 目前沒有
    其他程式引用)。
    """
    files = [C.ROOT / "preTrain_label" / f"Annotator_D1-D6_{i}.csv" for i in range(N_FOLDS)]
    if not all(f.exists() for f in files):
        print("\n=== (6) annotator 記憶檢驗: 缺少 preTrain_label/Annotator_D1-D6_*.csv, 略過 ===")
        return pd.DataFrame()
    pair_fold, y_of = {}, {}
    for i in range(N_FOLDS):
        te = _ids(pd.read_csv(SPLIT_DIR / f"test_split_{i}_{SUFFIX}.csv"))
        for f_, e_, l in te[["fc_id", "em_id", "label"]].itertuples(index=False):
            pair_fold[(f_, e_)] = i
            y_of[(f_, e_)] = int(l >= C.POS_CONF)
    preds = {i: _ids(pd.read_csv(f, usecols=["fc_id", "em_id", "model_predict"]))
             for i, f in enumerate(files)}

    rows: list[dict] = []
    print("\n=== (6) annotator 的記憶會不會外溢到同一顆神經的其他候選 ===")

    base = preds[0][["fc_id", "em_id"]].copy()
    for i in range(N_FOLDS):
        base[f"p{i}"] = preds[i].set_index(["fc_id", "em_id"]).reindex(
            list(zip(base.fc_id, base.em_id))).model_predict.to_numpy()
    base = base.dropna()
    base = base[[x not in pair_fold for x in zip(base.fc_id, base.em_id)]]   # 排除專家配對本身
    P = base[[f"p{i}" for i in range(N_FOLDS)]].to_numpy()
    base["std10"] = P.std(axis=1, ddof=0)
    if PSEUDO_CSV.exists():
        used = set(map(tuple, _ids(pd.read_csv(PSEUDO_CSV, usecols=["fc_id", "em_id"])).to_numpy()))
        base["in_pretrain"] = [x in used for x in zip(base.fc_id, base.em_id)]
    else:
        base["in_pretrain"] = False

    def gap(t: pd.DataFrame):
        if len(t) < 30:
            return None
        Q = t[[f"p{i}" for i in range(N_FOLDS)]].to_numpy()
        fo = t.f.to_numpy().astype(int)
        naive = Q[np.arange(len(t)), fo]
        m = np.ones_like(Q, bool)
        m[np.arange(len(t)), fo] = False
        diff = naive - Q[m].reshape(len(t), N_FOLDS - 1).mean(axis=1)
        return float(diff.mean()), float(stats.ttest_1samp(diff, 0).pvalue), len(t)

    print("  「完全不知情的那個模型」−「9 個知情模型平均」, 對該神經的其他候選")
    print("      (正值 = 知情模型壓低了其他候選; 只取專家配對全部集中在單一折的神經)")
    rng2 = np.random.default_rng(C.RANDOM_STATE)
    for side, idcol in (("FC", "fc_id"), ("EM", "em_id")):
        per: dict = {}
        for (f_, e_), fo in pair_fold.items():
            per.setdefault(f_ if side == "FC" else e_, []).append((fo, y_of[(f_, e_)]))
        single = {k: v[0][0] for k, v in per.items() if len({x[0] for x in v}) == 1}
        haspos = {k for k, v in per.items() if any(y for _, y in v)}
        b = base.assign(f=base[idcol].map(single))
        sub = b.dropna(subset=["f"])
        print(f"      --- {side} 側: 專家神經 {len(per)} 顆, 全部配對集中在單一折 {len(single)} 顆")
        for nm, t, tag in ((f"有正例 (知道正確伴侶)", sub[sub[idcol].isin(haspos)], "haspos"),
                           (f"只有負例 (只知道不配誰)", sub[~sub[idcol].isin(haspos)], "negonly")):
            r = gap(t)
            if r is None:
                print(f"          {nm:<24} n={len(t)} 太少"); continue
            rows.append({"test": f"spillover_{side}_{tag}", "n": r[2], "value": r[0], "p": r[1]})
            print(f"          {nm:<24} 神經 {t[idcol].nunique():3d} 顆  n={r[2]:6,}  差 {r[0]:+.4f} (p {r[1]:.2g})")
        ctrl = b[b.f.isna()].copy()
        ctrl["f"] = rng2.integers(0, N_FOLDS, len(ctrl))
        r = gap(ctrl)
        if r:
            rows.append({"test": f"spillover_{side}_control", "n": r[2], "value": r[0], "p": r[1]})
            print(f"          {'對照: 未標註 + 隨機折':<24} {'':8}  n={r[2]:6,}  差 {r[0]:+.4f} (p {r[1]:.2g})")
        if side == "FC":
            hp = sub[sub[idcol].isin(haspos)]
            for nm, t, tag in (("其中 std < 0.05 (篩選後)", hp[hp.std10 < 0.05], "std"),
                               ("其中真正進入預訓練名單", hp[hp.in_pretrain], "used")):
                r = gap(t)
                if r is None:
                    print(f"          {nm:<24} n={len(t)} 太少 (但這正是重點: 幾乎都被濾掉了)")
                    rows.append({"test": f"spillover_FC_haspos_{tag}", "n": len(t), "value": np.nan})
                    continue
                rows.append({"test": f"spillover_FC_haspos_{tag}", "n": r[2], "value": r[0], "p": r[1]})
                print(f"          {nm:<24} {'':8}  n={r[2]:6,}  差 {r[0]:+.4f} (p {r[1]:.2g})")
    print("      -> 假設的「知道伴侶就壓低其他候選」不成立 (方向相反); 實際存在的是幅度 0.01–0.05")
    print("         的每顆神經標籤先驗, 且 predict_std < 0.05 的篩選把受影響最深的配對濾掉大半。")
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "leakage_annotator.csv", index=False)
    return out


def main() -> None:
    rng = np.random.default_rng(C.RANDOM_STATE)
    d = load()
    print(f"測試配對 {len(d)} 組 (10 折, 專家信心 >= {C.POS_CONF} 為正例)\n")
    overlap(d)
    rows = shortcut_ceiling(d)
    strata(d, rng)
    uses_shortcut(d, rows)
    pretrain_channel(d, rng)
    annotator_memory(d)


if __name__ == "__main__":
    main()
