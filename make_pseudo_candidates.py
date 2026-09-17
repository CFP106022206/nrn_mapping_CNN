"""產生 pseudo label 用的候選名單：從 prescreening 結果剔除所有被人工標註過的神經。

為什麼需要這一步
----------------
交叉驗證是以「配對」為單位切分的，annotator 的十個 fold 模型合起來看過**全部**專家標註。
只要候選池裡留著「某一側是被標註過的神經」的配對，annotator 對它們的預測就帶有身分記憶；
而 pseudo label 正是由這些預測產生、再拿去 pre-train，預訓練階段因此不是 fold-clean。

配對層級的排除不夠：專家配對 (X, Y) 被排掉了，但「X 配其他候選」與「Y 配其他候選」全都
還在池裡，記憶正是經由這些配對外溢。本步驟改成**神經層級**——只要 `fc_id` 或 `em_id`
其中一邊出現在人工標註中，該配對就整筆剔除。

跑完之後把 `Model_predict.py` 的 `Config.pairs_csv` 指向輸出檔，再跑十折 annotator 預測。

用法
----
    python3 make_pseudo_candidates.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# prescreening 產生的候選配對，必須有 fc_id / em_id 兩欄
CANDIDATES_CSV = ROOT / "data" / "pairs_label" / "EMxFC_6KK_last.csv"
OUT_CSV = ROOT / "data" / "pairs_label" / "EMxFC_6KK_last_noexpert.csv"

# 人工標註的來源。TOTAL 是去重後的權威版本；labeled_info/D*_conf.csv 是原始檔，
# 一併取聯集，這樣日後補標了新神經而忘記更新 TOTAL 也不會漏掉。
LABEL_TOTAL_CSV = ROOT / "data" / "pairs_label" / "D1-D6_total_conf.csv"
LABEL_RAW_GLOB = str(ROOT / "labeled_info" / "D*_conf.csv")


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def expert_neurons() -> tuple[set[str], set[str]]:
    """回傳 (被標註過的 FC id, 被標註過的 EM id)。

    merge_pseudo_label.py 也會 import 這個函式做最後把關，所以放在這裡而不是 main 裡。
    """
    fc: set[str] = set()
    em: set[str] = set()
    sources = [LABEL_TOTAL_CSV] + [Path(p) for p in sorted(glob.glob(LABEL_RAW_GLOB))]
    for p in sources:
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str)
        cols = {c.lower(): c for c in d.columns}
        if "fc_id" not in cols or "em_id" not in cols:
            print(f"  ! {p.name} 沒有 fc_id / em_id 欄位，略過")
            continue
        fc |= set(_norm(d[cols["fc_id"]]))
        em |= set(_norm(d[cols["em_id"]]))
    if not fc or not em:
        raise RuntimeError(f"讀不到任何人工標註神經，檢查 {LABEL_TOTAL_CSV}")
    return fc, em


def main() -> None:
    fc_exp, em_exp = expert_neurons()
    print(f"人工標註過的神經：FC {len(fc_exp)} 顆，EM {len(em_exp)} 顆")

    c = pd.read_csv(CANDIDATES_CSV, dtype=str)
    missing = {"fc_id", "em_id"} - set(c.columns)
    if missing:
        raise RuntimeError(f"{CANDIDATES_CSV.name} 缺少欄位 {missing}")
    c["fc_id"] = _norm(c["fc_id"])
    c["em_id"] = _norm(c["em_id"])
    n0 = len(c)
    print(f"候選池 {CANDIDATES_CSV.name}：{n0:,} 組，"
          f"FC {c.fc_id.nunique():,} 顆，EM {c.em_id.nunique():,} 顆")

    fc_hit = c.fc_id.isin(fc_exp)
    em_hit = c.em_id.isin(em_exp)
    drop = fc_hit | em_hit
    print(f"  FC 側命中 {int(fc_hit.sum()):,} 組（{c.loc[fc_hit, 'fc_id'].nunique()} 顆專家 FC 出現在池中）")
    print(f"  EM 側命中 {int(em_hit.sum()):,} 組（{c.loc[em_hit, 'em_id'].nunique()} 顆專家 EM 出現在池中）")
    print(f"  任一側命中 → 剔除 {int(drop.sum()):,} 組（{100 * drop.mean():.2f} %）")

    out = c.loc[~drop].reset_index(drop=True)
    # 事後驗證：剩下的池子必須一顆專家神經都不剩
    assert not out.fc_id.isin(fc_exp).any(), "仍有專家 FC 殘留"
    assert not out.em_id.isin(em_exp).any(), "仍有專家 EM 殘留"

    out.to_csv(OUT_CSV, index=False)
    print(f"\n保留 {len(out):,} 組（{100 * len(out) / n0:.2f} %），"
          f"FC {out.fc_id.nunique():,} 顆，EM {out.em_id.nunique():,} 顆")
    print(f"輸出：{OUT_CSV}")
    print("\n下一步：把 Model_predict.py 的 Config.pairs_csv 指向這個檔案，再跑 Model_predict.sh")


if __name__ == "__main__":
    main()
