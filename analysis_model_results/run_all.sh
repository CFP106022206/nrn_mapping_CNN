#!/usr/bin/env bash
# 重現 D1 (projection) vs D2 (dense) 的全部量化分析。
#   bash analysis_dense_vs_projection/run_all.sh
# 每一步都寫進 analysis_dense_vs_projection/results/, 且可重複執行。
#
# 前置: NBLAST 基準分數需要 navis, 裝在獨立的 conda 環境:
#   conda create -n nblast python=3.10 -y && conda run -n nblast pip install navis
#   conda run -n nblast python analysis_dense_vs_projection/run_nblast_official.py --emit-figure-csv
# 該步驟只需跑一次, 輸出 results/nblast_official.csv (s08/s09/s10 依賴它)。
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f results/nblast_official.csv ]; then
  echo "缺 results/nblast_official.csv — 請先執行:" >&2
  echo "  conda run -n nblast python $(pwd)/run_nblast_official.py --emit-figure-csv" >&2
  exit 1
fi

python3 s01_build_neuron_lists.py      # 兩組的神經名單
python3 s02_neuropil_metrics.py        # neuropil 佔位描述子 (FC)
python3 s03_morphology_metrics.py      # 骨架幾何 + 多尺度密度 (FC + EM), ~5 min
python3 s04_group_contrast.py          # 所有描述子依可分離度排名
python3 s05_size_control.py            # 尺寸配對後 neuropil 訊號是否還在
python3 s06_region_criteria.py         # D1 的解剖身分與納入門檻
python3 s07_selection_rule.py          # 交叉驗證的納入條件, ~3 min
python3 s08_expert_and_nblast.py       # 專家信心 + 官方 NBLAST 可分離度
python3 s09_soma_and_strahler.py       # soma 距離、Strahler 稀釋、cable 分箱
python3 s10_em_sponge_effect.py        # hemibrain 海綿效應 (NBLAST 失效的主因)
python3 s12_cnn_sponge_effect.py       # MorphoMatcher 並排: 海綿效應是否為 NBLAST 特有
python3 s11_figures.py                 # 論文用圖 fig1-fig5

echo
echo "完成。表格在 results/, 圖在 results/figures/。"
