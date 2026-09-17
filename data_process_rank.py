"""排序微調：用 InfoNCE + 專家二元損失聯合重訓打分器（卷積層預設凍結，可選擇一起訓練）。

與 `data_process_fineTune.py` 的關係
-----------------------------------
前處理、三視圖載入、模型建圖全部沿用既有那一套
（`swc_util._pad_to_same_size` / `_resize_to_50`、`make_numpy_from_standard_views`、
`model.MVCNN_Siamese`），確保與線上服務的打分路徑逐位元一致。
不同的是訓練目標與可訓練範圍，理由如下。

為什麼這樣設計（每一條都來自實驗，見 `analysis_model_results/FINDINGS.md` §10）
--------------------------------------------------------------------------
1. **預設凍結 trunk。** 參數分布是反直覺的：trunk（conv1-4 + 6 個 BN）只有 66 528 個
   參數（1.4 %），head（Dense 18432→256 + BN + Dense 256→1）有 4 720 129 個（98.6 %）。
   階段 2 的探針證明卷積特徵夠用，問題出在上面那個佔絕大多數參數的分類頭
   是為錯的目標訓練的。見第 8 點——這個預設值得被挑戰。

   ⚠️ **凍結時 trunk 的 forward 走在 GradientTape 外面**，分塊算完再把特徵餵進 head。
   不這樣做會 OOM：負例批次是 128 x 32 = 4 096 對，tape 要保留卷積活化值
   （4 096 對 x 2 側 x 3 視角 = 24 576 張 50x50 圖，光 conv1 就 7.9 GB，四層約 24 GB），
   實測在 48 GB 的卡上炸掉（已配置 40.4 GiB）。凍結時那些活化值根本用不到——
   trunk 的權重不在 `train_vars` 裡。移到 tape 外之後記憶體降到 1 GB 級、也更快。
   `--tune-trunk` 需要 trunk 的梯度，只能走端到端，所以會自動把
   `batch_anchors` 降到 16。

2. **排序損失（InfoNCE）而非二元交叉熵。** 全庫檢索要的是「這顆 FC 的正解排第一」，
   而不是「這一對像不像」。專家標註的正例率 51 %，全庫檢索是 1/2900——先驗差三個數量級。

3. **⚠️ 跨 FC 負例（最關鍵）。** 若負例只從同一顆 FC 的池抽，
   「把某顆 EM 全域推高」對損失是免費的午餐。實測：從既有 head 以低 lr 出發訓練的探針，
   其 rank-1 有 **39.4 %** 可以被「完全忽略 FC、只用每顆 EM 的平均分」這個虛無模型重現
   （KC 上 46.6 %），也就是它學的是 EM 先驗而不是形態比對。
   對策是加入 **cross 負例**：從這顆 FC 自己的池裡，挑那些「是別顆 FC 的正例、
   但對本 FC 型別不符」的 EM。這正是「全域受歡迎但對這顆 FC 錯」的集合。
   （教科書寫法是拿同批其他 FC 的正例當 in-batch 負例，但本專案的前處理
   `_pad_to_same_size` 是**成對相依**的，同一顆 EM 配不同 FC 時像素不同，
   沒辦法零成本重用特徵，所以改成上面這個等效作法。）

4. **保留專家二元項。** 排序損失只看分數差，對全域平移無感：模型可把分數壓進
   [0.30, 0.52]，排序完全正確但 0.5 門檻就廢了。專家標註同時也是**個體層級**的
   （每顆 FC 中位只給 1 個正例），比型別層級嚴格，是防止被弱正例拉偏的錨。
   兩項必須**同步**聯合訓練——分兩階段的話，後一階段會重新施加自己的校準，
   把前面的修正洗掉（那正是現行 annotator→pseudo→pretrain→finetune 的毛病）。

   ⚠️ **`w_bce` 預設 10.0 而非 1.0。** 兩項都是對 128 個樣本取平均，資料量差異
   （78 500 對 vs 1 097 對）**不造成稀釋**；失衡來自 `tau=0.07` 把 InfoNCE 的梯度
   放大約 14 倍。實測梯度範數比 **19.6x**，`w_bce=1` 等於讓排序項以 20:1 壓過校準項。
   訓練時每 10 個 epoch 會印出實測比值。

5. **一般 BCE，不用 focal。** `BinaryFocalCrossentropy(gamma=2.0)` 會**降低**容易樣本
   的權重；這批負例是「模型極度自信但錯」的（中位分數 0.878），focal 會**放大**其梯度。

6. **每 epoch 重抽負例；增強逐樣本施加。** head 有 4.7 M 參數、錨點只有數千個，
   過擬合風險是真的。探針版用整輪固定的一次抽樣，這裡每 epoch 重抽。
   ⚠️ 增強必須**每個樣本各抽一組 (rot, flip)**（兩側同步），不能整個 epoch 共用一組——
   那樣 60 個 epoch 只是從 8 種變換抽 60 次、epoch 內零多樣性，而且 head 的 BN
   在一個 epoch 內只看到單一朝向，推論卻固定 rot=0/flip=False，統計量會對不上。
   成本為零：每顆神經只有 8 種變換，`PairEncoder` 全部快取。
   （尚未加 FC/EM swap。既有 annotator 靠 swap 學兩分支的對稱性，這裡少了 2x 資料量；
   不是 bug，但若觀察到過擬合，這是第一個該加回來的東西。）

7. **從頭初始化 head。** 從既有 head 以低 lr 出發會退化成學 EM 先驗（見第 3 點）。

8. **trunk 凍不凍結是開放問題，值得兩種都跑。** 預設凍結，因為階段 2 證明
   annotator 的卷積特徵**夠用**——但那沒有證明它**最好**。
   那個 trunk 當初只用 `train_split_0` 的 1 097 對（增強後 10 970）訓練，
   而現在每 epoch 就有約 78 500 對可用、母體達 185 萬對，監督量差約 170 倍，
   且 trunk 只有 66 528 個參數。`--tune-trunk` 會連卷積層一起訓練。
   代價是多一次反向傳播（約 2 倍時間），以及過擬合風險（由專家 BCE 項把關）。

9. **有 dev 集。** 從 train 半邊再切 15 %，每 5 個 epoch 量一次三元組正確率並保留最佳權重。
   不切的話 eval 半邊會在調 `w_bce` / `tau` / `epochs` 的過程中變成 dev 集。

執行
----
環境：conda env `ming`（`python3` 已指向它，不必先 activate）。

    python3 data_process_rank.py                        # fold 0、凍結 trunk（預設）
    python3 data_process_rank.py --tune-trunk           # 連卷積層一起訓練
    python3 data_process_rank.py --fold 3 --epochs 80

參數
----
| 參數 | 預設 | 說明 |
|---|---|---|
| `--fold` | 0 | 用哪一折。**會連帶決定 trunk 的來源**（`Annotator_D1-D6_{fold}`）與專家 split |
| `--epochs` | 60 | 訓練輪數。dev 會挑最佳 checkpoint，設太大不會直接害到結果 |
| `--seed` | 3407 | 影響 FC 切分、負例抽樣、增強。**不影響 TF 的權重初始化**，所以重跑仍有變異 |
| `--tune-trunk` | 關 | 連 conv1-4 一起訓練（trunk 的 BN 仍保持推論模式）。約 2 倍耗時 |
| `--head-init` | `scratch` | `existing` 會從 annotator 既有 head 出發。**不建議**：已證實會退化成學 EM 先驗 |
| `--no-augment` | 關 | 關閉幾何增強。只在除錯時用 |
| `--w-bce` | 10.0 | 專家二元項的權重。不是 1.0，理由見第 4 點 |
| `--tau` | 0.07 | InfoNCE 溫度。調它會連動改變 `w_bce` 該設多少（梯度比 ∝ 1/τ） |

只能改 `Config` 不能從命令列改的：`n_pos`(4)、`k_within`(24)、`k_cross`(8)、
`batch_anchors`(128，`--tune-trunk` 時自動降到 16)、`lr`(1e-3)、`dev_ratio`(0.15)、
`eval_every`(5)、`train_ratio`(0.7)。

⚠️ **FC 切分隨 fold 轉**（`split_seed = seed + fold * 1000`）。
先前固定用 `seed` 會讓 10 折共用同一個檢索評估集，折間的檢索數字不是獨立估計。
現在 fold 0 與 fold 3 的 eval 只重疊 109/383。

⚠️ **評估指標用的是型別標籤，不是專家標註。**
檢索表（MRR / rank-1 / median_rank / p@5 / em_prior）一律以
`type_label == 1.0`（FC 的 VFB 策展型別 == EM 的 neuPrint 策展型別）為準——
這是**弱**標籤：KC 的池裡中位有 111 顆同型 EM，答對只要命中任一顆。
專家標註只用在訓練的 BCE 項與**守門指標**（專家測試集 AUC），後者才是個體層級。
所以「rank-1 56 %」讀成「排第一的 EM **型別對**」，不是「就是那一顆」。

耗時（實測推算，RTX PRO 5000）
----
| | 凍結 trunk | `--tune-trunk` |
|---|---|---|
| 訓練 60 epochs | 15 分（15 s/epoch） | 30 分（30 s/epoch） |
| 最終檢索評估（638 顆 FC、185 萬對） | 5–12 分 | 5–12 分 |
| **合計** | **約 25 分鐘** | **約 40 分鐘** |

記憶體：三視圖快取上限 0.64 GB。

訓練途中要盯的三個訊號
----
    epoch  10  rank 0.2494  bce 0.3812  |grad| rank/bce = 8.2/4.1 = 2.0x
               dev 0.612 (within 0.741 / cross 0.688)

1. **`cross` 明顯落後 `within`** -> 模型在學 EM 先驗而不是形態。
   這是提早抓到退化的唯一辦法（dev 的 EM 有 100 % 在 train 出現過，
   整體正確率抓不到這件事）。
2. **`|grad|` 比值單調下滑** -> InfoNCE 收斂後梯度衰減而 BCE 不會，
   後期可能變成 BCE 主導。若確實如此，下次調低 `--w-bce`。
3. **結束時印「採用 epoch N 的權重」且 N < epochs** -> 後段過擬合，可以調低 `--epochs`。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "analysis_external_validation"))
sys.path.insert(0, str(PROJECT / "analysis_hubness_debias"))


@dataclass
class Config:
    fold: int = 0
    seed: int = 3407

    # --- 起點 ---------------------------------------------------------
    # trunk 取 annotator：全庫檢索 31.0 % vs fine-tune 22.0 %，
    # 階段 2 的探針也是在它上面做的。
    #
    # ⚠️ **必須跟著 fold 走**。`Annotator_D1-D6_{fold}` 是在 `train_split_{fold}`
    #    上訓練的，沒看過 `test_split_{fold}`。若固定用 fold 0 的權重去跑其他 fold，
    #    trunk 就看過該 fold 測試集的配對，專家測試集那道守門會失效。
    base_weights_tpl: str = "./Annotator_Model/Annotator_D1-D6_{fold}.weights.h5"
    input_size: tuple[int, int, int] = (50, 50, 3)
    freeze_trunk: bool = True

    @property
    def base_weights(self) -> str:
        return self.base_weights_tpl.format(fold=self.fold)

    # --- 型別標籤與候選池 ---------------------------------------------
    # 型別標籤（1.0/0.0）來自 VFB + neuPrint，與模型無關，任何 fold 都通用。
    # 但檔裡的 `score` 欄是 `Annotator_D1-D6_0` 打的分，retrieval 報表用它當基線；
    # 跑 fold != 0 時該基線與 trunk 來源不同源，只影響對照欄，不影響訓練。
    pool_scores: str = "analysis_hubness_debias/scores/pool_scores_annotator.parquet"
    fc_views_dir: str = "./data/standard_views/FC"
    em_views_dir: str = "./data/standard_views/EM"

    # --- 專家標註 -----------------------------------------------------
    split_dir: str = "./train_test_split"
    split_name: str = "D1-D6"

    # --- 取樣（每 epoch 重抽）-----------------------------------------
    n_pos: int = 4          # 每顆 FC 的錨點數
    k_within: int = 24      # 同 FC 池內的型別不符負例
    k_cross: int = 8        # 「別顆 FC 的正例、但對本 FC 型別不符」的負例

    # 每顆 FC 的負例池上限（0 = 用整個型別不符池，中位數千個）。
    # 階段 2 的探針是每顆 FC 預先固定抽 64 個、每步再從中抽 K 個
    # （probe_frozen_head.py:49-50）；正式版每步從整個池重抽，是難得多的任務。
    # 這是探針與正式版之間**與增強無關**的第二個差異，回歸測試時要一起對齊。
    neg_pool: int = 0

    # 正例只抽一次、整個訓練過程固定（探針就是這樣，而正式版每 epoch 重抽）。
    # 這一項是先驗爆炸的頭號嫌疑：KC 的池裡中位有 111 顆同型 EM，同族 FC 的
    # 同型集合高度重疊；每 epoch 重抽等於 60 個 epoch 幾乎覆蓋整個同型集合，
    # 反覆訓練模型「把整個 KCg-m EM 集合推高」——那就是 EM 先驗本身。
    # 實測：探針（固定）em_prior 2.74 %，正式版（重抽）34.60 %，同切分同超參數。
    fixed_anchors: bool = False

    # EM 型別 hold-out：這些 np_type 的 EM 完全不進訓練（不當正例也不當負例）。
    # FC 層級的切分不夠嚴格——實測每顆 eval FC 的正例中位 100 % 也是訓練正例，
    # 所以 EM 先驗可以直接轉移過去。整個型別拿掉之後，那些 EM body 從沒被推高過，
    # 先驗在結構上無法轉移，剩下的增益只能是形態。
    # 正例完全落在被 hold out 型別裡的 FC 會因為沒有正例而自動退出訓練，
    # 它們正是乾淨的測試集（報表會單獨列出 `holdout:<型別>` 這一層）。
    holdout_emtypes: str = ""       # 逗號分隔，如 "LC12,KCab-s,DL2d_adPN"

    # 只在報表裡多列出這些型別的分層，**不影響訓練**。用途是 hold-out 的控制組：
    # 同一批 FC、但模型訓練時看過那些 EM 型別，才能把「hold-out 殺死它」
    # 與「這些 FC 本來就難」分開。
    report_emtypes: str = ""

    # 從頭訓練時 trunk 用的正規化（只在 --trunk-init scratch 時有效）。
    # 必須是 "gn"：`--tune-trunk` 會把 trunk 的 BN 鎖在推論模式，從頭訓練時
    # 它們停在初始統計量、從未正規化；讓 BN 正常訓練又會讓正例/負例/專家配對
    # 三次 forward 用不同的批次統計量互相比較（head BN 那類 bug）。
    # 從頭訓練一律用 model.MVCNN_Siamese_3View（修正了視角切片）。
    trunk_norm: str = "gn"

    # 評估時一併打分的對照模型（MVCNN_Siamese_3View 結構），例如新訓練的
    # "./Annotator_Model/Annotator3v_D1-D6_{fold}.weights.h5"。檔案不存在就略過。
    compare_weights_tpl: str = ""
    eval_bs: int = 1024

    # --- 損失 ---------------------------------------------------------
    tau: float = 0.07
    w_rank: float = 1.0

    # ⚠️ 不是 1.0。兩項損失都是對 128 個樣本取平均，所以「InfoNCE 有 78 500 對、
    #    BCE 只有 1 097 對」**不會**稀釋 BCE——每步的樣本數相同。
    #    真正的失衡來自 tau：InfoNCE 的 logits 被除以 0.07，梯度是
    #    (softmax − onehot)/τ，而 BCE 對 logit 的梯度是 (sigmoid(z) − y)、上界為 1。
    #    實測梯度範數比（fold 0、batch 32）：
    #        tau=0.07 -> 19.6x     tau=0.2 -> 4.1x     tau=1.0 -> 0.6x
    #    w_bce = 1.0 等於讓排序項以約 20:1 壓過校準項，會架空「BCE 當錨」的設計目的。
    #    10.0 讓比值回到約 2:1——排序仍是主目標，但校準有實質影響力。
    #    訓練時每 10 個 epoch 會印出實測比值，可據以調整。
    w_bce: float = 10.0

    # --- 訓練 ---------------------------------------------------------
    epochs: int = 60
    batch_anchors: int = 128
    lr: float = 1e-3
    augment: bool = True

    # ⚠️ 從 train 半邊再切一份 dev，用來挑 checkpoint 與調超參。
    #    不切的話，唯一的回饋訊號就是最後那張 retrieval 表，而 w_bce / tau / epochs
    #    都要調——調過幾輪之後 eval 半邊就變成 dev 集，不再是乾淨的 held-out。
    #    eval 半邊在整個流程中只在最後碰一次。
    dev_ratio: float = 0.15
    eval_every: int = 5          # 每幾個 epoch 量一次 dev 並更新最佳權重

    # winnable FC 有多少比例進 train（其餘進 eval）。
    # 0.5 是為了讓 eval 數字與 headline（全 1 276 顆的 31.0 %）可比而選的，
    # 不是為訓練效果最佳化。fold 0 實測 dev 在 epoch 35 見頂後過擬合，
    # 顯示訓練資料是瓶頸，所以提高到 0.7；代價是 eval 從 638 降到 383，
    # 測量精度下降——因此檢索報表一律附 cluster bootstrap 的 95 % CI。
    train_ratio: float = 0.7

    # ⚠️ FC 切分的種子**帶上 fold**。先前固定用 cfg.seed，導致 10 折共用同一個
    #    檢索評估集——折與折之間只有 trunk 來源與專家 split 不同，
    #    檢索數字不是 10 個獨立估計，不能拿來算跨折 CI。
    @property
    def split_seed(self) -> int:
        return self.seed + self.fold * 1000

    # head 的初始化。預設 scratch，理由見檔頭第 7 點。
    # ⚠️ 那個理由有時效性：existing 會退化成學 EM 先驗，是因為當時的 InfoNCE
    #    只從同一顆 FC 的池抽負例，把某顆 EM 全域推高是免費的。加入 cross 負例後
    #    這條路已被堵死，existing 未必還會退化。值得跑完 scratch 後比一次，
    #    比較時務必看 retrieval 表裡的 em_prior_rank1_pct 那一欄。
    head_init: str = "scratch"      # scratch | existing

    # trunk 的初始權重。"annotator" = 載入 Annotator_D1-D6_{fold}（預設）；
    # "scratch" = 完全隨機初始化，整個打分器不繼承任何 annotator 的東西。
    # 用途是把「EM 先驗是不是被 annotator 權重帶進來的」這個問題問到底：
    # 完全從頭訓練若仍勝不過 annotator 基線，問題就不在繼承的權重，
    # 而在訓練資料或目標本身。scratch 會自動解凍 trunk（凍結一個隨機
    # 特徵抽取器等於隨機投影，不是這裡要問的問題）。
    trunk_init: str = "annotator"   # annotator | scratch

    # 強制評估指定 epoch 的權重，而不是 dev 最佳的那一個（0 = 照舊用 dev 最佳）。
    # dev 只有 133 顆 FC，挑最佳 epoch 帶有樂觀偏差，而且 KC 佔多數會主導選擇，
    # 少數族（ALPN）可能要更久才學得起來。訓練是確定性的，所以
    # `--epochs N --pick-epoch N` 能重現任一 epoch 的狀態拿去評估。
    pick_epoch: int = 0

    # 診斷用：載入既有 head、跳過訓練（配 --epochs 0），只做 BN 重校與評估。
    # 用來回答「舊 checkpoint 的數字有多少是被 BN moving 統計錯配拖累的」，
    # 不必重訓。recalib = 要跑幾個 128 筆的批次（momentum 0.9，50 批之後
    # 舊狀態只剩 0.9^50 ≈ 0.005）；recalib_src 決定用哪種分布重校。
    load_head: str = ""
    recalib: int = 0
    recalib_src: str = "pool"       # pool | expert

    # --- 輸出 ---------------------------------------------------------
    save_dir: str = "./RankTune_Model"
    result_dir: str = "./result"
    model_name: str = "RankTune_annotator"


# ---------------------------------------------------------------- 前處理

def _aug(v: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    """對 (3,H,W) 的三視圖施加幾何增強。同一組 (rot, flip) 必須套用在配對的兩側。"""
    out = np.rot90(v, rot, axes=(1, 2))
    if flip:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


class PairEncoder:
    """把 (fc_id, em_id) 逐對轉成模型輸入。

    ⚠️ `_pad_to_same_size` 以「這一對裡較大的那張」為準補零，是**成對相依**的，
    所以不能 per-neuron 快取 50x50 的圖（`nrn_service/scoring.py:8-10` 有記載）。
    這裡只快取原始三視圖，padding 與縮放仍逐對做。
    """

    def __init__(self, cfg: Config) -> None:
        from nrn_service.config import ServiceConfig
        from nrn_service.view_store import ViewStore

        sc = ServiceConfig()
        # render_version 必須帶：不帶就關掉了打包檔與 npz 目錄的一致性檢查，
        # 若 view_store 的 bin 是舊的，訓練會吃到舊圖，而專家測試集走
        # make_numpy_from_standard_views（讀 npz），兩邊會靜默地不一致。
        self.vs = {
            s: ViewStore(s, store_dir=sc.paths.view_store_dir,
                         npz_dir=sc.paths.views_dir(s),
                         render_version=sc.render.render_version)
            for s in ("FC", "EM")
        }
        # 每顆神經只有 8 種幾何變換（4 rot × 2 flip），全部快取，
        # 逐樣本增強就不會比整個 epoch 共用一組更貴。
        self.cache: dict[tuple[str, str, int, bool], np.ndarray] = {}

    def view(self, side: str, nid: str, rot: int = 0, flip: bool = False) -> np.ndarray:
        k = (side, nid, rot, flip)
        if k not in self.cache:
            base = self.cache.get((side, nid, 0, False))
            if base is None:
                base = self.vs[side].get(nid)
                self.cache[(side, nid, 0, False)] = base
            self.cache[k] = base if (rot == 0 and not flip) else _aug(base, rot, flip)
        return self.cache[k]

    def batch(self, pairs, rng: np.random.Generator | None = None,
              tfs: list[tuple[int, bool]] | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """取一批配對的三視圖。

        `tfs` 給定時用它指定每一對的 (rot, flip)（長度須等於 pairs）；
        否則 rng 給定時每一對各自抽一組；都沒有就不增強。兩側一律同步施加。

        ⚠️ **對比損失請用 `tfs` 讓同一個比較群組共用朝向。**
        逐對各抽會讓正例與它的 K 個負例落在不同朝向下卻被直接比大小；
        實測最難與次難負例的 logit 差距中位 0.265 < 同一對跨 8 種朝向的標準差
        0.449，導致 45.9 % 的時候「哪個負例最難」由朝向運氣決定。
        （這與 SimCLR 逐樣本增強不同：那裡編碼器被訓練成對增強不變，
        這裡 trunk 凍結，噪音是不可約的地板。）
        BCE 那批沒有跨樣本比較，逐對抽即可。

        ⚠️ 也不要退回「整個 epoch 共用一組」：那樣 60 個 epoch 只是從 8 種變換抽
        60 次，epoch 內零多樣性；而且 head 的 BN 在一個 epoch 內只看到單一朝向，
        batch 統計量會隨 epoch 大幅擺盪，推論卻固定用 rot=0/flip=False，
        訓練與推論的 BN 統計量對不上。
        """
        assert tfs is None or len(tfs) == len(pairs), "tfs 長度須等於 pairs"
        from swc_util import _pad_to_same_size, _resize_to_50

        n = len(pairs)
        q = np.empty((n, 50, 50, 3), dtype=np.float32)
        t = np.empty_like(q)
        for i, (fc, em) in enumerate(pairs):
            if tfs is not None:
                r, f = tfs[i]
            elif rng is None:
                r, f = 0, False
            else:
                r, f = int(rng.integers(4)), bool(rng.integers(2))
            a, b = _pad_to_same_size(self.view("FC", str(fc), r, f),
                                     self.view("EM", str(em), r, f))
            q[i] = np.transpose(_resize_to_50(a, (50, 50)), (1, 2, 0))
            t[i] = np.transpose(_resize_to_50(b, (50, 50)), (1, 2, 0))
        return q / 255.0, t / 255.0


# ---------------------------------------------------------------- 資料

def load_pool(cfg: Config) -> pd.DataFrame:
    """全池分數 + 外部型別判定（1.0 同型 / 0.0 確定不同 / NaN 無法判斷）。"""
    from debias import build_judge, label_pairs

    df = pd.read_parquet(PROJECT / cfg.pool_scores)
    df["fc_id"] = df.fc_id.astype(str)
    fc, judge = build_judge()
    return label_pairs(df, fc, judge)


def split_fc(df: pd.DataFrame, seed: int, train_ratio: float = 0.5
             ) -> tuple[list[str], list[str]]:
    """對 winnable FC（池中確有同型 EM）依族群分層切開。"""
    p = PROJECT / "analysis_external_validation" / "results" / "pair_labels.csv"
    fam = pd.read_csv(p, usecols=["fc_id", "family"]).drop_duplicates("fc_id")
    fam = dict(zip(fam.fc_id, fam.family))
    win = sorted(df[df.type_label == 1.0].fc_id.unique())
    rng = np.random.default_rng(seed)
    tr, ev = [], []
    for f in sorted({fam.get(x, "?") for x in win}):
        grp = np.array([x for x in win if fam.get(x, "?") == f])
        idx = rng.permutation(len(grp))
        h = int(round(len(grp) * train_ratio))
        tr += list(grp[idx[:h]])
        ev += list(grp[idx[h:]])
    return sorted(tr), sorted(ev)


def em_type_map() -> dict[int, str]:
    """全庫 em_id -> neuPrint 策展型別（`build_judge` 用的是同一份快取）。"""
    e = pd.read_csv(PROJECT / "analysis_external_validation" / "cache"
                    / "em_types_all_db.csv")
    return dict(zip(e.em_id.astype("int64"), e.np_type.fillna("")))


def holdout_em_ids(cfg: Config) -> set[int]:
    if not cfg.holdout_emtypes:
        return set()
    want = {t.strip() for t in cfg.holdout_emtypes.split(",") if t.strip()}
    return {i for i, t in em_type_map().items() if t in want}


def build_index(df: pd.DataFrame, train_fc: list[str], cfg: Config) -> dict:
    """每顆訓練 FC 的：同型 EM、型別不符 EM、以及可當 cross 負例的 EM。

    cross 負例 = 這顆 FC 池內、是**別顆**訓練 FC 的同型 EM、但對本 FC 型別不符。
    這正是「全域受歡迎但對這顆 FC 是錯的」集合，用來堵死 EM 先驗那條捷徑。
    """
    sub = df[df.fc_id.isin(train_fc)]
    ho = holdout_em_ids(cfg)
    if ho:
        n0 = len(sub)
        sub = sub[~sub.em_id.isin(ho)]
        print(f"[rank] EM 型別 hold-out {cfg.holdout_emtypes}："
              f"移除 {len(ho):,} 顆 EM、{n0 - len(sub):,} 對（訓練側）")
    pos = {fc: g[g.type_label == 1.0].em_id.to_numpy()
           for fc, g in sub.groupby("fc_id", observed=True)}
    neg = {fc: g[g.type_label == 0.0].em_id.to_numpy()
           for fc, g in sub.groupby("fc_id", observed=True)}
    globally_good = set()
    for v in pos.values():
        globally_good.update(v.tolist())

    idx = {}
    for fc in train_fc:
        p, n = pos.get(fc, np.array([])), neg.get(fc, np.array([]))
        if len(p) == 0 or len(n) < cfg.k_within:
            continue
        # ⚠️ 必須與 `neg`（type_label == 0.0）取交集，不能與整個池取交集。
        #    池裡還有 0.5（同亞族、無法斷定）與未定型的 EM，把它們當負例推下去是錯的——
        #    KC 的 0.5 正是「同屬 alpha/beta、只是 hemibrain 與 VFB 切法不同」，
        #    很可能就是對的答案。實測：用整個池時 KC 有 18.1 % 的 cross 候選是這種，
        #    每個錨點平均誤抽 1.4 個（佔 32 個負例的 4.5 %），而 KC 佔訓練 FC 的 66 %。
        cross = np.array(sorted((globally_good & set(n.tolist())) - set(p.tolist())),
                         dtype="int64")
        if cfg.neg_pool and len(n) > cfg.neg_pool:
            # 固定子集：用獨立、與 fold 綁定的 rng，跨 epoch 不變
            n = np.sort(np.random.default_rng(cfg.split_seed + 77).choice(
                n, cfg.neg_pool, replace=False))
        idx[fc] = {"pos": p, "neg": n, "cross": cross}
    return idx


# ---------------------------------------------------------------- 評估

def em_prior_rank1(full: pd.DataFrame, key: str, subset_fc: set | None = None,
                   seed: int = 11) -> float:
    """虛無模型：完全忽略 FC，只用每顆 EM 的平均分排序。

    這是本任務的 null model。一個真正在做形態比對的模型，它的分數不該能被這樣重現。
    先驗用「一半 FC 估、另一半評估」的樣本外作法。

    ⚠️ **先驗一定要在全體 FC 上估，不能只在族群子集內估。**
    在 KC 子集內估等於問「用『這顆 EM 對 KC 神經普遍評分高』能做多好」——
    而一個**正確**學會型別的模型本來就會讓 KCg-m 對所有 KC FC 評分高，
    那個 null 會把「學會型別」誤判成「學會先驗」。
    實測同一份探針分數：族群內估先驗 KC 得 49.1 %，全體估只有 0.5 %。
    要偵測的是「不看 FC 是哪一顆就能做多好」，所以估計集必須跨族群。
    """
    rng = np.random.default_rng(seed)
    fcs = full.fc_id.unique()
    perm = rng.permutation(fcs)
    h = len(perm) // 2
    est, val = set(perm[:h]), set(perm[h:])
    prior = full[full.fc_id.isin(est)].groupby("em_id", observed=True)[key].mean()
    v = full[full.fc_id.isin(val)]
    if subset_fc is not None:
        v = v[v.fc_id.isin(subset_fc)]
    v = v.copy()
    v["_p"] = v.em_id.map(prior)
    t = v.sort_values("_p", ascending=False).groupby("fc_id", observed=True).head(1)
    t = t[t.type_label.notna()]
    return 100 * float((t.type_label == 1.0).mean()) if len(t) else float("nan")


def em_prior_metrics(full: pd.DataFrame, key: str, subset_fc: set | None = None,
                     seed: int = 11) -> dict[str, float]:
    """同 `em_prior_rank1`，但一併回傳 MRR 與 p@5。

    ⚠️ 只看 rank-1 分辨不出深度。純先驗是「對每顆 FC 給同一份排名表」，
    它的 p@5 理應遠低於模型的 p@5；若兩者接近，模型的 p@5 也要打折。
    """
    rng = np.random.default_rng(seed)
    fcs = full.fc_id.unique()
    perm = rng.permutation(fcs)
    h = len(perm) // 2
    est, val = set(perm[:h]), set(perm[h:])
    prior = full[full.fc_id.isin(est)].groupby("em_id", observed=True)[key].mean()
    v = full[full.fc_id.isin(val)]
    if subset_fc is not None:
        v = v[v.fc_id.isin(subset_fc)]
    v = v.copy()
    v["_p"] = v.em_id.map(prior)
    s_ = v.sort_values("_p", ascending=False)
    g = s_.groupby("fc_id", observed=True)
    t = g.head(1)
    t = t[t.type_label.notna()]
    s_ = s_.copy()
    s_["_r"] = g.cumcount() + 1
    corr = s_[s_.type_label == 1.0].groupby("fc_id", observed=True)._r.min()
    nan = float("nan")
    return {
        "rank1": 100 * float((t.type_label == 1.0).mean()) if len(t) else nan,
        "MRR": float((1 / corr).mean()) if len(corr) else nan,
        "p@5": 100 * float((corr <= 5).mean()) if len(corr) else nan,
    }


def retrieval_report(ev: pd.DataFrame, keys: dict[str, str],
                     extra: dict[str, set] | None = None) -> pd.DataFrame:
    """分族群報 MRR / 正解 rank 中位 / rank-1 / EM 先驗。

    ⚠️ 主指標用 **MRR** 與「最佳正解的 rank 中位」，不用池內 AUC：
    每顆 FC 的正例數差兩個數量級（KC 中位 111、ALPN 只有 1），
    AUC 對全部正例取平均、又看不到未定型候選，跨族群不可比。
    實測 ALPN 基線 AUC 0.997 但 rank-1 只有 5.8 %，兩者甚至反向。
    rank-1 的執行間變異約 ±10 pp，也不宜單獨採信。
    """
    p = PROJECT / "analysis_external_validation" / "results" / "pair_labels.csv"
    fam = pd.read_csv(p, usecols=["fc_id", "family"]).drop_duplicates("fc_id")
    ev = ev.merge(fam, on="fc_id", how="left")

    rows = []
    for name, key in keys.items():
        scopes = ["全部"] + sorted(ev.family.dropna().unique()) + sorted(extra or {})
        for scope in scopes:
            if scope == "全部":
                d = ev
            elif extra and scope in extra:
                d = ev[ev.fc_id.isin(extra[scope])]
            else:
                d = ev[ev.family == scope]
            if not len(d):
                continue
            s = d.sort_values(key, ascending=False)
            g = s.groupby("fc_id", observed=True)
            t1 = g.head(1)
            t1 = t1[t1.type_label.notna()]
            s = s.copy()
            s["_r"] = g.cumcount() + 1
            corr = s[s.type_label == 1.0].groupby("fc_id", observed=True)._r.min()
            # cluster bootstrap by FC：eval 只有 383 顆時測量噪音不可忽略，
            # 而 rank-1 本身還有 ±10 pp 的執行間變異，沒有 CI 就無法判斷差異真假。
            hit = t1.set_index("fc_id").type_label.eq(1.0)
            rr = 1 / corr
            bk = sorted(set(hit.index) & set(rr.index))   # 不可叫 keys：會蓋掉參數
            brng = np.random.default_rng(17)
            bm, bh = [], []
            for _ in range(2000):
                pick = brng.choice(bk, len(bk), replace=True)
                bm.append(float(rr.loc[pick].mean()))
                bh.append(100 * float(hit.loc[pick].mean()))
            rows.append({
                "model": name, "family": scope, "n_fc": d.fc_id.nunique(),
                "MRR": round(float((1 / corr).mean()), 4),
                "MRR_lo": round(float(np.percentile(bm, 2.5)), 4),
                "MRR_hi": round(float(np.percentile(bm, 97.5)), 4),
                "median_rank": int(corr.median()),
                "rank1_pct": round(100 * float((t1.type_label == 1.0).mean()), 2),
                "rank1_lo": round(float(np.percentile(bh, 2.5)), 2),
                "rank1_hi": round(float(np.percentile(bh, 97.5)), 2),
                "p@5_pct": round(100 * float((corr <= 5).mean()), 2),
            })
            # 先驗一律在全體 ev 上估，只把「評估對象」限縮到該族群
            pm = em_prior_metrics(ev, key, None if scope == "全部" else set(d.fc_id))
            rows[-1].update({
                "em_prior_rank1_pct": round(pm["rank1"], 2),
                "em_prior_MRR": round(pm["MRR"], 4),
                "em_prior_p@5_pct": round(pm["p@5"], 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程

def main(cfg: Config) -> None:
    import tensorflow as tf
    from data_process_fineTune import make_numpy_from_standard_views, set_seed
    from model import MVCNN_Siamese, MVCNN_Siamese_3View

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)

    # --- 型別標籤 -----------------------------------------------------
    df = load_pool(cfg)
    train_fc, eval_fc = split_fc(df, cfg.split_seed, cfg.train_ratio)
    # 從 train 半邊再切 dev（挑 checkpoint 用），eval 半邊完全不碰
    n_dev = int(len(train_fc) * cfg.dev_ratio)
    perm = rng.permutation(len(train_fc))
    dev_fc = [train_fc[i] for i in perm[:n_dev]]
    train_fc = [train_fc[i] for i in perm[n_dev:]]
    idx = build_index(df, train_fc, cfg)
    dev_idx = build_index(df, dev_fc, cfg)
    print(f"[rank] 訓練 FC {len(idx)} / dev FC {len(dev_idx)} / 評估 FC {len(eval_fc)}")
    ncross = np.array([len(v["cross"]) for v in idx.values()])
    print(f"[rank] cross 負例可選數：中位 {int(np.median(ncross))}，"
          f"最少 {int(ncross.min())}")

    # --- 專家標註 -----------------------------------------------------
    sp = Path(cfg.split_dir)
    tr_df = pd.read_csv(sp / f"train_split_{cfg.fold}_{cfg.split_name}.csv")
    te_df = pd.read_csv(sp / f"test_split_{cfg.fold}_{cfg.split_name}.csv")
    x_exp, pair_exp, miss_exp, _ = make_numpy_from_standard_views(
        tr_df[["fc_id", "em_id", "label"]],
        fc_dir=cfg.fc_views_dir, em_dir=cfg.em_views_dir)
    y_exp = pair_exp["label"].to_numpy(dtype=np.float32)
    print(f"[rank] 專家訓練配對 {len(x_exp)}（缺三視圖 {len(miss_exp)}）")

    # --- 模型：凍結 trunk，新 head 輸出 logit --------------------------
    if cfg.trunk_init == "annotator":
        # 舊權重是用 MVCNN_Siamese（單視角 bug）訓練的，只能配舊函數
        net = MVCNN_Siamese(cfg.input_size)
        net.load_weights(cfg.base_weights)
    else:
        net = MVCNN_Siamese_3View(cfg.input_size, trunk_norm=cfg.trunk_norm)
        print(f"[rank] trunk 隨機初始化（MVCNN_Siamese_3View，trunk_norm={cfg.trunk_norm}），"
              f"完全不繼承 annotator")
    cat = next(l for l in net.layers if l.__class__.__name__ == "Concatenate")
    trunk = tf.keras.Model(net.input, cat.output)
    trunk.trainable = not cfg.freeze_trunk

    feat_dim = int(cat.output.shape[-1])
    inp = tf.keras.Input(shape=(feat_dim,))
    h = tf.keras.layers.Dropout(0.3)(inp)
    h = tf.keras.layers.Dense(256)(h)
    # ⚠️ 這裡刻意用 LayerNormalization 而非 BatchNormalization。
    #    用於檢索的打分函數，對同一對神經的輸出不該取決於「同批次裡還有誰」，
    #    而 BN 在訓練時正是批次相依的。這個落差咬過三次：
    #      1. momentum 0.99 讓 moving 統計嚴重滯後 -> 池內 AUC 掉到 0.33
    #      2. 正例（128 筆）/ 負例（4096 筆）/ 專家（128 筆）三次 forward 各自更新
    #         moving 統計，穩態權重 29.9 % / 33.2 % / 36.9 %；但檢索評估只打 pool、
    #         守門只打專家，兩個指標都在錯配的正規化下量測
    #      3. 更根本的：同一個 InfoNCE 比較裡，正例用正例批次的統計、
    #         負例用負例批次的統計，兩個經過不同正規化的量被直接比大小
    #    第 3 點事後無法補救——實測把 moving 統計重校到 pool 或 expert，
    #    兩個指標都比混合值更差（rank-1 34.6 -> 17.4 / 14.7），因為訓練時
    #    根本不存在單一正規化可供重現。LN 逐樣本正規化，訓練與推論完全相同。
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.keras.layers.Activation("gelu")(h)
    out = tf.keras.layers.Dense(1)(h)          # logit；sigmoid 只在推論與 BCE 時套
    head = tf.keras.Model(inp, out, name="rank_head")

    if cfg.head_init == "existing":
        # 原 head 是 Dense(256) -> BN -> gelu -> Dense(1, sigmoid)，
        # 權重形狀與這裡完全相同，只差最後的 sigmoid（這裡輸出 logit），直接搬。
        src = [l for l in net.layers
               if l.__class__.__name__ in ("Dense", "BatchNormalization")][-3:]
        dst = [l for l in head.layers
               if l.__class__.__name__ in ("Dense", "LayerNormalization")]
        assert len(src) == len(dst) == 3, (len(src), len(dst))
        for a, b in zip(src, dst):
            if b.__class__.__name__ == "LayerNormalization":
                # BN 有 (gamma, beta, moving_mean, moving_var)，LN 只有 (gamma, beta)。
                # 搬前兩個；moving 統計無對應物，本來就是要擺脫的東西。
                b.set_weights(a.get_weights()[:2])
            else:
                b.set_weights(a.get_weights())
        print("[rank] head 由 annotator 既有權重初始化")

    if cfg.load_head:
        head.load_weights(cfg.load_head)
        print(f"[rank] head 由 {Path(cfg.load_head).name} 載入（診斷模式）")

    print(f"[rank] trunk {trunk.count_params():,} 參數（init={cfg.trunk_init}，"
          f"凍結={cfg.freeze_trunk}）"
          f" / head {head.count_params():,} 參數（init={cfg.head_init}）")

    # 端到端的打分器：影像 -> trunk -> head -> logit。
    # 一律走這條路，`freeze_trunk` 只決定 trunk 的權重要不要更新。
    # 為什麼不預先快取特徵：每個 epoch 都重抽負例與增強，每一對在該 epoch 只用一次，
    # 所以「快取特徵再跑多個 minibatch」並不會減少 trunk 的 forward 次數，
    # 只是多花 4.7 GB 記憶體，而且會讓 trunk 無法接收梯度。
    scorer = tf.keras.Model(net.input, head(cat.output), name="rank_scorer")

    # ⚠️ trunk 的 6 個 BN 一律保持 trainable=False。
    #    凍結時本來就如此（Keras 的 BatchNormalization 在 trainable=False 會走推論模式，
    #    實測三次 training=True 的 forward 後 moving_mean 完全不變）——這是凍結路徑的
    #    隱性優點。但 --tune-trunk 會讓它們改用各自 batch 的統計量，而 sp_(128 對)、
    #    sn_(4096 對)、l_bce(128 對) 是三次獨立 forward、batch 大小差 32 倍且分布不同，
    #    margin 有一部分可以靠統計量差異被滿足，推論改用 moving average 時就消失。
    #    所以解凍時只訓練卷積權重，不動 BN。
    if not cfg.freeze_trunk:
        for l in trunk.layers:
            if l.__class__.__name__ == "BatchNormalization":
                l.trainable = False

    train_vars = (head.trainable_weights if cfg.freeze_trunk
                  else scorer.trainable_weights)

    if not cfg.freeze_trunk and cfg.batch_anchors > 16:
        # 解凍時 tape 必須保留卷積活化值，128 個錨點會 OOM（實測 40.4 GiB / 48 GB）
        cfg = replace(cfg, batch_anchors=16)
        print(f"[rank] --tune-trunk：batch_anchors 自動降到 {cfg.batch_anchors}（避免 OOM）")
    print(f"[rank] 可訓練參數 {sum(int(np.prod(v.shape)) for v in train_vars):,}")

    enc = PairEncoder(cfg)
    opt = tf.keras.optimizers.AdamW(learning_rate=cfg.lr)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    K_tot = cfg.k_within + cfg.k_cross
    # 控制組（w_rank=0，只有專家 BCE）不需要候選池那段前向計算。
    # 只在端到端路徑跳過；凍結路徑照舊（成本低，也不必另外處理）。
    skip_rank = (cfg.w_rank == 0 and not cfg.freeze_trunk)
    if skip_rank:
        print("[rank] w_rank=0：跳過排序項的前向計算（控制組，只訓練專家 BCE）")

    def _losses(qp, tp, qn, tn, qe, te_, ye, training=False):
        # 量測用途一律 training=False，否則這三次 forward 會順帶更新 head BN 的
        # moving 統計量（每 10 epoch 一次、每次 3 步，影響不大但沒必要）。
        if skip_rank:
            l_rank = tf.constant(0.0)
        else:
            B = tf.shape(qp)[0]
            sp_ = scorer({"FC": qp, "EM": tp}, training=training)           # (B,1)
            sn_ = tf.reshape(scorer({"FC": qn, "EM": tn}, training=training),
                             [B, K_tot])                                    # (B,K)
            logits = tf.concat([sp_, sn_], axis=1) / cfg.tau
            l_rank = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros(B, tf.int32), logits=logits))
        # 同 head_step：w_bce==0 時不可用 training=True，否則 BN moving 統計被污染
        l_bce = bce(ye, tf.squeeze(
            scorer({"FC": qe, "EM": te_},
                   training=training and bool(cfg.w_bce)), -1))
        return l_rank, l_bce

    def grad_norms(*batch) -> tuple[float, float]:
        """分別量兩項損失的梯度範數。呼叫端只傳 16 個錨點（見下方 ⚠️）。

        ⚠️ `w_rank` / `w_bce` 該怎麼設，看這個比值比看 loss 數值準。
        兩項損失都是**對 128 個樣本取平均**，所以「InfoNCE 有 78 500 對、
        BCE 只有 1 097 對」不會稀釋 BCE——每步的權重是相等的。
        真正的失衡來自 `tau`：InfoNCE 的 logits 被除以 0.07，梯度放大約 14 倍
        （梯度是 (softmax − onehot)/τ），而 BCE 對 logit 的梯度是
        (sigmoid(z) − y)，上界只有 1。所以 w_rank = w_bce = 1 實際上不是 1:1。
        """
        # ⚠️ 這裡沒有分塊，而且 persistent tape 會保留兩份活化值，
        #    所以呼叫端必須先把 batch 縮到 16 個錨點（512 對負例）；
        #    直接丟整批 4 096 對會吃掉 40 GB 並 OOM。
        with tf.GradientTape(persistent=True) as tape:
            l_rank, l_bce = _losses(*batch)
        # ⚠️ 不能寫成 tape.gradient(cfg.w_rank * l_rank, ...)：
        #    那個乘法發生在 tape 外面，沒被記錄，梯度會全部回傳 None。
        #    改成對原始損失求梯度，再把權重乘在範數上（線性，等價）。
        gr = tape.gradient(l_rank, train_vars)
        gb = tape.gradient(l_bce, train_vars)
        del tape

        def nrm(g):
            g = [x for x in g if x is not None]
            return float(tf.linalg.global_norm(g)) if g else 0.0
        return cfg.w_rank * nrm(gr), cfg.w_bce * nrm(gb)

    @tf.function
    def train_step(qp, tp, qn, tn, qe, te_, ye):
        with tf.GradientTape() as tape:
            if skip_rank:
                l_rank = tf.constant(0.0)
            else:
                B = tf.shape(qp)[0]
                sp_ = scorer({"FC": qp, "EM": tp}, training=True)           # (B,1)
                sn_ = tf.reshape(scorer({"FC": qn, "EM": tn}, training=True),
                                 [B, K_tot])                                # (B,K)
                logits = tf.concat([sp_, sn_], axis=1) / cfg.tau
                l_rank = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros(B, tf.int32), logits=logits))
            l_bce = bce(ye, tf.squeeze(
                scorer({"FC": qe, "EM": te_}, training=True), -1))
            loss = cfg.w_rank * l_rank + cfg.w_bce * l_bce
        g = tape.gradient(loss, train_vars)
        opt.apply_gradients(zip(g, train_vars))
        return l_rank, l_bce

    # ---- 凍結路徑的快速版 --------------------------------------------
    # ⚠️ 記憶體：負例批次是 128 x 32 = 4 096 對，跑完整卷積時 tape 必須保留活化值——
    #    4 096 對 x 2 側 x 3 視角 = 24 576 張 50x50 圖，光 conv1 就 7.9 GB，
    #    四層加起來約 24 GB，實測在 48 GB 的卡上 OOM（已配置 40.4 GiB）。
    #    但**凍結 trunk 時根本不需要那些活化值**（trunk 的權重不在 train_vars 裡），
    #    所以把 trunk 的 forward 移到 tape 外、分塊算，tape 內只留純 MLP 的 head。
    #    記憶體從 40 GB 降到 1 GB 級，速度也更快。
    @tf.function
    def _trunk(q, t):
        return trunk({"FC": q, "EM": t}, training=False)

    def feats(q, t, chunk: int = 512) -> tf.Tensor:
        return tf.concat([_trunk(q[i:i + chunk], t[i:i + chunk])
                          for i in range(0, len(q), chunk)], axis=0)

    @tf.function
    def head_step(fp, fn, fe, ye):
        with tf.GradientTape() as tape:
            B = tf.shape(fp)[0]
            sp_ = head(fp, training=True)
            sn_ = tf.reshape(head(tf.reshape(fn, [-1, feat_dim]), training=True),
                             [B, K_tot])
            logits = tf.concat([sp_, sn_], axis=1) / cfg.tau
            l_rank = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros(B, tf.int32), logits=logits))
            # w_bce==0 時這次 forward 走推論模式。head 還是 BatchNorm 時這是必要的：
            #    training=True 會照樣更新 BN moving 統計量，約 1/3 的狀態來自專家配對，
            #    實測探針組態 rank-1 因此從 53.46 % 掉到 32.59 %。
            #    換成 LayerNorm 後已無 moving 統計，這個旗標只剩 Dropout 的差別，保留無害。
            l_bce = bce(ye, tf.squeeze(
                head(fe, training=bool(cfg.w_bce)), -1))
            loss = (cfg.w_rank * l_rank + cfg.w_bce * l_bce
                    if cfg.w_bce else cfg.w_rank * l_rank)
        g = tape.gradient(loss, head.trainable_weights)
        opt.apply_gradients(zip(g, head.trainable_weights))
        return l_rank, l_bce

    exp_pairs = list(zip(pair_exp.fc_id.astype(str), pair_exp.em_id.astype(str)))

    # --- dev 的固定抽樣（不增強，整個訓練過程用同一批，才可比較）---------
    dev_rng = np.random.default_rng(cfg.seed + 1)
    dev_a, dev_n = [], []
    for fc in sorted(dev_idx):
        e = dev_idx[fc]
        for p in dev_rng.choice(e["pos"], min(2, len(e["pos"])), replace=False):
            w = dev_rng.choice(e["neg"], cfg.k_within, replace=False)
            c = (dev_rng.choice(e["cross"], cfg.k_cross,
                                replace=len(e["cross"]) < cfg.k_cross)
                 if len(e["cross"]) else dev_rng.choice(e["neg"], cfg.k_cross,
                                                        replace=False))
            dev_a.append((fc, int(p)))
            dev_n.append([(fc, int(x)) for x in np.concatenate([w, c])])
    if dev_a:
        dqp, dtp = enc.batch(dev_a)
        dqn, dtn = enc.batch([x for r in dev_n for x in r])
    print(f"[rank] dev 錨點 {len(dev_a)}"
          + ("（dev_ratio=0，訓練用滿全部 FC；必須配 --pick-epoch）" if not dev_a else ""))

    def dev_acc() -> tuple[float, float, float]:
        """dev 上的三元組正確率，回傳 (整體, within, cross)。

        比跑完整檢索便宜得多（約 2 秒 vs 12 分鐘），足以挑 checkpoint。

        ⚠️ **一定要看 cross 那一項。** dev 的 EM 有 100 % 在 train 出現過，
        所以整體正確率**抓不到 EM 先驗退化**——靠先驗取巧的模型在 dev 上照樣好看。
        但 cross 負例正好就是「別顆 FC 的正例」，也就是全域受歡迎的那批 EM；
        模型若是在學先驗，within 會漂亮而 **cross 會明顯落後**。
        兩者拉開就是退化的訊號，不必等到最後那張 retrieval 表。
        """
        if not dev_a:
            return float("nan"), float("nan"), float("nan")
        sp_ = scorer.predict({"FC": dqp, "EM": dtp}, verbose=0, batch_size=1024).reshape(-1)
        sn_ = scorer.predict({"FC": dqn, "EM": dtn}, verbose=0,
                             batch_size=1024).reshape(len(dev_a), K_tot)
        win = sn_[:, :cfg.k_within]       # 同 FC 池內的型別不符（24 個）
        cro = sn_[:, cfg.k_within:]        # 別顆 FC 的正例（8 個）
        # ⚠️ within / cross 必須用**每個負例的勝率**，不能用「贏過全部」：
        #    兩者負例數不同（24 vs 8），「贏過全部」的難度會被數量主導，不可比。
        return (float((sp_[:, None] > sn_).all(axis=1).mean()),   # 整體仍用嚴格定義
                float((sp_[:, None] > win).mean()),
                float((sp_[:, None] > cro).mean()))

    best = {"acc": -1.0, "epoch": -1, "w": None}
    hist = []
    fcs = sorted(idx)
    # 增強：**每個錨點群組抽一組 (rot, flip)**，正例與它的 K_tot 個負例共用；
    # 錨點之間仍各自不同。2026-09-15 修正——原本逐對各抽，導致正例與負例在不同
    # 朝向下被直接比大小，實測 45.9 % 的時候「哪個負例最難」由朝向運氣決定
    # （最難與次難差距中位 0.265 < 同一對跨 8 種朝向的標準差 0.449）。
    # 專家 BCE 那批維持逐對（無跨樣本比較）。見 PairEncoder.batch 的 docstring。
    aug = rng if cfg.augment else None
    fixed_pos = ({fc: rng.choice(idx[fc]["pos"],
                                 min(cfg.n_pos, len(idx[fc]["pos"])), replace=False)
                  for fc in fcs} if cfg.fixed_anchors else None)
    if fixed_pos:
        print(f"[rank] 固定錨點：{sum(len(v) for v in fixed_pos.values())} 個，"
              f"整個訓練過程不變（對齊探針）")

    for ep in range(cfg.epochs):
        # 每 epoch 重抽錨點與負例（--fixed-anchors 時正例固定）
        anchors, negs = [], []
        for fc in fcs:
            e = idx[fc]
            ps = (fixed_pos[fc] if fixed_pos is not None
                  else rng.choice(e["pos"], min(cfg.n_pos, len(e["pos"])), replace=False))
            for p in ps:
                w = rng.choice(e["neg"], cfg.k_within, replace=False)
                if len(e["cross"]):
                    c = rng.choice(e["cross"], cfg.k_cross,
                                   replace=len(e["cross"]) < cfg.k_cross)
                else:   # 極少數 FC 沒有 cross 候選，用池內負例補滿
                    c = rng.choice(e["neg"], cfg.k_cross, replace=False)
                anchors.append((fc, int(p)))
                negs.append([(fc, int(x)) for x in np.concatenate([w, c])])

        order = rng.permutation(len(anchors))
        lr_, lb_ = [], []
        for i in range(0, len(order), cfg.batch_anchors):
            sel = order[i:i + cfg.batch_anchors]
            # 一個錨點群組（正例 + 它的 K_tot 個負例）共用同一組 (rot, flip)
            grp = ([(int(rng.integers(4)), bool(rng.integers(2))) for _ in sel]
                   if aug is not None else None)
            qp, tp = enc.batch([anchors[j] for j in sel], tfs=grp)
            if skip_rank:
                qn, tn = qp[:0], tp[:0]
            else:
                flat = [p for j in sel for p in negs[j]]
                qn, tn = enc.batch(
                    flat, tfs=None if grp is None
                    else [g for g in grp for _ in range(K_tot)])
            je = rng.choice(len(exp_pairs), min(cfg.batch_anchors, len(exp_pairs)),
                            replace=False)
            qe, te_ = enc.batch([exp_pairs[j] for j in je], aug)
            if cfg.freeze_trunk:
                # trunk forward 在 tape 外、分塊算，tape 內只留 head（純 MLP）
                a, b = head_step(feats(qp, tp),
                                 tf.reshape(feats(qn, tn), [len(sel), K_tot, feat_dim]),
                                 feats(qe, te_), tf.constant(y_exp[je]))
            else:
                a, b = train_step(tf.constant(qp), tf.constant(tp),
                                  tf.constant(qn), tf.constant(tn),
                                  tf.constant(qe), tf.constant(te_),
                                  tf.constant(y_exp[je]))
            lr_.append(float(a)); lb_.append(float(b))
            # 診斷用的樣本要小：grad_norms 沒有分塊，整批 4 096 對會 OOM
            m = min(16, len(sel))
            last = (qp[:m], tp[:m],
                    qn[:m * K_tot], tn[:m * K_tot],
                    qe[:m], te_[:m], y_exp[je][:m])

        rec = {"epoch": ep + 1, "rank": float(np.mean(lr_)), "bce": float(np.mean(lb_))}
        if (ep + 1) % cfg.eval_every == 0 or ep == cfg.epochs - 1:
            acc, acc_w, acc_c = dev_acc()
            rec.update({"dev_acc": acc, "dev_acc_within": acc_w, "dev_acc_cross": acc_c})
            if not cfg.pick_epoch and acc > best["acc"]:
                # trunk 解凍時它的權重也在變，只存 head 會還原到不一致的組合
                best = {"acc": acc, "epoch": ep + 1,
                        "w": [w.numpy().copy() for w in head.weights],
                        "tw": (None if cfg.freeze_trunk
                               else [w.numpy().copy() for w in trunk.weights])}
        if cfg.pick_epoch == ep + 1:
            best = {"acc": rec.get("dev_acc", float("nan")), "epoch": ep + 1,
                    "w": [w.numpy().copy() for w in head.weights],
                    "tw": (None if cfg.freeze_trunk
                           else [w.numpy().copy() for w in trunk.weights])}
        if ep < 3 or (ep + 1) % 10 == 0:
            # 每 10 個 epoch 量一次兩項的梯度範數（多一次反向傳播，成本可忽略）
            gr, gb = grad_norms(*[tf.constant(x) for x in last])
            rec.update({"grad_rank": gr, "grad_bce": gb,
                        "grad_ratio": gr / gb if gb else float("inf")})
            dv = (f"  dev {rec['dev_acc']:.3f}"
                  f" (within {rec['dev_acc_within']:.3f} / cross {rec['dev_acc_cross']:.3f})"
                  if "dev_acc" in rec else "")
            print(f"  epoch {ep+1:3d}  rank {rec['rank']:.4f}  bce {rec['bce']:.4f}"
                  f"  |grad| rank/bce = {gr:.3g}/{gb:.3g} = {rec['grad_ratio']:.1f}x{dv}",
                  flush=True)
        hist.append(rec)

    # --- 還原 dev 上最好的 checkpoint ----------------------------------
    if best["w"] is not None:
        head.set_weights(best["w"])
        if best.get("tw") is not None:
            trunk.set_weights(best["tw"])
        how = "強制指定" if cfg.pick_epoch else "dev 最佳"
        print(f"[rank] 採用 epoch {best['epoch']} 的權重（{how}，dev 三元組正確率 {best['acc']:.3f}）")
        if not cfg.pick_epoch and best["epoch"] < cfg.epochs:
            print(f"       ⚠️ 不是最後一個 epoch，最後那個過擬合了")

    # --- BN 重校（診斷）-----------------------------------------------
    # head 的 BN moving 統計在訓練時被三種分布混合更新（正例 / 負例 / 專家配對，
    # 穩態權重 29.9 % / 33.2 % / 36.9 %），但檢索評估只打 pool 配對、
    # 守門只打專家配對——兩個指標各自都在錯配的正規化下量測。
    # 這裡把 moving 統計重新收斂到指定分布，再跑同一套評估。
    if cfg.recalib:
        rc = np.random.default_rng(cfg.seed + 5)
        n_need = cfg.recalib * 128
        if cfg.recalib_src == "pool":
            sub = df.sample(min(n_need, len(df)), random_state=cfg.seed + 5)
            rc_pairs = list(zip(sub.fc_id.astype(str), sub.em_id.astype(str)))
        else:   # 用**訓練**的專家配對，不能碰測試集
            j = rc.choice(len(exp_pairs), n_need, replace=True)
            rc_pairs = [exp_pairs[i] for i in j]
        for i in range(0, len(rc_pairs), 128):
            q, t = enc.batch(rc_pairs[i:i + 128])
            head(feats(q, t), training=True)    # 只為更新 BN moving 統計
        print(f"[rank] BN 重校完成：{len(rc_pairs)} 筆 {cfg.recalib_src} 配對、"
              f"{len(rc_pairs)//128} 批")

    # --- 保存 ---------------------------------------------------------
    frz = "frozen" if cfg.freeze_trunk else "tuned"
    stem = f"{cfg.model_name}_{cfg.head_init}_{frz}_{cfg.split_name}_{cfg.fold}"
    # 消融跑不能蓋掉正式權重：w_bce 非預設時寫進檔名
    if cfg.w_bce != Config.w_bce:
        stem += f"_wbce{cfg.w_bce:g}"
    if cfg.pick_epoch:
        stem += f"_ep{cfg.pick_epoch}"
    if cfg.trunk_init != "annotator":
        stem += f"_trunk{cfg.trunk_init}3v{cfg.trunk_norm}"
    if cfg.w_rank != Config.w_rank:
        stem += f"_wrank{cfg.w_rank:g}"
    if cfg.neg_pool:
        stem += f"_np{cfg.neg_pool}"
    if not cfg.augment:
        stem += "_noaug"
    if cfg.recalib:
        stem += f"_recal{cfg.recalib_src}{cfg.recalib}"
    if cfg.fixed_anchors:
        stem += "_fixanc"
    if cfg.holdout_emtypes:
        stem += "_ho" + cfg.holdout_emtypes.replace(",", "-").replace("'", "")
    # trunk 解凍時整個打分器都變了，只存 head 會接不回去
    head.save_weights(Path(cfg.save_dir) / f"{stem}_head.weights.h5")
    if not cfg.freeze_trunk:
        net.save_weights(Path(cfg.save_dir) / f"{stem}_trunk.weights.h5")
    with open(Path(cfg.result_dir) / f"Train_History_{stem}.pkl", "wb") as f:
        pickle.dump(hist, f)
    print(f"[rank] -> {cfg.save_dir}/{stem}_*.weights.h5")

    # --- 評估：型別層級檢索（含 EM 先驗虛無模型）-----------------------
    ev = df[df.fc_id.isin(eval_fc)].copy()
    print(f"[rank] 評估 {ev.fc_id.nunique()} 顆 FC、{len(ev):,} 對…", flush=True)
    ev_pairs = list(zip(ev.fc_id.astype(str), ev.em_id.astype(str)))

    def score_pairs(pairs, bs=None, model=None) -> np.ndarray:
        bs = bs or cfg.eval_bs
        m = model or scorer
        outs = []
        for i in range(0, len(pairs), bs):
            q, t = enc.batch(pairs[i:i + bs])   # 評估不增強
            outs.append(m.predict({"FC": q, "EM": t}, verbose=0, batch_size=bs).reshape(-1))
        return np.concatenate(outs)

    ev["ranktune"] = score_pairs(ev_pairs)

    # 對照模型（例如修正視角切片後重訓的 annotator）
    cmp = None
    cmp_path = cfg.compare_weights_tpl.format(fold=cfg.fold) if cfg.compare_weights_tpl else ""
    if cmp_path and Path(cmp_path).exists():
        cmp = MVCNN_Siamese_3View(cfg.input_size)
        cmp.load_weights(cmp_path)
        print(f"[rank] 對照模型 {Path(cmp_path).name}：為 {len(ev_pairs):,} 對打分…", flush=True)
        ev["compare"] = score_pairs(ev_pairs, model=cmp)
    elif cmp_path:
        print(f"[rank] ⚠️ 找不到對照模型 {cmp_path}，略過")

    # hold-out 層：正例**完全**落在被 hold out 型別裡的 eval FC。
    # 它們在訓練時正例歸零而自動退出，那些 EM body 也從沒被推高過，
    # 所以先驗在結構上無法轉移——這一層的增益只能是形態。
    extra = {}
    scope_types = cfg.report_emtypes or cfg.holdout_emtypes
    if scope_types:
        tmap = em_type_map()
        po = ev[ev.type_label == 1.0].copy()
        po["_t"] = po.em_id.map(tmap)
        byfc = po.groupby("fc_id", observed=True)._t.agg(set)
        for T in (t.strip() for t in scope_types.split(",") if t.strip()):
            fcs_ = {f for f, st in byfc.items() if st == {T}}
            if fcs_:
                extra[f"holdout:{T}"] = fcs_
        print(f"[rank] 分層（{'hold-out' if cfg.holdout_emtypes else '僅報表'}）："
              + "、".join(
            f"{k} {len(v)} 顆 FC" for k, v in extra.items()))

    keys = {"基線(原 head)": "score"}
    if cmp is not None:
        keys[Path(cmp_path).name.split("_D1")[0]] = "compare"
    keys["RankTune"] = "ranktune"
    rep = retrieval_report(ev, keys, extra)
    rep.to_csv(Path(cfg.result_dir) / f"retrieval_{stem}.csv", index=False)
    # 每一對的分數也要留下：per-EM 置中、hubness 診斷等事後分析都需要它，
    # 沒有的話每問一個問題就要重跑一次 12 分鐘的評估。
    ev[["fc_id", "em_id", "type_label", "score", "ranktune"]
       + (["compare"] if cmp is not None else [])].to_parquet(
        Path(cfg.result_dir) / f"evalscores_{stem}.parquet", index=False)
    print("\n=== 型別層級檢索 ===")
    print(rep.to_string(index=False))

    # --- 評估：專家測試集（二分類守門）--------------------------------
    from sklearn.metrics import roc_auc_score

    x_te, pair_te, _, _ = make_numpy_from_standard_views(
        te_df[["fc_id", "em_id", "label"]],
        fc_dir=cfg.fc_views_dir, em_dir=cfg.em_views_dir)
    te_pairs = list(zip(pair_te.fc_id.astype(str), pair_te.em_id.astype(str)))
    prob = 1 / (1 + np.exp(-score_pairs(te_pairs)))
    y = (pair_te.label.to_numpy() >= 0.5).astype(int)
    # ⚠️ 不能用 `net` 當基線：它與 `trunk` 共用圖層，--tune-trunk 時 trunk 已被訓練過，
    #    而 --trunk-init scratch 時它根本沒載入過 annotator 權重（實測誤報 AUC 0.4627，
    #    照字面讀會變成「RankTune 大勝原模型」，其實是輸）。每次都重建一個乾淨的參照。
    ref = MVCNN_Siamese(cfg.input_size)
    ref.load_weights(cfg.base_weights)
    base = ref.predict({"FC": x_te[:, 0], "EM": x_te[:, 1]}, verbose=0).reshape(-1)
    print(f"\n=== 專家測試集（fold {cfg.fold}，守門）===")
    print(f"  原模型 AUC {roc_auc_score(y, base):.4f}  acc@0.5 {((base>=.5)==y).mean():.4f}"
          f"   [{Path(cfg.base_weights).name}]")
    out_te = {"fc_id": pair_te.fc_id, "em_id": pair_te.em_id,
              "label": pair_te.label, "base": base}
    if cmp is not None:
        cp = cmp.predict({"FC": x_te[:, 0], "EM": x_te[:, 1]}, verbose=0).reshape(-1)
        print(f"  對照模型 AUC {roc_auc_score(y, cp):.4f}  acc@0.5 {((cp>=.5)==y).mean():.4f}"
              f"   [{Path(cmp_path).name}]")
        out_te["compare"] = cp
    print(f"  RankTune AUC {roc_auc_score(y, prob):.4f}  acc@0.5 {((prob>=.5)==y).mean():.4f}")
    out_te["ranktune"] = prob
    pd.DataFrame(out_te).to_csv(Path(cfg.result_dir) / f"test_label_{stem}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--head-init", choices=["scratch", "existing"], default="scratch")
    ap.add_argument("--tune-trunk", action="store_true",
                    help="連 annotator 的卷積層一起訓練（預設凍結）")
    ap.add_argument("--w-bce", type=float, default=Config.w_bce,
                    help="專家二元項的權重（預設 10.0，見 Config 的說明）")
    ap.add_argument("--tau", type=float, default=Config.tau)
    ap.add_argument("--pick-epoch", type=int, default=0,
                    help="強制評估這個 epoch 的權重而非 dev 最佳（0 = dev 最佳）")
    ap.add_argument("--trunk-init", choices=["annotator", "scratch"], default="annotator",
                    help="scratch = 完全從頭訓練，不繼承 annotator（會自動解凍 trunk）")
    ap.add_argument("--n-pos", type=int, default=Config.n_pos)
    ap.add_argument("--k-within", type=int, default=Config.k_within)
    ap.add_argument("--k-cross", type=int, default=Config.k_cross)
    ap.add_argument("--report-emtypes", default="",
                    help="只在報表多列這些型別的分層，不影響訓練（hold-out 的控制組）")
    ap.add_argument("--holdout-emtypes", default="",
                    help="逗號分隔的 neuPrint 型別，這些 EM 完全不進訓練")
    ap.add_argument("--fixed-anchors", action="store_true",
                    help="正例只抽一次、整個訓練固定（探針的作法）")
    ap.add_argument("--neg-pool", type=int, default=Config.neg_pool,
                    help="每顆 FC 的負例池上限（0 = 整個型別不符池）；探針用 64")
    ap.add_argument("--train-ratio", type=float, default=Config.train_ratio)
    ap.add_argument("--load-head", default="", help="載入既有 head 權重（診斷；配 --epochs 0）")
    ap.add_argument("--recalib", type=int, default=0,
                    help="用幾個 128 筆的批次重校 head BN 的 moving 統計（0 = 不做）")
    ap.add_argument("--recalib-src", choices=["pool", "expert"], default="pool")
    ap.add_argument("--w-rank", type=float, default=Config.w_rank,
                    help="排序項權重；0 = 控制組，只訓練專家 BCE")
    ap.add_argument("--trunk-norm", choices=["gn", "bn"], default=Config.trunk_norm,
                    help="從頭訓練時 trunk 的正規化（必須 gn，見 Config 說明）")
    ap.add_argument("--batch-anchors", type=int, default=Config.batch_anchors)
    ap.add_argument("--eval-bs", type=int, default=Config.eval_bs)
    ap.add_argument("--compare-weights", default="",
                    help="評估時一併打分的對照模型，可含 {fold}，例如 "
                         "./Annotator_Model/Annotator3v_D1-D6_{fold}.weights.h5")
    ap.add_argument("--dev-ratio", type=float, default=Config.dev_ratio,
                    help="從訓練 FC 切多少當 dev（0 = 不切，須配 --pick-epoch）")
    a = ap.parse_args()
    if a.trunk_init == "scratch":
        if a.head_init == "existing":
            ap.error("--trunk-init scratch 不能配 --head-init existing："
                     "head 的權重是針對 annotator 的特徵訓練的，接到隨機 trunk 上沒有意義")
        if a.trunk_norm != "gn":
            ap.error("--trunk-init scratch 必須配 --trunk-norm gn："
                     "--tune-trunk 會把 trunk 的 BN 鎖在推論模式，從頭訓練時它們停在初始統計量、"
                     "從未正規化；讓 BN 正常訓練又會在同一次比較裡混用不同批次的統計量")
        if not a.tune_trunk:
            a.tune_trunk = True
            print("[rank] --trunk-init scratch 自動啟用 --tune-trunk"
                  "（凍結的隨機 trunk 只是隨機投影）")
    main(Config(fold=a.fold, epochs=a.epochs, seed=a.seed,
                augment=not a.no_augment, head_init=a.head_init,
                freeze_trunk=not a.tune_trunk, w_bce=a.w_bce, tau=a.tau,
                pick_epoch=a.pick_epoch, trunk_init=a.trunk_init,
                n_pos=a.n_pos, k_within=a.k_within, k_cross=a.k_cross,
                neg_pool=a.neg_pool, train_ratio=a.train_ratio,
                dev_ratio=a.dev_ratio, load_head=a.load_head,
                recalib=a.recalib, recalib_src=a.recalib_src,
                fixed_anchors=a.fixed_anchors,
                holdout_emtypes=a.holdout_emtypes,
                report_emtypes=a.report_emtypes,
                w_rank=a.w_rank, trunk_norm=a.trunk_norm,
                batch_anchors=a.batch_anchors, eval_bs=a.eval_bs,
                compare_weights_tpl=a.compare_weights))
