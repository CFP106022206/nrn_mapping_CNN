"""重新訓練 annotator，修正視角切片（三張視圖都讀到）。

背景（2026-09-18）：`model.MVCNN_Siamese` 的視角切片有 Python 閉包延遲綁定 bug，
三個「視角」全部讀到第 3 張圖。現有 `Annotator_D1-D6_{fold}` 就是在這個狀態下訓練的
（三視角 BN 的 moving 統計量逐位元相同）。舊權重已備份在
`Annotator_Model_backup_singleview_20260918/`，且原檔保留不動，服務照舊可用。

本腳本沿用 `Data_process_Train.main` 的完整流程，只換掉這些設定：
不載入預訓練、lr 1e-5、300 epochs（原 annotator 的設定，見 Data_process_Train.Config 的註解）、
改用 `model.MVCNN_Siamese_3View`，並存成**新檔名** `Annotator_Model/Annotator3v_D1-D6_{fold}.weights.h5`，
不覆蓋舊權重。trunk 維持原本的 BatchNorm（trunk_norm 預設 "bn"），所以與舊 annotator 的唯一差別就是視角切片。

執行：python3 train_annotator_3view.py <fold>
"""

import sys
from dataclasses import dataclass

import Data_process_Train as T


@dataclass(frozen=True)
class Annotator3vConfig(T.Config):
    use_pretrain_model = False
    pretrain_model: str = ""
    save_model_dir: str = "./Annotator_Model"
    model_name = "Annotator3v"
    initial_lr: float = 1e-5
    train_epochs: int = 300
    model_builder: str = "MVCNN_Siamese_3View"


if __name__ == "__main__":
    T.main(Annotator3vConfig(fold=int(sys.argv[1])))
