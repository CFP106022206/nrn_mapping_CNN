# %%
import numpy as np
import pandas as pd
import os
from pathlib import Path

label_csv_folder = Path('result/unlabel_data_predict')
model_name = 'Annotator_D1-D6_'
model_cross_validation_num = 10

csv_lst = os.listdir(label_csv_folder)
# %%
total_df_lst = []
for i in range(model_cross_validation_num):
    label_csv_name = model_name+str(i)
    #讀取csv_lst中以label_csv_name開頭的全部csv檔案並且合併
    csv_lst_i = [f for f in csv_lst if f.startswith(label_csv_name) and f.endswith('.csv')]
    df_lst = []
    for csv_name in csv_lst_i:
        df = pd.read_csv(label_csv_folder/csv_name)
        df_lst.append(df)
    if not df_lst:
        continue
    df_i = pd.concat(df_lst, ignore_index=True)

    # 只保留這三列
    df_i = df_i[["fc_id", "em_id", "model_predict"]].copy()
    # 若同一 fold 里出现重复 pair，先去重避免 merge 后行数膨胀
    df_i = df_i.drop_duplicates(["fc_id", "em_id"], keep="first")
    # 横向合并时每个 fold 需要独立列名
    df_i = df_i.rename(columns={"model_predict": f"model_predict_{i}"})
    total_df_lst.append(df_i)
# 根據fc_id和em_id合併所有df_i
total_df = None
for df_i in total_df_lst:
    if total_df is None:
        total_df = df_i
    else:
        total_df = total_df.merge(df_i, on=["fc_id", "em_id"], how="outer")

# 計算 mean predict score, std（跨 10 個 fold；缺失值會自動略過）
pred_cols = [f"model_predict_{i}" for i in range(model_cross_validation_num)]
pred_cols = [c for c in pred_cols if c in total_df.columns]
if not pred_cols:
    raise RuntimeError("No model_predict_* columns found to compute mean/std")

# 確保是數值欄位
total_df[pred_cols] = total_df[pred_cols].apply(pd.to_numeric, errors="coerce")

total_df["predict_mean"] = total_df[pred_cols].mean(axis=1, skipna=True)
# 用 ddof=0 與 numpy.std 預設一致
total_df["predict_std"] = total_df[pred_cols].std(axis=1, skipna=True, ddof=0)

# 保留predict_std < 0.05
total_hc = total_df[total_df["predict_std"] < 0.05].copy()
total_pos = total_hc[total_hc["predict_mean"] >= 0.5].copy()
total_neg = total_hc[total_hc["predict_mean"] < 0.5].copy()
# 平衡正負樣本數量（负样本数量一定远大于正样本）
total_neg = total_neg.sample(n=len(total_pos), random_state=42)  # 隨機抽樣負樣本，使其數量與正樣本相同
total_final = pd.concat([total_pos, total_neg], ignore_index=True)
total_final = total_final[["fc_id", "em_id", "predict_mean"]].copy()
total_final.rename(columns={"predict_mean": "label"}, inplace=True)
# %%
total_final.to_csv("data/pairs_label/EMxFC_all_high_confidence.csv", index=False)
# %%
