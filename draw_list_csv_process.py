# %%
import numpy as np
import pandas as pd

draw = pd.read_csv('data/selected_data/EMxFC_all_0_rk20.csv')
# 修改列名稱
draw.columns = ['em_id', 'fc_id', 'score', 'rank']
# %%
# 保留第 rank>10 and rank<21 的資料 
selected_row = draw[(draw['rank'] < 21) & (draw['rank'] > 10)]
# selected_row = draw[draw['rank'] < 21]

selected_row.to_csv('data/selected_data/test_0to20.csv', index=False)
# %% 測試畫圖
import pickle

# 打開 pkl
with open('data/statistical_results/abc/mapping_data_sn_104198-F-000000.pkl', 'rb') as f:
    fc = pickle.load(f)

test = fc[0]

import matplotlib.pyplot as plt

plt.imshow(test[3].transpose(1, 2, 0))
plt.show()

plt.imshow(test[4].transpose(1, 2, 0))
plt.show()
# %%
