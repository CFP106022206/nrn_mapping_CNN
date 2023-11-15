# %%
import numpy as np
import pandas as pd

draw = pd.read_csv('data/selected_data/FCxEM_all_0_rk100.csv')
# 修改列名稱
draw.columns = ['fc_id', 'em_id', 'score', 'rank']
# %%
selected_row = draw[draw['fc_id'] == '104198-F-000000']
# 確認排序正確
selected_row = selected_row.sort_values(by=['rank'])
# 取前n
selected_row = selected_row.iloc[:10]
# %%
selected_row.to_csv('data/selected_data/test.csv', index=False)
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
