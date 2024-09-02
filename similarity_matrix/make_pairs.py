# %% 從fc_id名單建構兩兩組合的pair csv
import pandas as pd
import os
import numpy as np

fc_ids = pd.read_csv('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/fc_fc_id.csv')
fc_lst = fc_ids['fc_id'].drop_duplicates().to_list()
em_lst = fc_lst.copy()

# 使用np.meshgrid生成两两组合
fc_grid, em_grid = np.meshgrid(fc_lst, em_lst)
pairs = np.vstack([fc_grid.ravel(), em_grid.ravel()]).T

# 将组合转换为DataFrame
pairs_df = pd.DataFrame(pairs, columns=['fc_id', 'em_id'])
#添加score、rank欄位
pairs_df['score'] = 0
pairs_df['rank'] = 0

# 保存为CSV文件
output_path = '/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/fc_fc_pairs.csv'
pairs_df.to_csv(output_path, index=False)

# %%
