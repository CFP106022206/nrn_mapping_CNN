# %%
import pandas as pd
import os
import numpy as np

# %% human label data
label_csv_D1 = './labeled_info/D1_conf.csv'
label_csv_D2 = './labeled_info/D2_conf.csv'
label_csv_D3 = './labeled_info/D3_conf.csv'
label_csv_D4 = './labeled_info/D4_conf.csv'
label_csv_D5 = './labeled_info/D5_conf.csv'
label_csv_D6 = './labeled_info/D6_conf.csv'

D1 = pd.read_csv(label_csv_D1)     # FC, EM, label
D2 = pd.read_csv(label_csv_D2)     # FC, EM, label
D3 = pd.read_csv(label_csv_D3)     # FC, EM, label
D4 = pd.read_csv(label_csv_D4)     # FC, EM, label
D5 = pd.read_csv(label_csv_D5)     # FC, EM, label
D6 = pd.read_csv(label_csv_D6)     # FC, EM, label

label_table_all = pd.concat([D1, D2, D3, D4, D5, D6])   # fc_id, em_id, score, rank, label
label_table_all.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

# nblast data
nblast_path = 'labeled_info/nblast_all_list_D2_D5.csv'
nblast_score = pd.read_csv(nblast_path)
nblast_score.rename(columns={'fc': 'fc_id', 'em': 'em_id'}, inplace=True)
# add label info
nblast_merge = pd.merge(nblast_score, label_table_all, on=['fc_id', 'em_id'], how='left')
# delete score column
nblast_merge.drop(columns=['score'], inplace=True)
# save
nblast_merge.to_csv('labeled_info/nblast_all_list_D2_D5_label.csv', index=False)
# %% nblast include inverse
nblast_path = 'labeled_info/nblast_all_list_D2_D5_include_inverse.csv'
nblast_score = pd.read_csv(nblast_path)
nblast_score.rename(columns={'fc': 'find_by', 'em': 'target'}, inplace=True)

# 按間隔順序重新對齊EM、FC
fc, fc_inv, em, em_inv, score, score_inv = [], [], [], [], [], []
for i in range(1, len(nblast_score), 2):
    fc.append(nblast_score.iloc[i-1]['find_by'])
    em.append(nblast_score.iloc[i-1]['target'])
    score.append(nblast_score.iloc[i-1]['similarity score'])
    em_inv.append(nblast_score.iloc[i]['find_by'])
    fc_inv.append(nblast_score.iloc[i]['target'])
    score_inv.append(nblast_score.iloc[i]['similarity score'])

nblast_df = pd.DataFrame({'fc_id': fc, 'em_id': em, 'similarity score': score})
nblast_df_inv = pd.DataFrame({'fc_id': fc_inv, 'em_id': em_inv, 'inverse score': score_inv})

nblast_df = pd.merge(nblast_df, nblast_df_inv, on=['fc_id', 'em_id'], how='left')


# add label info
label_table_all['em_id'] = label_table_all['em_id'].astype(str) # EM label to string
nblast_merge = pd.merge(nblast_df, label_table_all, on=['fc_id', 'em_id'], how='left')
nblast_merge.drop(columns=['score'], inplace=True)
nblast_merge['asymmetry'] = np.abs(nblast_merge['similarity score'] - nblast_df['inverse score'])
nblast_merge.to_csv('labeled_info/nblast_all_list_D2_D5_include_inverse_label.csv', index=False)
# %%
