'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
'''

'''
注意!
這個檔案split的規則是隨機shuffle 10次/ 或指定testing set名單
要使用10-fold validation規則, 改用另外的檔案
'''
# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import pandas as pd
import os
import random


# %% 此档案目的为 load 最佳参数模型, 然后对yifan那边画出的未标注三视图进行标注
# Mode 1: Shuffle 多次label csv, 統計學上做多次分割
# Mode 2: 指定的test data csv(D2+D5_nblast_test)中在 D2 的數據作為testing data, 剩下所有不重複資料做train data
# Mode 3: 指定的test data csv中在 D5 的數據作為testing data, 剩下所有不重複資料做train data
# 在 Mode 2 下, 輸出的分割數據集檔案編號採用特殊編號 98
# 在 Mode 3 下, 輸出的分割數據集檔案編號採用特殊編號 99

mode = 1
mode_csv_path = './data/D2+D5_nblast_test.csv'  #指定測試集檔案位置

use_new_label = False        # 使用日期為01-13的label table, 這項標注中將人類信心50%的標注視為Negative

num_splits = 10


seed = 10                       # Random Seed

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'


train_range_to = 'D5'   # 'D4' or 'D5'


# Load labeled csv

label_csv_D1 = './data/D1_20230113.csv'
label_csv_D2 = './data/D2_20230113.csv'
label_csv_D3 = './data/D3_20230113.csv'
label_csv_D4_1 = './data/D4-1_20230113.csv'
label_csv_D4_2 = './data/D4-2_20230113.csv'
label_csv_D5 = './data/D5_20230113.csv'


D1 = pd.read_csv(label_csv_D1)     # FC, EM, label
D1.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D2 = pd.read_csv(label_csv_D2)     # FC, EM, label
D2.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D3 = pd.read_csv(label_csv_D3)     # FC, EM, label
D3.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D4_1 = pd.read_csv(label_csv_D4_1)     # FC, EM, label
D4_1.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D4_2 = pd.read_csv(label_csv_D4_2)
D4_2.drop_duplicates(subset=['fc_id','em_id'], inplace=True)

if train_range_to == 'D5':
    D5 = pd.read_csv(label_csv_D5)     # FC, EM, label
    D5.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

    label_table_all = pd.concat([D1, D2, D3, D4_1, D4_2, D5])   # fc_id, em_id, score, rank, label

else:
    label_table_all = pd.concat([D1, D2, D3, D4_1, D4_2])   # fc_id, em_id, score, rank, label



label_table_all.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复


if mode == 1:
    for i in range(num_splits):

        seed = i

        # shuffle label_table_all 
        label_table_all = label_table_all.sample(frac=1, random_state=seed).reset_index(drop=True)

        # 将 label_table_all 分成两份: train and test
        test_ratio = 0.1
        test_size = int(label_table_all.shape[0]*test_ratio)

        label_table_test = label_table_all.iloc[:test_size]
        label_table_train = label_table_all.iloc[test_size:]

        # save as csv
        label_table_test.to_csv('./data/test_split_' + str(i) +'_D1-' + train_range_to + '.csv', index=False)
        label_table_train.to_csv('./data/train_split_' + str(i) +'_D1-' + train_range_to + '.csv', index=False)

elif mode == 2:
    test_table = pd.read_csv(mode_csv_path)
    test_table.drop_duplicates(subset=['fc_id', 'em_id'], inplace=True) # 删除重复

    # 使用label_table_all的列名来筛选label_table中的列
    selected_columns = [col for col in test_table.columns if col in label_table_all.columns]
    test_table = test_table[selected_columns]

    # 創建一個輔助列'Merge', 表示在 test_table 中 D2 部分(作為 testing data)
    test_table_merge = test_table.merge(D2, on=['fc_id', 'em_id'], how='left', indicator=True)

    # 保留同時在D2中的pair，並將要保留的column name 還原(對D2+D5nblast的標注有疑慮，目前以20221230為主)
    test_table_D2 = test_table_merge[test_table_merge['_merge']=='both'].rename(columns={'score_y': 'score', 'label_y': 'label'})
    test_table_D2 = test_table_D2.drop(columns=['score_x', 'label_x', '_merge'])

    # 創建一個輔助列'Merge', 表示是僅label_table_all 原有的還是同時在test_tabl中也出現
    label_table_cleaned = label_table_all.merge(test_table_D2, on=['fc_id', 'em_id'], how='left', indicator=True)

    # 从csv2中删除交集'Both'的行, 刪除新增的輔助列label
    label_table_cleaned = label_table_cleaned[label_table_cleaned['_merge'] == 'left_only'].drop(columns='_merge')

    # 重命名列
    label_table_cleaned = label_table_cleaned.rename(columns={'score_x': 'score', 'label_x': 'label'})

    # 刪除不需要的列
    label_table_cleaned = label_table_cleaned.drop(columns=['score_y', 'label_y'])

    # save as csv No.98
    label_table_cleaned.to_csv('./data/train_split_98_D1-'+train_range_to+'.csv', index=False)
    test_table_D2.to_csv('./data/test_split_98_D1-'+train_range_to+'.csv', index=False)

elif mode == 3:
    test_table = pd.read_csv('./data/D5_230503.csv')
    test_table.drop_duplicates(subset=['fc_id', 'em_id'], inplace=True) # 删除重复

    # 使用label_table_all的列名来筛选label_table中的列
    selected_columns = [col for col in test_table.columns if col in label_table_all.columns]
    test_table = test_table[selected_columns]

    # 創建一個輔助列'Merge', 表示在 test_table 中 D5 部分(作為 testing data)
    test_table_merge = test_table.merge(D5, on=['fc_id', 'em_id'], how='left', indicator=True)
    
    # 保留同時在D5中的pair，並將要保留的column name 還原
    test_table_D5 = test_table_merge[test_table_merge['_merge']=='both'].rename(columns={'score_y': 'score', 'label_y': 'label'})
    test_table_D5 = test_table_D5.drop(columns=['score_x', 'label_x', '_merge'])

    # 創建一個輔助列'Merge', 表示是僅label_table_all 原有的還是同時在test_tabl中也出現
    label_table_cleaned = label_table_all.merge(test_table_D5, on=['fc_id', 'em_id'], how='left', indicator=True)
    
    # 从csv2中删除交集'Both'的行, 刪除新增的輔助列label
    label_table_cleaned = label_table_cleaned[label_table_cleaned['_merge'] == 'left_only'].drop(columns='_merge')

    # 重命名列
    label_table_cleaned = label_table_cleaned.rename(columns={'score_x': 'score', 'label_x': 'label'})

    # 刪除不需要的列
    label_table_cleaned = label_table_cleaned.drop(columns=['score_y', 'label_y'])

    # save as csv No.99
    label_table_cleaned.to_csv('./data/train_split_99_D1-'+train_range_to+'.csv', index=False)
    test_table_D5.to_csv('./data/test_split_99_D1-'+train_range_to+'.csv', index=False)



# %% 測試程序

# # read csv file
# test = pd.read_csv('./data/test_split_1_D1-D5.csv')
# train = pd.read_csv('./data/train_split_1_D1-D5.csv')
# %%
