'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
5, Iterative self-labeling
6, Transfer Big Model...
'''

'''
這個檔案使用10-fold validation規則
'''
# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import KFold

# %% 此档案目的为 load 最佳参数模型, 然后对yifan那边画出的未标注三视图进行标注
# Mode 1: 用所有標注data做cross validation
# Mode 2: 指定test data csv(用於nBLAST)做cross validation, 剩下所有不重複資料做train data

mode = 2
mode2_file_path = './data/nblast_D2+D5+D6_50as1.csv'
label_threshold = 0.5   # 50%信心 or 60%信心

cross_validation_num = 3


seed = 37                       # Random Seed

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'


train_range_to = 'D6'   # 'D5' or 'D6'


# Load labeled csv
if label_threshold == 0.5:
    label_csv_D1 = './data/D1_20221230.csv'
    label_csv_D2 = './data/D2_20230710.csv'
    label_csv_D3 = './data/D3_20221230.csv'
    label_csv_D4 = './data/D4_20230710.csv'
    label_csv_D5 = './data/D5_20221230.csv'
    label_csv_D6 = './data/D6_20230523.csv'
elif label_threshold == 0.6:
    label_csv_D1 = './data/D1_20230113.csv'
    label_csv_D2 = './data/D2_60as1.csv'
    label_csv_D3 = './data/D3_20230113.csv'
    label_csv_D4 = './data/D4_60as1.csv'
    label_csv_D5 = './data/D5_60as1.csv'
    label_csv_D6 = './data/D6_60as1.csv'

D1 = pd.read_csv(label_csv_D1)     # FC, EM, label
D1.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D2 = pd.read_csv(label_csv_D2)     # FC, EM, label
D2.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D3 = pd.read_csv(label_csv_D3)     # FC, EM, label
D3.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D4 = pd.read_csv(label_csv_D4)     # FC, EM, label
D4.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D5 = pd.read_csv(label_csv_D5)     # FC, EM, label
D5.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

if train_range_to == 'D6':
    D6 = pd.read_csv(label_csv_D6)     # FC, EM, label
    D6.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

    label_table_all = pd.concat([D1, D2, D3, D4, D5, D6])   # fc_id, em_id, score, rank, label

else:
    label_table_all = pd.concat([D1, D2, D3, D4, D5])   # fc_id, em_id, score, rank, label



label_table_all.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复


# 设置 KFold 参数
kf = KFold(n_splits=cross_validation_num, shuffle=True, random_state=seed)

# 分割数据集并执行交叉验证
if mode == 1:
    i = 0

    for train_index, test_index in kf.split(label_table_all):
        label_table_train = label_table_all.iloc[train_index]
        label_table_test = label_table_all.iloc[test_index]

        # save as csv
        label_table_test.to_csv('./data/test_split_' + str(i) +'_D1-' + train_range_to + '.csv', index=False)
        label_table_train.to_csv('./data/train_split_' + str(i) +'_D1-' + train_range_to + '.csv', index=False)
        i += 1

elif mode == 2:
    test_table = pd.read_csv(mode2_file_path)

    # 使用label_table_all的列名来筛选label_table中的列, 保留和label_table_all 一樣的列
    selected_columns = [col for col in test_table.columns if col in label_table_all.columns]
    test_table = test_table[selected_columns]
    
    # test_table label有誤,用 lebal_table_all 修正
    test_table = test_table.merge(label_table_all, on=['fc_id', 'em_id'], how='inner')
    test_table = test_table.rename(columns={'score_y':'score', 'label_y':'label'}).drop(columns=['score_x','label_x'])

    test_table.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

    #2023/8/12 添加, 分離出D2和D5的test data, 保證在做KFold時均勻
    test_table_D2 = test_table.merge(pd.concat([D2,D6], ignore_index=True), on=['fc_id', 'em_id'], how='inner')
    test_table_D2 = test_table_D2.rename(columns={'score_x':'score', 'label_x':'label'}).drop(columns=['score_y','label_y'])

    test_table_D5 = test_table.merge(D5, on=['fc_id', 'em_id'], how='inner')
    test_table_D5 = test_table_D5.rename(columns={'score_x':'score', 'label_x':'label'}).drop(columns=['score_y','label_y'])


    def kfold_split(test_table):

        test_table_lst = []

        # 使用 KFold 分出test data, train data會是label_table_all 剔除 test data
        for train_index, test_index in kf.split(test_table):
            test_table_lst.append(test_table.iloc[test_index])

        return test_table_lst

    # 分開處理D2和D5 將他們分別分成三份
    D2_test_lst = kfold_split(test_table_D2)
    D5_test_lst = kfold_split(test_table_D5)

    # 將分別分好三份的D2和D5的test data合併
    for i in range(len(D2_test_lst)):
        label_table_test = pd.concat([D2_test_lst[i], D5_test_lst[i]], ignore_index=True)

        # 創建一個輔助列'Merge', 表示是僅 label_table_all 原有的還是同時在test_tabl中也出現
        label_table_merged = label_table_all.merge(label_table_test, on=['fc_id', 'em_id'], how='left', indicator=True)
        # 从csv2中删除交集'Both'的行, 刪除新增的輔助列label
        label_table_cleaned = label_table_merged[label_table_merged['_merge'] == 'left_only'].drop(columns='_merge')

        # 重命名列
        label_table_cleaned = label_table_cleaned.rename(columns={'score_x': 'score', 'label_x': 'label'})

        # 刪除不需要的列
        label_table_cleaned = label_table_cleaned.drop(columns=['score_y', 'label_y'])

        # save as csv
        label_table_test.to_csv('./data/test_split_' + str(i) +'_D1-' + train_range_to + '.csv', index=False)
        label_table_cleaned.to_csv('./data/train_split_' + str(i) +'_D1-' + train_range_to + '.csv', index=False)


# %%
