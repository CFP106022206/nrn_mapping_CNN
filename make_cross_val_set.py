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
# Mode 0: 不做cross validation
# Mode 1: 用所有標注data做cross validation
# Mode 2: 指定test data csv(用於nBLAST)做cross validation, 剩下所有不重複資料做train data
# Mode 3: 选同一条fc有对应到比较多em的pair作为testing data, 这样做的目的是为了评估时在评估几率从高到低排序时前n名中是否有正确答案

mode = 2
mode2_file_path = './labeled_info/nblast_D2+D5+D6_50as1.csv'
cross_validation_num = 3


used_label = 'soft_label'   # thres0.5(confidence>0.5 label as 1), thres0.6(confidence>0.6 label as 1), soft_label(keep confidence)


seed = 3407                       # Random Seed

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'


# Load labeled csv
if used_label == 'thres0.5':
    label_csv_D1 = './labeled_info/D1_20221230.csv'
    label_csv_D2 = './labeled_info/D2_20230710.csv'
    label_csv_D3 = './labeled_info/D3_20221230.csv'
    label_csv_D4 = './labeled_info/D4_20230710.csv'
    label_csv_D5 = './labeled_info/D5_20221230.csv'
    label_csv_D6 = './labeled_info/D6_20230523.csv'

elif used_label == 'thres0.6':
    label_csv_D1 = './labeled_info/D1_20230113.csv'
    label_csv_D2 = './labeled_info/D2_60as1.csv'
    label_csv_D3 = './labeled_info/D3_20230113.csv'
    label_csv_D4 = './labeled_info/D4_60as1.csv'
    label_csv_D5 = './labeled_info/D5_60as1.csv'
    label_csv_D6 = './labeled_info/D6_60as1.csv'

elif used_label == 'soft_label':
    label_csv_D1 = './labeled_info/D1_conf.csv'
    label_csv_D2 = './labeled_info/D2_conf.csv'
    label_csv_D3 = './labeled_info/D3_conf.csv'
    label_csv_D4 = './labeled_info/D4_conf.csv'
    label_csv_D5 = './labeled_info/D5_conf.csv'
    label_csv_D6 = './labeled_info/D6_conf.csv'


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

D6 = pd.read_csv(label_csv_D6)     # FC, EM, label
D6.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复


label_table_all = pd.concat([D1, D2, D3, D4, D5, D6])   # fc_id, em_id, score, rank, label
label_table_all.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复


# 设置 KFold 参数
kf = KFold(n_splits=cross_validation_num, shuffle=True, random_state=seed)

# %% 分割数据集并执行交叉验证
if mode == 0:

    # shuffle label_table_all 
    label_table_all = label_table_all.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 将 label_table_all 分成两份: train and test
    test_ratio = 0.1
    test_size = int(label_table_all.shape[0]*test_ratio)

    label_table_test = label_table_all.iloc[:test_size]
    label_table_train = label_table_all.iloc[test_size:]

    # save as csv
    label_table_test.to_csv('./train_test_split/test_split_0_D1-D6.csv', index=False)
    label_table_train.to_csv('./train_test_split/train_split_0_D1-D6.csv', index=False)


elif mode == 1:
    i = 0

    for train_index, test_index in kf.split(label_table_all):
        label_table_train = label_table_all.iloc[train_index]
        label_table_test = label_table_all.iloc[test_index]

        # save as csv
        label_table_test.to_csv('./train_test_split/test_split_' + str(i) +'_D1-D6.csv', index=False)
        label_table_train.to_csv('./train_test_split/train_split_' + str(i) +'_D1-D6.csv', index=False)
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
        label_table_test.to_csv('./train_test_split/test_split_' + str(i) +'_D1-D6.csv', index=False)
        label_table_cleaned.to_csv('./train_test_split/train_split_' + str(i) +'_D1-D6.csv', index=False)


elif mode == 3:
    # 针对label_table_all中的pair，統計相同的fc_id有多少個pair
    fc_id_dict = {}

    for index, row in label_table_all.iterrows():
        if row['fc_id'] in fc_id_dict:
            fc_id_dict[row['fc_id']] += 1   # 如果fc_id已经在字典中，就在对应的value(pair數量)上加1
        else:
            fc_id_dict[row['fc_id']] = 1


    # 依value 大小排序
    # fc_id_dict = dict(sorted(fc_id_dict.items(), key=lambda item: item[1], reverse=True))

    # 篩選出約 100 條 test data
    print("\nTest set's fc_id / Number of pairs")
    num = 0
    test_table_lst= []
    for key in fc_id_dict:
        print(key, fc_id_dict[key])   # 将fc_id_dict中每个key对应的value长度print出
        test_table = label_table_all[label_table_all['fc_id'] == key]
        # 檢查test_tabel中是否有label=1
        if 1 in test_table['label'].tolist():
            # 找到 有1 所在的那一行，這一步是為了保證test_tabel中至少有一個positive
            test_table_pos = test_table[test_table['label'] == 1]
            # 保留第一個
            test_table_pos = test_table_pos.iloc[0:1]
            #shuffle test_table
            test_table_shuffle = test_table.sample(frac=1, random_state=seed)
            # 將一半隨機放進test_tabel
            test_table = test_table_shuffle[:len(test_table)//1]
            # 加回test_tabel_pos，保證至少有一個positive在test中
            test_table = pd.concat([test_table, test_table_pos], ignore_index=True)

            test_table.drop_duplicates(subset=['fc_id','em_id'], inplace=True)

            test_table_lst.append(test_table)
            num += len(test_table)
        
        if num > 100:
            break
    
    # 將test_table_lst合併
    label_table_test = pd.concat(test_table_lst, ignore_index=True)
    # 分离出 train data
    label_table_train = label_table_all.merge(label_table_test, on=['fc_id', 'em_id'], how='left', indicator=True)
    label_table_train = label_table_train[label_table_train['_merge'] == 'left_only'].drop(columns=['score_y', 'label_y', '_merge'])
    label_table_train = label_table_train.rename(columns={'score_x': 'score', 'label_x': 'label'})

    # save as csv
    label_table_test.to_csv('./train_test_split/test_split_0_D1-D6.csv', index=False)
    label_table_train.to_csv('./train_test_split/train_split_0_D1-D6.csv', index=False)
# %%
