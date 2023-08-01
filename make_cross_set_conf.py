'''
make_cross_val_set.py 使用信心度而不是label的版本。

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
import pickle
from sklearn.model_selection import KFold

# %% 此档案目的为 load 最佳参数模型, 然后对yifan那边画出的未标注三视图进行标注
# Mode 1: 用所有標注data做cross validation
# Mode 2: 指定test data csv(用於nBLAST)做cross validation, 剩下所有不重複資料做train data

mode = 2
mode2_file_path = './data/nblast_D2+D5+D6_60as1.csv'

cross_validation_num = 5


seed = 10                       # Random Seed

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'


# Load labeled csv

label_csv_D1 = './data/D1_conf.csv'
label_csv_D2 = './data/D2_conf.csv'
label_csv_D3 = './data/D3_conf.csv'
label_csv_D4 = './data/D4_conf.csv'
label_csv_D5 = './data/D5_conf.csv'
label_csv_D6 = './data/D6_conf.csv'

D1 = pd.read_csv(label_csv_D1)     # fc_id, em_id, confidence
D1.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D2 = pd.read_csv(label_csv_D2)     # fc_id, em_id, confidence
D2.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D3 = pd.read_csv(label_csv_D3)     # fc_id, em_id, confidence
D3.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D4 = pd.read_csv(label_csv_D4)     # fc_id, em_id, confidence
D4.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D5 = pd.read_csv(label_csv_D5)     # fc_id, em_id, confidence
D5.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D6 = pd.read_csv(label_csv_D6)     # fc_id, em_id, confidence
D6.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

# label_table_all = pd.concat([D1, D2, D3, D4, D5, D6])   # fc_id, em_id, confidence
label_table_all = pd.concat([D2, D3, D4, D5, D6])   # fc_id, em_id, confidence
label_table_all.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

# 分析已標注的所有數據
confidence_lvl_lst = []
for k in range(11): # 遍歷所有信心度等級
    confidence_lvl = k/10
    n = 0
    for label in label_table_all['confidence']:
        if label == confidence_lvl:
            n += 1
    confidence_lvl_lst.append(n)

print('all confidence levels number:')
print(confidence_lvl_lst)

# 合併信心等級以減小等級之間的數據量差異
#[0%, [10% ~ 40%], [50% ~ 70%], [80% ~ 100%]]
integration_method = [[0], [1,2,3,4], [5,6,7], [8,9,10]]

# 計算合併後的信心度（加權平均）
conf_lvl_merge, class_num_merge = [], []
for i_lst in integration_method:
    conf_lvl = 0    # 合併後該類別加權計算的信心度
    class_num = 0   # 統計合併後該類別的數量
    for i in i_lst:
        class_num += confidence_lvl_lst[i]
        conf_lvl += confidence_lvl_lst[i] * (i/10)    #加權
    class_num_merge.append(class_num)
    conf_lvl_merge.append(np.round(conf_lvl/class_num, 1))

print('Confidence Level after merge:')
print(conf_lvl_merge)

print('Number of each class')
print(class_num_merge)

# 將label_table_all 的 confidence用合併後的信心度替代
conf_new = []
for conf in label_table_all['confidence']:
    # 找到對應信心度在融合規則中在哪個位置
    for i, sublist in enumerate(integration_method):
        if int(conf*10) in sublist: # i索引對應的位置為新信心度位置
            conf_new.append(conf_lvl_merge[i])

label_table_all['confidence'] = conf_new

# 设置 KFold 参数
kf = KFold(n_splits=cross_validation_num, shuffle=True, random_state=seed)


# 分割数据集并执行交叉验证
i = 0
if mode == 1:
    for train_index, test_index in kf.split(label_table_all):
        label_table_train = label_table_all.iloc[train_index]
        label_table_test = label_table_all.iloc[test_index]

        # save as csv
        label_table_test.to_csv('./data/test_split_' + str(i) +'_D1-D6.csv', index=False)
        label_table_train.to_csv('./data/train_split_' + str(i) +'_D1-D6.csv', index=False)
        i += 1

elif mode == 2:
    test_table = pd.read_csv(mode2_file_path)

    # 針對需要拿來作為test data的資料, 僅保留其中神經對編號
    selected_columns = ['fc_id', 'em_id']
    test_table = test_table[selected_columns]
    
    # test_table 無正確confidence資料,用 lebal_table_all 補上
    test_table = test_table.merge(label_table_all, on=['fc_id', 'em_id'], how='inner')

    # 使用 KFold 分出test data, train data會是label_table_all 剔除 test data
    for train_index, test_index in kf.split(test_table):
        label_table_test = test_table.iloc[test_index]
    
        # 創建一個輔助列'Merge', 表示是僅 label_table_all 原有的還是同時在test_tabl中也出現
        label_table_merged = label_table_all.merge(label_table_test, on=['fc_id', 'em_id'], how='left', indicator=True)
        # 从csv2中删除交集'Both'的行, 刪除新增的輔助列label
        label_table_cleaned = label_table_merged[label_table_merged['_merge'] == 'left_only'].drop(columns=['confidence_y','_merge'])

        # 重命名列
        label_table_cleaned = label_table_cleaned.rename(columns={'confidence_x': 'confidence'})

        # save as csv
        label_table_test.to_csv('./data/test_split_' + str(i) +'_D1-D6.csv', index=False)
        label_table_cleaned.to_csv('./data/train_split_' + str(i) +'_D1-D6.csv', index=False)
        i += 1

# %%
