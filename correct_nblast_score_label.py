'''
本程序在使用正確的nblast分數和label文件修正並規範化nblast test的csv檔案。
'''

# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from util import load_pkl
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# %%
uncorrect_file_path = './data/nblast_D2+D5_correct.csv'    # 只提供fc_id, em_id
correct_nblast_score_file_path = './data/D2p_nblast_score.csv'  # 提供正確的score
correct_label_file_path = './data/D2_20221230.csv'  # 提供正確的label

uncorrect = pd.read_csv(uncorrect_file_path)[['fc_id', 'em_id', 'score', 'label']]

score_correct = pd.read_csv(correct_nblast_score_file_path)
score_correct = score_correct.rename(columns={'fc':'fc_id', 'em':'em_id', 'similarity_score':'score'})

label_correct = pd.read_csv(correct_label_file_path)[['fc_id', 'em_id', 'label']]

# merge score
uncorrect_merge_score = uncorrect.merge(score_correct, on=['fc_id', 'em_id'], how='left', indicator=True)

score_lst = []
for i in range(len(uncorrect_merge_score)):
    row = uncorrect_merge_score.iloc[i]

    if row['_merge'] == 'left_only':
        score_lst.append(row['score_x'])
    elif row['_merge'] == 'both':
        score_lst.append(row['score_y'])

uncorrect['score'] = score_lst


#merge label
uncorrect_merge_label = uncorrect.merge(label_correct, on=['fc_id', 'em_id'], how='left', indicator=True)

label_lst = []
for i in range(len(uncorrect_merge_label)):
    row = uncorrect_merge_label.iloc[i]

    if row['_merge'] == 'left_only':
        label_lst.append(row['label_x'])
    elif row['_merge'] == 'both':
        label_lst.append(row['label_y'])

uncorrect['label'] = np.int32(label_lst)

# save correct file
uncorrect.to_csv('./data/nblast_D2+D5_correct.csv', index=False)