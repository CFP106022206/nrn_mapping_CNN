'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
5, Iterative self-labeling
6, Transfer Big Model...
7 model_predict
'''
# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import pickle
import cv2
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import *
from keras.layers import *
from keras.losses import BinaryFocalCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from util import load_pkl


cross_fold_num = 5
data_range = 'D6'

annotator_model_name = 'Annotator_D1-' + data_range + '_'
model_name = 'model_D1-'+data_range+'_'

model_folder = './Annotator_Model/'
predict_map_folder = './predict_mapping_data/'
predict_label_csv = './data/Gad1-F-100228.csv'
# %%

predict_lst = pd.read_csv(predict_label_csv)
pred_pair_nrn = predict_lst[['target_ID', 'candidate_ID', 'score']].to_numpy()

def data_preprocess(file_path, pair_nrn):

    print('\nCollecting 3-View Data Numpy Array..')
    # 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
    file_list = [file_name for file_name in os.listdir(file_path) if file_name.endswith('.pkl')]

    #使用字典存储有三視圖数据, 以 FC_EM 作为键, 使用字典来查找相应的数据, 减少查找时间
    data_dict = {}
    for file_name in file_list:
        pkl_path = os.path.join(file_path, file_name)
        data_lst = load_pkl(pkl_path)
        for data in data_lst:
            key = f"{data[0]}_{data[1]}"
            data_dict[key] = data

    resolutions = data[3].shape
    print('\n Resolutions:', resolutions)

    data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
    fc_nrn_lst, em_nrn_lst, score_lst = [], [], []

    # 依訓練名單從已有三視圖名單中查找是否存在
    for i, row in enumerate(pair_nrn):
        
        key = f"{row[0]}_{row[1]}"

        if key in data_dict:
            data = data_dict[key]   # 找出data的所有信息
            # 三視圖填入 data_np
            for k in range(3):
                data_np[i, 0, :, :, k] = data[3][k] # FC Image
                data_np[i, 1, :, :, k] = data[4][k] # EM Image
            # 其餘信息填入list
            fc_nrn_lst.append(data[0])
            em_nrn_lst.append(data[1])
            score_lst.append(row[2])
    


    # map data 中有可能找不到pair_nrn裡面的組合, 刪除那些找不到的0矩陣
    not_found_data = []
    for i, data in enumerate(data_np):
        if not(np.any(data)):
            not_found_data.append(i)
    data_np = np.delete(data_np, not_found_data, axis=0)

    not_found_df = []
    if not_found_data:
        print('How many pairs Not Found in map_data: ')
        for i in not_found_data:
            not_found_df.append(pair_nrn[i])
        print(len(not_found_df))
        not_found_df = pd.DataFrame(not_found_df, columns=['fc_id', 'em_id', 'label'])



    # Normalization : x' = x - min(x) / max(x) - min(x)
    data_np = (data_np - np.min(data_np))/(np.max(data_np) - np.min(data_np))

    pair_df = pd.DataFrame({'fc_id':fc_nrn_lst, 'em_id':em_nrn_lst, 'score':score_lst})    # list of pairs

    return data_np, pair_df, not_found_df

pred_data_np, predict_pair_df, not_found_df = data_preprocess(predict_map_folder, pred_pair_nrn)

pred_fc = pred_data_np[:, 0, :]
pred_em = pred_data_np[:, 1, :]

pred_result = []
for i in range(cross_fold_num):
    model_path = model_folder + annotator_model_name + str(i) + '.h5'
    model = load_model(model_path)
    y_pred = model.predict({'FC':pred_fc, 'EM':pred_em}, verbose=2)
    pred_result.append(y_pred)

# 平均 pred_result
pred_result_mean = np.mean(pred_result, axis=0)

#建構完整表格
predict_pair_df['ML_predict_mean'] = pred_result_mean
# 詳細每一個模型參數
# for i in range(cross_fold_num):
#     predict_pair_df['ML_predict_'+str(i)] = pred_result[i]

# 按照 ML_predict_mean 排序
predict_pair_df_sort = predict_pair_df.sort_values(by=['ML_predict_mean'], ascending=False)
predict_pair_df_sort.to_csv('./result/predict_pair_df.csv', index=False)

# %% 畫圖
def imshow_pred_pair(predict_pair_df, pred_data_np):
    for p in range(len(predict_pair_df)):
        fc_img = pred_data_np[p,0,:]
        em_img = pred_data_np[p,1,:]

        fc_id = predict_pair_df.iloc[p]['fc_id']
        em_id = predict_pair_df.iloc[p]['em_id']
        # score = predict_pair_df.iloc[p]['ML_predict_mean']

        plt.figure(figsize=(9,6))
        for i in range(3):
            plt.subplot(2,3,i+1)
            plt.imshow(fc_img[:,:,i], cmap='magma')
            plt.xticks([])
            plt.yticks([])      # 隱藏刻度線
            plt.subplot(2,3,i+4)
            plt.imshow(em_img[:,:,i], cmap='magma')
            plt.xticks([])
            plt.yticks([])      # 隱藏刻度線

        plt.suptitle(f'{fc_id}_{em_id}')
        plt.savefig(f'./Figure/predict_3view/{fc_id}_{em_id}.png', dpi=150, bbox_inches='tight')

imshow_pred_pair(predict_pair_df, pred_data_np)

# %%
