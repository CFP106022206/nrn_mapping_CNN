'''
分析模型各层输出

'''

# %%
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from keras.models import load_model, Model
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from umap import UMAP
from util import load_pkl


# %%
num_splits = 2 #0~9, or 99 for whole nBLAST testing set
use_map_from = 'yf' #'kt': map_data 冠廷, 'yf': map_folder from 懿凡
map_dict_folder = './data/labeled_sn'

scheduler_exp = 0#1.5      #學習率調度器的約束力指數，越小約束越強
initial_lr = 0.00001
train_epochs = 300

add_low_score = False
low_score_neg_rate = 2

seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)


save_model_name  = 'Annotator_D1-D6_' +str(num_splits)


model = load_model('./Annotator_Model/' + save_model_name + '.h5')

# %% Load data  

# load train, test
label_table_train = pd.read_csv('./train_test_split/train_split_' + str(num_splits) +'_D1-D6.csv')
label_table_test = pd.read_csv('./train_test_split/test_split_' + str(num_splits) +'_D1-D6.csv')


# turn to numpy array
test_pair_nrn = label_table_test[['fc_id','em_id','label']].to_numpy()
train_pair_nrn = label_table_train[['fc_id','em_id','label']].to_numpy()


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
    fc_nrn_lst, em_nrn_lst, score_lst, label_lst = [], [], [], []

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
            score_lst.append(data[2])
            label_lst.append(row[2])
    


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

    pair_df = pd.DataFrame({'fc_id':fc_nrn_lst, 'em_id':em_nrn_lst, 'label':label_lst, 'score':score_lst})    # list of pairs

    return data_np, pair_df, not_found_df


data_np_test, nrn_pair_test, test_not_found = data_preprocess(map_dict_folder, test_pair_nrn)
data_np_train, nrn_pair_train, train_not_found = data_preprocess(map_dict_folder, train_pair_nrn)

data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.1, random_state=seed)

print('\nTrain data:', len(data_np_train),'\nValid data:', len(data_np_valid),'\nTest data:', len(data_np_test))


X_val = data_np_valid
X_test = data_np_test
y_val = np.array(nrn_pair_valid['label'])
y_test = np.array(nrn_pair_test['label'])

X_train = data_np_train
y_train = np.array(nrn_pair_train['label'])
y_train_bin = np.array([1 if y > 0.5 else 0 for y in y_train])

# FC/EM Split
X_train_FC = X_train[:,0,:]
X_train_EM = X_train[:,1,:]

del X_train

X_val_FC = X_val[:,0,:]
X_val_EM = X_val[:,1,:]

X_test_FC = X_test[:,0,:]
X_test_EM = X_test[:,1,:]


# %%
map1 = Model(inputs=model.input, outputs=model.get_layer('ac1').output)
map2 = Model(inputs=model.input, outputs=model.get_layer('ac2').output)
map3 = Model(inputs=model.input, outputs=model.get_layer('ac3').output)
map4 = Model(inputs=model.input, outputs=model.get_layer('ac4').output)

train_map1 = map1.predict({'FC': X_train_FC, 'EM': X_train_EM})
train_map2 = map2.predict({'FC': X_train_FC, 'EM': X_train_EM})
train_map3 = map3.predict({'FC': X_train_FC, 'EM': X_train_EM})
train_map4 = map4.predict({'FC': X_train_FC, 'EM': X_train_EM})


# Average pooling 50*50 to 5*5
# average_pooling_layer = AveragePooling2D(pool_size=(5, 5), strides=(5, 5))

# train_map1 = average_pooling_layer(train_map1).numpy()
# train_map2 = average_pooling_layer(train_map2).numpy()

# average_pooling_layer = AveragePooling2D(pool_size=(5, 5), strides=(5, 5))

# train_map3 = average_pooling_layer(train_map3).numpy()
# train_map4 = average_pooling_layer(train_map4).numpy()

# Umap 降维
umap = UMAP(n_components=50, random_state=seed)
# TSNE 降维
tsne = TSNE(n_components=2, random_state=seed)

train_map1 = umap.fit_transform(train_map1.reshape(train_map1.shape[0], -1))
train_map1 = tsne.fit_transform(train_map1)
plt.scatter(train_map1[:,0], train_map1[:,1], s=5, linewidth=0, c=y_train_bin, cmap='coolwarm')
plt.colorbar()
plt.show()

train_map2 = umap.fit_transform(train_map2.reshape(train_map2.shape[0], -1))
train_map2 = tsne.fit_transform(train_map2)
plt.scatter(train_map2[:,0], train_map2[:,1], s=5, linewidth=0, c=y_train_bin, cmap='coolwarm')
plt.colorbar()
plt.show()

train_map3 = umap.fit_transform(train_map3.reshape(train_map3.shape[0], -1))
train_map3 = tsne.fit_transform(train_map3)
plt.scatter(train_map3[:,0], train_map3[:,1], s=5, linewidth=0,  c=y_train_bin, cmap='coolwarm')
plt.colorbar()
plt.show()

train_map4 = umap.fit_transform(train_map4.reshape(train_map4.shape[0], -1))
train_map4 = tsne.fit_transform(train_map4)
plt.scatter(train_map4[:,0], train_map4[:,1], s=5, linewidth=0,  c=y_train_bin, cmap='coolwarm')
plt.colorbar()
plt.show()


# val_map1 = map1.predict({'FC': X_val_FC, 'EM': X_val_EM})
# val_map2 = map2.predict({'FC': X_val_FC, 'EM': X_val_EM})
# val_map3 = map3.predict({'FC': X_val_FC, 'EM': X_val_EM})
# val_map4 = map4.predict({'FC': X_val_FC, 'EM': X_val_EM})

# test_map1 = map1.predict({'FC': X_test_FC, 'EM': X_test_EM})
# test_map2 = map2.predict({'FC': X_test_FC, 'EM': X_test_EM})
# test_map3 = map3.predict({'FC': X_test_FC, 'EM': X_test_EM})
# test_map4 = map4.predict({'FC': X_test_FC, 'EM': X_test_EM})


# %%
