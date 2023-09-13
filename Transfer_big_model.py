'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
5, Iterative self-labeling
6, Transfer Big Model...
'''
# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import tensorflow as tf
import pickle
import cv2
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import *
from keras.layers import *
from keras.losses import BinaryFocalCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score
from util import load_pkl
from tqdm import tqdm
from numba import jit, prange


def data_preprocess_annotator(unlabel_path, pair_nrn):

    print('\nCollecting 3-View Data Numpy Array..')

    data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
    FC_nrn_lst, EM_nrn_lst, score_lst, label_lst = [], [], [], []


    file_list = os.listdir(unlabel_path)


    # 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
    file_list = [file_name for file_name in os.listdir(unlabel_path) if file_name.endswith('.pkl')]

    #使用字典存储有三視圖数据, 以 FC_EM 作为键, 使用字典来查找相应的数据, 减少查找时间
    data_dict = {}
    for file_name in file_list:
        file_path = os.path.join(unlabel_path, file_name)
        data_lst = load_pkl(file_path)
        for data in data_lst:
            key = f"{data[0]}_{data[1]}"
            data_dict[key] = data

    # 依訓練名單從已有三視圖名單中查找是否存在
    for i, row in tqdm(enumerate(pair_nrn), total=len(pair_nrn)):
        
        key = f"{row[0]}_{row[1]}"

        if key in data_dict:
            data = data_dict[key]   # 找出data的所有信息
            # 三視圖填入 data_np
            for k in range(3):
                data_np[i, 0, :, :, k] = data[3][k] # FC Image
                data_np[i, 1, :, :, k] = data[4][k] # EM Image
            # 其餘信息填入list
            FC_nrn_lst.append(data[0])
            EM_nrn_lst.append(data[1])
            score_lst.append(data[2])
            label_lst.append(row[2])



    # map data 中有可能找不到pair_nrn裡面的組合, 刪除那些找不到的0矩陣
    not_found_data = []
    for i, data in enumerate(data_np):
        if not(np.any(data)):
            not_found_data.append(i)
    data_np = np.delete(data_np, not_found_data, axis=0)

    if not_found_data:
        print('Not Found in map_data: ')
        for i in not_found_data:
            print(pair_nrn[i])


    # Normalization : x' = x - min(x) / max(x) - min(x)
    data_np = (data_np - np.min(data_np))/(np.max(data_np) - np.min(data_np))

    pair_df = pd.DataFrame({'fc_id':FC_nrn_lst, 'em_id':EM_nrn_lst, 'label':label_lst, 'score':score_lst})    # list of pairs

    return data_np, pair_df


resolutions = [3,50,50]

num_splits = 2 #0~9
data_range = 'D6'   #D4 or D5

fix_kernel = True  # 微調時是否凍結參數
use_scheduler = False #是否使用scheduler 控制學習率
scheduler_exp = 1      #學習率調度器的約束力指數，越小約束越強
# initial_lr = 0.00005
train_epochs = 50

add_low_score = False
low_score_neg_rate = 2

use_map_from = 'yf' #'kt': map_data 冠廷, 'yf': map_folder from 懿凡
map_dict_folder = './data/labeled_sn'


seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)

save_model_name  = 'model_D1-'+data_range+'_'+str(num_splits)

def scheduler(epoch, lr): 

    min_lr=0.0000001
    total_epoch = train_epochs
    epoch_lr = lr*((1-epoch/total_epoch)**scheduler_exp)
    if epoch_lr<min_lr:
        epoch_lr = min_lr

    return epoch_lr

# load annotator data
label_annotator = pd.read_csv('./result/label_df_with_Annotator_D1-'+ data_range +'_' + str(num_splits)+'.csv')
# TODO: 比較10個不同分割下的annotator表現差異性

# load train, test
label_table_train = pd.read_csv('./data/train_split_' + str(num_splits) +'_D1-' + data_range + '.csv')
label_table_test = pd.read_csv('./data/test_split_' + str(num_splits) +'_D1-' + data_range + '.csv')


# %% 刪除label_annotator 中與人工標註重複的部分

# 删除label_table_train 在annotator中出现的数据
merged = pd.merge(label_annotator, label_table_train, how='left', indicator=True)
label_annotator = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)

# 删除label_table_test 在annotator中出现的数据
merged = pd.merge(label_annotator, label_table_test, how='left', indicator=True)
label_annotator = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)



#针对 label_annotator做 data_preprocess
train_pair_nrn = label_annotator[['fc_id','em_id','label']].to_numpy()




unlabel_path = './data/mapping_data_0.7+'



# %%


data_np_train, nrn_pair_train = data_preprocess_annotator(unlabel_path, train_pair_nrn)

# Train Validation Split
data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.1, random_state=seed)

print('\nTrain data:', len(data_np_train),'\nValid data:', len(data_np_valid))


X_val = data_np_valid
y_val = np.array(nrn_pair_valid['label'],dtype=np.int32)


# Data Augmentation: cross expand(Truned off)

#   以下說明增加資料方法，大寫字母表示 FC id, 小寫表示 EM id
#   已知 A -> a
#       B -> a
#       B -> b  
#   則可以推論出    [不行，因為無法保證三視圖在同個角度拍攝，只能回到冠廷程序那邊重做圖]
#       A -> b
#   
#   實現：利用Dictionary，建立 A:[a], B:[a,b]
#   遍歷每個key中的元素，尋找其他key中此元素是否存在，若有，則將其他key中的所有元素添加進次key中
#   A:[a](Found in B) -> A:[a]+[a,b](in B) -> A:list(set([a]+[a,b])) 去重

#   相似的，
#   已知 A != b
#       B == c
#   可以推出
#       A != b,c
#   
#   實現方式類似，建立 label=0 的 Dictionary，遍歷每個 keys 中的值，尋找在擴增的label=1 Dict 中
#   與此元素同組的其他元素，將其他元素添加進 label=0 的 Dictionary 中
cross_expand = False

if cross_expand:
    true_label_pair_df = nrn_pair_train.loc[nrn_pair_train['label']==1]
    false_label_pair_df = nrn_pair_train.loc[nrn_pair_train['label']==0]

    def cluster_expansion(pair_df_pos, pair_df_neg):

        label_dict = {}
        for _, row in pair_df_pos.iterrows():
            if row['fc_id'] in label_dict:
                label_dict[row['fc_id']].append(row['em_id'])
            else:
                label_dict[row['fc_id']] = [row['em_id']]

        # 以FC為key的dictionary建立完成，接下來搜索可擴張項
        label_dict_new = label_dict.copy()
        for k in label_dict.keys():
            for v in label_dict[k]:
                for i in label_dict.keys():
                    if v in label_dict[i] and i != k:
                        label_dict_new[k] = list(set(label_dict[k]+label_dict[i]))

        pair_df_pos_expand = pd.DataFrame({'fc_id':[], 'em_id':[], 'label':[]})
        for k in label_dict_new.keys():
            for v in label_dict_new[k]:
                pair_df_pos_expand = pair_df_pos_expand.append({'fc_id':k, 'em_id':v, 'label':1}, ignore_index=True)


        # label=1資料擴張完畢，現在製造label=0的pair使資料量平衡  
        neg_dict = {}
        for _, row in pair_df_neg.iterrows():
            if row['fc_id'] in neg_dict:
                neg_dict[row['fc_id']].append(row['em_id'])
            else:
                neg_dict[row['fc_id']] = [row['em_id']]


        #擴張
        neg_dict_new = neg_dict.copy()
        for k in neg_dict.keys():
            for v in neg_dict[k]:
                for i in label_dict_new.keys():
                    if v in label_dict_new[i]:
                        neg_dict_new[k] = list(set(neg_dict_new[k]+label_dict_new[i]))


        # 從字典建立擴張後的 pair df
        pair_df_neg_expand = pd.DataFrame({'fc_id':[], 'em_id':[], 'label':[]})
        for k in neg_dict_new.keys():
            for v in neg_dict_new[k]:
                pair_df_neg_expand = pair_df_neg_expand.append({'fc_id':k, 'em_id':v, 'label':0}, ignore_index=True)


        return pair_df_pos_expand, pair_df_neg_expand

    pair_df_pos_expand, pair_df_neg_expand = cluster_expansion(true_label_pair_df, false_label_pair_df)

    def image_expand(data_np, pair_df_neg_expand, pair_df):
        # 增加 data_np, 更新 label lst
        data_np_add = np.zeros((len(pair_df_neg_expand), data_np.shape[1],data_np.shape[2], data_np.shape[3], data_np.shape[4]))
        for i in range(len(data_np_add)):
            fc_id = pair_df_neg_expand['fc_id'][i]
            em_id = pair_df_neg_expand['em_id'][i]
            for j in range(len(data_np)):
                if pair_df['fc_id'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['em_id'][k] == em_id:
                    data_np_add[i,1,:] = data_np[k,1,:]
                    break

        data_np = np.concatenate((data_np, data_np_add))
        label_lst = np.array(pair_df['label'].append(pair_df_neg_expand['label']),dtype=np.int32)
        return data_np, label_lst

    # expand negative data
    data_np_train, label_lst_train = image_expand(data_np_train, pair_df_neg_expand, nrn_pair_train)

    def image_expand2(data_np, pair_df_neg_expand, pair_df, label_lst):
        # 增加 data_np, 更新 label lst
        data_np_add = np.zeros((len(pair_df_neg_expand), data_np.shape[1],data_np.shape[2], data_np.shape[3], data_np.shape[4]))
        for i in range(len(data_np_add)):
            fc_id = pair_df_neg_expand['fc_id'][i]
            em_id = pair_df_neg_expand['em_id'][i]
            for j in range(len(data_np)):
                if pair_df['fc_id'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['em_id'][k] == em_id:
                    data_np_add[i,1,:] = data_np[k,1,:]
                    break

        data_np = np.concatenate((data_np, data_np_add))
        label_lst = np.concatenate((label_lst, np.array(pair_df_neg_expand['label'],dtype=np.int32)))
        return data_np, label_lst

    # expand positive data
    X_train, y_train = image_expand2(data_np_train, pair_df_pos_expand, nrn_pair_train, label_lst_train)
else:
    X_train = data_np_train
    y_train = np.array(nrn_pair_train['label'],dtype=np.int32)





# 找出 label為1的 X_train
true_label_idx, false_label_idx = [], []
for i in range(y_train.shape[0]):
    if y_train[i] == 1:
        true_label_idx.append(i)
    else:
        false_label_idx.append(i)

# Balanced Weight
neg, pos = np.bincount(y_train)     #label為0, label為1
print('Total: {}\nPositive: {} ({:.2f}% of total)\n'.format(neg + pos, pos, 100 * pos / (neg + pos)))
weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0:weight[0], 1:weight[1]}
print('Balanced Weight in: \n', np.unique(y_train),'\n', weight)



neg, pos = np.bincount(y_train)     #label為0, label為1

# UpSampling: Augmentation label為1的 X_train(旋轉)
X_train_add = np.zeros((abs(neg-pos), X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))   # 製作需要增加的x_train 量

if neg > pos:
    x_add_idx = true_label_idx
    y_train_add = np.ones(X_train_add.shape[0], dtype=np.int64)

else:
    x_add_idx = false_label_idx
    y_train_add = np.zeros(X_train_add.shape[0], dtype=np.int64)


k=0
for i in range(X_train_add.shape[0]):
    rotation_angle = 1  # 1*90 度旋轉
    X_train_add[i,0,:] = np.rot90(X_train[x_add_idx[k],0,:],rotation_angle) # FC img
    X_train_add[i,1,:] = np.rot90(X_train[x_add_idx[k],1,:],rotation_angle) # EM img

    if k >= len(x_add_idx)-1:
        k=0
        rotation_angle += 1
    else:
        k+=1


X_train = np.vstack((X_train, X_train_add))
y_train = np.hstack((y_train, y_train_add))

print('UpSampling: After Augmentation:\nTrue Label/Total in X_train:\n',np.sum(y_train),'/', len(X_train))



# 圖片旋轉任一角度
def rotate_and_pad(image, angle, border_value=(0, 0, 0)):
    # 获取图像尺寸
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # 计算旋转矩阵
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新图像的尺寸
    new_w = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
    new_h = int(h * abs(np.cos(np.radians(angle))) + w * abs(np.sin(np.radians(angle))))

    # 更新旋转矩阵
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # 应用旋转和填充
    rotated_image = cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=border_value)

    # 裁剪或填充旋转后的图像以保持原始尺寸
    if new_h > h and new_w > w:
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        rotated_image = rotated_image[y_offset:y_offset + h, x_offset:x_offset + w]
    else:
        y_padding_top = (h - new_h) // 2
        y_padding_bottom = h - new_h - y_padding_top
        x_padding_left = (w - new_w) // 2
        x_padding_right = w - new_w - x_padding_left
        rotated_image = cv2.copyMakeBorder(rotated_image, y_padding_top, y_padding_bottom, x_padding_left, x_padding_right, cv2.BORDER_CONSTANT, value=border_value)

    return rotated_image


def augment_data(X_train, y_train, angle_range, resize_range, aug_seed):
    X_augmented, y_augmented = [], []

    for i in range(X_train.shape[0]):
        current_seed = aug_seed + i         #為每個循環定義一個種子。每張圖片旋轉角度因此不同
        rng = np.random.default_rng(current_seed)
        angle = rng.uniform(angle_range[0], angle_range[1])
        # scale = rng.random.uniform(resize_range[0], resize_range[1])

        rotate_pair = np.zeros(X_train.shape[1:])   # shape=(2,50,50,3)
        resize_pair = np.zeros(X_train.shape[1:])
        for j in range(X_train.shape[1]):
            rotate_pair[j] = rotate_and_pad(X_train[i, j], angle)
            # resize_pair[j] = resize_and_pad(X_train[i, j], scale)
        
        X_augmented.append(rotate_pair)
        # X_augmented.append(resize_pair)

        y_augmented.append(y_train[i])
        # y_augmented.append(y_train[i])

    return np.array(X_augmented), np.array(y_augmented)

# # 示例用法
# angle_range = [-45, 45]  # 旋转角度范围（在 -10 到 10 之间）
# resize_range = [0.8, 1.2]   # 縮放範圍（在 0.8 到 1.2 之間）
# X_train_augmented, y_train_augmented = augment_data(X_train, y_train, angle_range, resize_range, seed)

# X_train = np.vstack((X_train, X_train_augmented))
# y_train = np.hstack((y_train, y_train_augmented))






# 翻倍  All train data augmentation

X_train_aug1 = np.zeros_like(X_train)
for i in range(X_train_aug1.shape[0]):
    X_train_aug1[i,0,:] = np.fliplr(X_train[i,0,:])
    X_train_aug1[i,1,:] = np.fliplr(X_train[i,1,:])

X_train = np.vstack((X_train, X_train_aug1))
y_train = np.hstack((y_train, y_train))


# # 翻倍
# X_train_aug2 = np.zeros_like(X_train)

# for i in range(X_train_aug2.shape[0]):
#     X_train_aug2[i,0,:] = np.flipud(X_train[i,0,:])
#     X_train_aug2[i,1,:] = np.flipud(X_train[i,1,:])

# X_train = np.vstack((X_train, X_train_aug2))
# y_train = np.hstack((y_train, y_train))


del X_train_add, X_train_aug1
# del X_train_aug2

# # # 翻倍
X_train_aug3 = np.zeros_like(X_train)

for i in range(X_train_aug3.shape[0]):
    X_train_aug3[i,0,:] = np.flipud(np.rot90(X_train[i,0,:],1))
    X_train_aug3[i,1,:] = np.flipud(np.rot90(X_train[i,1,:],1))

# 判斷x_train是否太大, 否則內存不足
if len(X_train) + len(X_train_aug3) > 90000:
    X_train_aug3 = X_train_aug3[:(90000-len(X_train))]
    y_train_aug3 = y_train[:(90000-len(y_train))]

    X_train = np.vstack((X_train, X_train_aug3))
    y_train = np.hstack((y_train, y_train_aug3))
else:
    X_train = np.vstack((X_train, X_train_aug3))
    y_train = np.hstack((y_train, y_train))

# FC/EM Split
X_train_FC = X_train[:,0,:]
X_train_EM = X_train[:,1,:]

X_val_FC = X_val[:,0,:]
X_val_EM = X_val[:,1,:]


print('X_train shape:', X_train_FC.shape, X_train_EM.shape)
print('y_train shape:', len(y_train))
print('X_val shape:', X_val_FC.shape, X_val_EM.shape)
print('y_val shape:', len(y_val))








from model import CNN_best, CNN_shared

# cnn = CNN_best((resolutions[1],resolutions[2],3))
cnn = CNN_shared((resolutions[1], resolutions[2],3))

train_epochs = 50


reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)

# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint('./Second_Stage_Model/' + save_model_name + '.h5', verbose=2, monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode="auto")


# Scheduler
if use_scheduler:
    second_history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=128, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=train_epochs, shuffle=True, callbacks = [checkpoint, reduce_lr], verbose=2)
else:
    cnn.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[BinaryAccuracy(name='Bi-acc')])
    second_history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=128, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=train_epochs, shuffle=True, callbacks = [checkpoint], verbose=2)

plt.plot(second_history.history['loss'], label='loss')
plt.plot(second_history.history['val_loss'], label='val_loss')
plt.legend()
# plt.savefig('./Figure/Second_Train_Curve_' + str(num_splits) + '.png', dpi=100)
plt.show()
plt.close('all')

# cnn_train_loss = history.history['loss']
# cnn_valid_loss = history.history['val_loss']
with open('./result/Train_History_Second_stage_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(second_history.history, f)

# %% Testing 1

# turn to numpy array
test_pair_nrn = label_table_test[['fc_id','em_id','label']].to_numpy()


if use_map_from == 'kt':
    # 讀神經三視圖資料
    map_data_D1toD4 = load_pkl('./data/mapping_data_sn.pkl')
    # map_data(lst) 中每一项内容为: 'FC nrn','EM nrn ', Score, FC Array, EM Array

    map_data_D5 = load_pkl('./data/mapping_data_sn_D5_old.pkl')

    map_data = map_data_D1toD4 + map_data_D5
    del map_data_D1toD4, map_data_D5


    def data_preprocess(map_data, pair_nrn):
        data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
        FC_nrn_lst, EM_nrn_lst, score_lst, label_lst = [], [], [], []

        #使用字典存储有三視圖数据, 以 FC_EM 作为键, 使用字典来查找相应的数据, 减少查找时间
        data_dict = {}
        for data in map_data:
            key = f"{data[0]}_{data[1]}"
            data_dict[key] = data

        for i, row in enumerate(pair_nrn):
            key = f'{row[0]}_{row[1]}'
            if key in data_dict:
                data = data_dict[key]   # 找出data的所有信息              

                # 三視圖填入data_np
                for k in range(3):
                    data_np[i, 0, :, :, k] = data[3][k] # FC Image
                    data_np[i, 1, :, :, k] = data[4][k] # EM Image
                
                # 其餘信息填入
                FC_nrn_lst.append(data[0])
                EM_nrn_lst.append(data[1])
                score_lst.append(data[2]) 
                label_lst.append(row[2])

        # map data 中有可能找不到pair_nrn裡面的組合, 刪除那些找不到的0矩陣
        not_found_data = []
        for i, data in enumerate(data_np):
            if not(np.any(data)):
                not_found_data.append(i)
        data_np = np.delete(data_np, not_found_data, axis=0)

        if not_found_data:
            print('Not Found in map_data: ')
            for i in not_found_data:
                print(pair_nrn[i])


        # Normalization : x' = x - min(x) / max(x) - min(x)
        data_np = (data_np - np.min(data_np))/(np.max(data_np) - np.min(data_np))

        pair_df = pd.DataFrame({'fc_id':FC_nrn_lst, 'em_id':EM_nrn_lst, 'label':label_lst, 'score':score_lst})    # list of pairs

        return data_np, pair_df
    
    
    data_np_test, nrn_pair_test = data_preprocess(map_data, test_pair_nrn)



elif use_map_from == 'yf':

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




X_test = data_np_test
y_test = np.array(nrn_pair_test['label'],dtype=np.int32)

X_test_FC = X_test[:,0,:]
X_test_EM = X_test[:,1,:]



# %%
model = load_model('./Second_Stage_Model/' + save_model_name + '.h5')
y_pred = model.predict({'FC':X_test_FC, 'EM':X_test_EM})
pred_test_compare = np.hstack((y_pred, y_test.reshape(len(y_test), 1)))
y_pred_binary = []
for ans in y_pred:
    if ans > 0.5:
        y_pred_binary.append(1)
    else:
        y_pred_binary.append(0)

conf_matrix = confusion_matrix(y_test.tolist(), y_pred_binary, labels=[1,0])# 統一標籤格式

def print_conf_martix(conf_matrix, name='0'):

    print('\nConfusion Matrix for ' + name)
    print('True Pos','False Neg')
    print(conf_matrix[0])
    print('False Pos','True Neg')
    print(conf_matrix[1])

print_conf_martix(conf_matrix)


# Precision and recall
precision = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0])
recall = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[0,1])
print("Precision:", precision)
print("Recall:", recall)

# F1 Score
result_f1_score = f1_score(y_test, y_pred_binary, average=None)
print('F1 Score for Neg:', result_f1_score[0])
print('F1 Score for Pos:', result_f1_score[1])

# save results
result = {'conf_matrix': conf_matrix, 'Precision': precision, 'Recall': recall, 'F1_pos':result_f1_score[1]}

with open('./result/Second_stage_Result_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(result, f)





# %% 冻结前层参数

if fix_kernel:
    for layer in model.layers[:-5]:
        layer.trainable = False

    # 确认模型状态
    for layer in model.layers:
        print(layer, layer.trainable)
    # model.summary()

    # 编译模型
    model.compile(optimizer='rmsprop', loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics = [BinaryAccuracy(name='Bi-Acc')])
else:
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[BinaryAccuracy(name='Bi-Acc')])


# 准备人工标注的train data
train_pair_nrn = label_table_train[['fc_id','em_id','label']].to_numpy()
if use_map_from == 'kt':
    data_np_train, nrn_pair_train = data_preprocess(map_data, train_pair_nrn)
elif use_map_from == 'yf':
    data_np_train, nrn_pair_train, train_not_found = data_preprocess(map_dict_folder, train_pair_nrn)


# Train Validation Split
data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.2, random_state=seed)

print('\nTrain data:', len(data_np_train),'\nValid data:', len(data_np_valid),'\nTest data:', len(data_np_test))

X_val = data_np_valid
y_val = np.array(nrn_pair_valid['label'],dtype=np.int32)


cross_expand = False

if cross_expand:
    true_label_pair_df = nrn_pair_train.loc[nrn_pair_train['label']==1]
    false_label_pair_df = nrn_pair_train.loc[nrn_pair_train['label']==0]

    def cluster_expansion(pair_df_pos, pair_df_neg):

        label_dict = {}
        for _, row in pair_df_pos.iterrows():
            if row['fc_id'] in label_dict:
                label_dict[row['fc_id']].append(row['em_id'])
            else:
                label_dict[row['fc_id']] = [row['em_id']]

        # 以FC為key的dictionary建立完成，接下來搜索可擴張項
        label_dict_new = label_dict.copy()
        for k in label_dict.keys():
            for v in label_dict[k]:
                for i in label_dict.keys():
                    if v in label_dict[i] and i != k:
                        label_dict_new[k] = list(set(label_dict[k]+label_dict[i]))

        pair_df_pos_expand = pd.DataFrame({'fc_id':[], 'em_id':[], 'label':[]})
        for k in label_dict_new.keys():
            for v in label_dict_new[k]:
                pair_df_pos_expand = pair_df_pos_expand.append({'fc_id':k, 'em_id':v, 'label':1}, ignore_index=True)


        # label=1資料擴張完畢，現在製造label=0的pair使資料量平衡  
        neg_dict = {}
        for _, row in pair_df_neg.iterrows():
            if row['fc_id'] in neg_dict:
                neg_dict[row['fc_id']].append(row['em_id'])
            else:
                neg_dict[row['fc_id']] = [row['em_id']]


        #擴張
        neg_dict_new = neg_dict.copy()
        for k in neg_dict.keys():
            for v in neg_dict[k]:
                for i in label_dict_new.keys():
                    if v in label_dict_new[i]:
                        neg_dict_new[k] = list(set(neg_dict_new[k]+label_dict_new[i]))


        # 從字典建立擴張後的 pair df
        pair_df_neg_expand = pd.DataFrame({'fc_id':[], 'em_id':[], 'label':[]})
        for k in neg_dict_new.keys():
            for v in neg_dict_new[k]:
                pair_df_neg_expand = pair_df_neg_expand.append({'fc_id':k, 'em_id':v, 'label':0}, ignore_index=True)


        return pair_df_pos_expand, pair_df_neg_expand

    pair_df_pos_expand, pair_df_neg_expand = cluster_expansion(true_label_pair_df, false_label_pair_df)

    def image_expand(data_np, pair_df_neg_expand, pair_df):
        # 增加 data_np, 更新 label lst
        data_np_add = np.zeros((len(pair_df_neg_expand), data_np.shape[1],data_np.shape[2], data_np.shape[3], data_np.shape[4]))
        for i in range(len(data_np_add)):
            fc_id = pair_df_neg_expand['fc_id'][i]
            em_id = pair_df_neg_expand['em_id'][i]
            for j in range(len(data_np)):
                if pair_df['fc_id'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['em_id'][k] == em_id:
                    data_np_add[i,1,:] = data_np[k,1,:]
                    break

        data_np = np.concatenate((data_np, data_np_add))
        label_lst = np.array(pair_df['label'].append(pair_df_neg_expand['label']),dtype=np.int32)
        return data_np, label_lst

    # expand negative data
    data_np_train, label_lst_train = image_expand(data_np_train, pair_df_neg_expand, nrn_pair_train)

    def image_expand2(data_np, pair_df_neg_expand, pair_df, label_lst):
        # 增加 data_np, 更新 label lst
        data_np_add = np.zeros((len(pair_df_neg_expand), data_np.shape[1],data_np.shape[2], data_np.shape[3], data_np.shape[4]))
        for i in range(len(data_np_add)):
            fc_id = pair_df_neg_expand['fc_id'][i]
            em_id = pair_df_neg_expand['em_id'][i]
            for j in range(len(data_np)):
                if pair_df['fc_id'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['em_id'][k] == em_id:
                    data_np_add[i,1,:] = data_np[k,1,:]
                    break

        data_np = np.concatenate((data_np, data_np_add))
        label_lst = np.concatenate((label_lst, np.array(pair_df_neg_expand['label'],dtype=np.int32)))
        return data_np, label_lst

    # expand positive data
    X_train, y_train = image_expand2(data_np_train, pair_df_pos_expand, nrn_pair_train, label_lst_train)
else:
    X_train = data_np_train
    y_train = np.array(nrn_pair_train['label'],dtype=np.int32)





# 找出 label為1的 X_train
true_label_idx, false_label_idx = [], []
for i in range(y_train.shape[0]):
    if y_train[i] == 1:
        true_label_idx.append(i)
    else:
        false_label_idx.append(i)

# Balanced Weight
neg, pos = np.bincount(y_train)     #label為0, label為1
print('Total: {}\nPositive: {} ({:.2f}% of total)\n'.format(neg + pos, pos, 100 * pos / (neg + pos)))
weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0:weight[0], 1:weight[1]}
print('Balanced Weight in: \n', np.unique(y_train),'\n', weight)


if add_low_score and pos >= neg:
    X_train_add = np.zeros((low_score_neg_rate*(pos-neg), X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))   # 製作需要增加的x_train 量
    y_train_add = np.zeros(X_train_add.shape[0], dtype=np.int64)
    
    nrn_pair_test_np = nrn_pair_test[['fc_id','em_id']].to_numpy()    # 以np格式獲取test中pair的神經名稱
    
    k=0
    for i in range(X_train_add.shape[0]):
        for j in range(k, len(map_data)):
            in_test = 0 # 檢查擴增的資料組是否出現在test中
            for row in range(nrn_pair_test_np.shape[0]):
                if str(map_data[j][0]) == str(nrn_pair_test_np[row,0]) and str(map_data[j][1]) == str(nrn_pair_test_np[row,1]):
                    in_test = 1
                    break
            
            if in_test == 0 and map_data[j][2] < 0.4:

                for n in range(3):
                    X_train_add[i, 0, :, :, n] = map_data[j][3][n] # FC Image
                    X_train_add[i, 1, :, :, n] = map_data[j][4][n] # EM Image

                k=j+1
                break

    X_train = np.vstack((X_train, X_train_add))
    y_train = np.hstack((y_train, y_train_add))



neg, pos = np.bincount(y_train)     #label為0, label為1

# UpSampling: Augmentation label為1的 X_train(旋轉)
X_train_add = np.zeros((abs(neg-pos), X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))   # 製作需要增加的x_train 量

if neg > pos:
    x_add_idx = true_label_idx
    y_train_add = np.ones(X_train_add.shape[0], dtype=np.int64)

else:
    x_add_idx = false_label_idx
    y_train_add = np.zeros(X_train_add.shape[0], dtype=np.int64)


k=0
for i in range(X_train_add.shape[0]):
    rotation_angle = 1  # 1*90 度旋轉
    X_train_add[i,0,:] = np.rot90(X_train[x_add_idx[k],0,:],rotation_angle) # FC img
    X_train_add[i,1,:] = np.rot90(X_train[x_add_idx[k],1,:],rotation_angle) # EM img

    if k >= len(x_add_idx)-1:
        k=0
        rotation_angle += 1
    else:
        k+=1


X_train = np.vstack((X_train, X_train_add))
y_train = np.hstack((y_train, y_train_add))

print('UpSampling: After Augmentation:\nTrue Label/Total in X_train:\n',np.sum(y_train),'/', len(X_train))



# 圖片旋轉任一角度
def rotate_and_pad(image, angle, border_value=(0, 0, 0)):
    # 获取图像尺寸
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # 计算旋转矩阵
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新图像的尺寸
    new_w = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
    new_h = int(h * abs(np.cos(np.radians(angle))) + w * abs(np.sin(np.radians(angle))))

    # 更新旋转矩阵
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # 应用旋转和填充
    rotated_image = cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=border_value)

    # 裁剪或填充旋转后的图像以保持原始尺寸
    if new_h > h and new_w > w:
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        rotated_image = rotated_image[y_offset:y_offset + h, x_offset:x_offset + w]
    else:
        y_padding_top = (h - new_h) // 2
        y_padding_bottom = h - new_h - y_padding_top
        x_padding_left = (w - new_w) // 2
        x_padding_right = w - new_w - x_padding_left
        rotated_image = cv2.copyMakeBorder(rotated_image, y_padding_top, y_padding_bottom, x_padding_left, x_padding_right, cv2.BORDER_CONSTANT, value=border_value)

    return rotated_image


# def resize_and_pad(image, scale_factor):
#     augmented_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

#     # 裁剪或填充缩放后的图像以保持原始尺寸
#     height, width = image.shape[:2]
#     if scale_factor > 1:
#         y_offset = (augmented_image.shape[0] - height) // 2
#         x_offset = (augmented_image.shape[1] - width) // 2
#         augmented_image = augmented_image[y_offset:y_offset+height, x_offset:x_offset+width]
#     else:
#         y_padding_top = (height - augmented_image.shape[0]) // 2
#         y_padding_bottom = height - augmented_image.shape[0] - y_padding_top
#         x_padding_left = (width - augmented_image.shape[1]) // 2
#         x_padding_right = width - augmented_image.shape[1] - x_padding_left
#         augmented_image = cv2.copyMakeBorder(augmented_image, y_padding_top, y_padding_bottom, x_padding_left, x_padding_right, cv2.BORDER_CONSTANT, value=0)
    
#     return augmented_image


def augment_data(X_train, y_train, angle_range, resize_range, aug_seed):
    X_augmented, y_augmented = [], []

    for i in range(X_train.shape[0]):
        current_seed = aug_seed + i         #為每個循環定義一個種子。每張圖片旋轉角度因此不同
        rng = np.random.default_rng(current_seed)
        angle = rng.uniform(angle_range[0], angle_range[1])
        # scale = rng.random.uniform(resize_range[0], resize_range[1])

        rotate_pair = np.zeros(X_train.shape[1:])   # shape=(2,50,50,3)
        resize_pair = np.zeros(X_train.shape[1:])
        for j in range(X_train.shape[1]):
            rotate_pair[j] = rotate_and_pad(X_train[i, j], angle)
            # resize_pair[j] = resize_and_pad(X_train[i, j], scale)
        
        X_augmented.append(rotate_pair)
        # X_augmented.append(resize_pair)

        y_augmented.append(y_train[i])
        # y_augmented.append(y_train[i])

    return np.array(X_augmented), np.array(y_augmented)

# 示例用法
angle_range = [-45, 45]  # 旋转角度范围（在 -10 到 10 之间）
resize_range = [0.8, 1.2]   # 縮放範圍（在 0.8 到 1.2 之間）
X_train_augmented, y_train_augmented = augment_data(X_train, y_train, angle_range, resize_range, seed)

X_train = np.vstack((X_train, X_train_augmented))
y_train = np.hstack((y_train, y_train_augmented))

# 再做一次
X_train_augmented, y_train_augmented = augment_data(X_train, y_train, angle_range, resize_range, seed+10000)

X_train = np.vstack((X_train, X_train_augmented))
y_train = np.hstack((y_train, y_train_augmented))






# 翻倍  All train data augmentation

X_train_aug1 = np.zeros_like(X_train)
for i in range(X_train_aug1.shape[0]):
    X_train_aug1[i,0,:] = np.fliplr(X_train[i,0,:])
    X_train_aug1[i,1,:] = np.fliplr(X_train[i,1,:])

X_train = np.vstack((X_train, X_train_aug1))
y_train = np.hstack((y_train, y_train))
# 翻倍
X_train_aug2 = np.zeros_like(X_train)

for i in range(X_train_aug2.shape[0]):
    X_train_aug2[i,0,:] = np.flipud(X_train[i,0,:])
    X_train_aug2[i,1,:] = np.flipud(X_train[i,1,:])

X_train = np.vstack((X_train, X_train_aug2))
y_train = np.hstack((y_train, y_train))

# 翻倍
X_train_aug3 = np.zeros_like(X_train)

for i in range(X_train_aug3.shape[0]):
    X_train_aug3[i,0,:] = np.flipud(np.rot90(X_train[i,0,:],1))
    X_train_aug3[i,1,:] = np.flipud(np.rot90(X_train[i,1,:],1))

X_train = np.vstack((X_train, X_train_aug3))
y_train = np.hstack((y_train, y_train))




# FC/EM Split
X_train_FC = X_train[:,0,:]
X_train_EM = X_train[:,1,:]

X_val_FC = X_val[:,0,:]
X_val_EM = X_val[:,1,:]

X_test_FC = X_test[:,0,:]
X_test_EM = X_test[:,1,:]

print('X_train shape:', X_train_FC.shape, X_train_EM.shape)
print('y_train shape:', len(y_train))
print('X_val shape:', X_val_FC.shape, X_val_EM.shape)
print('y_val shape:', len(y_val))
print('X_test shape:', X_test_FC.shape, X_test_EM.shape)
print('y_test shape:', len(y_test))



train_epochs = 100

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=0)

# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint('./Final_Model/' + save_model_name + '.h5', verbose=2, monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode="auto")


# Scheduler
if use_scheduler:
    final_history = model.fit({'FC':X_train_FC, 'EM':X_train_EM}, 
                              y_train,
                              batch_size=128, 
                              validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val),
                              epochs=train_epochs,
                              shuffle=True,
                              callbacks = [checkpoint, reduce_lr],
                              verbose=2)
else:
    final_history = model.fit({'FC':X_train_FC, 'EM':X_train_EM},
                                y_train,
                                batch_size = 128,
                                validation_data = ({'FC':X_val_FC, 'EM':X_val_EM}, y_val),
                                epochs = train_epochs,
                                shuffle = True,
                                callbacks = [checkpoint],
                                verbose=2)


plt.plot(final_history.history['loss'], label='loss')
plt.plot(final_history.history['val_loss'], label='val_loss')
plt.legend()
# plt.savefig('./Figure/Final_Train_Curve_' + str(num_splits) + '.png', dpi=100)
plt.show()
plt.close('all')
with open('./result/Train_History_Final_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(final_history.history, f)

# %%
model = load_model('./Final_Model/' + save_model_name + '.h5')
y_pred = model.predict({'FC':X_test_FC, 'EM':X_test_EM})
pred_test_compare = np.hstack((y_pred, y_test.reshape(len(y_test), 1)))
y_pred_binary = []
for ans in y_pred:
    if ans > 0.5:
        y_pred_binary.append(1)
    else:
        y_pred_binary.append(0)

conf_matrix = confusion_matrix(y_test.tolist(), y_pred_binary, labels=[1,0])# 統一標籤格式

def print_conf_martix(conf_matrix, name='0'):

    print('\nConfusion Matrix for ' + name)
    print('True Pos','False Neg')
    print(conf_matrix[0])
    print('False Pos','True Neg')
    print(conf_matrix[1])

print_conf_martix(conf_matrix)



# Precision and recall
precision = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0])
recall = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[0,1])
print("Precision:", precision)
print("Recall:", recall)

# F1 Score
result_f1_score = f1_score(y_test, y_pred_binary, average=None)
print('F1 Score for Neg:', result_f1_score[0])
print('F1 Score for Pos:', result_f1_score[1])

# save results
result = {'conf_matrix': conf_matrix, 'Precision': precision, 'Recall': recall, 'F1_pos':result_f1_score[1]}

with open('./result/Final_stage_Result_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(result, f)

print('Result Saved.')

# Save model prediction csv
pred_result_df = nrn_pair_test.copy()
pred_result_df['model_pred'] = y_pred
pred_result_df['model_pred_binary'] = y_pred_binary

# 将DataFrame存储为csv文件
pred_result_df.to_csv('./result/final_label_'+save_model_name+'.csv', index=False)
print('\nSaved')



print('\nProgram Completed')
# %%
