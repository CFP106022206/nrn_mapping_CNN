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
from tqdm import tqdm


# %%
num_splits = 9 #0~9, or 99 for whole nBLAST testing set

'''
使用冠廷的檔案寫法，因冠廷的檔案全部混在同一個黃瓜中.
新寫法是使用和data_preprocess_annotator 相同的方法。
'''
use_map_from = 'yf' #'kt': map_data 冠廷, 'yf': map_folder from 懿凡
map_dict_folder = './data/labeled_sn'

grid75_path = './data/D1-D5_grid75_sn'


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

# load train, test
label_table_train = pd.read_csv('./train_test_split/train_split_' + str(num_splits) +'_D1-D6.csv')
label_table_test = pd.read_csv('./train_test_split/test_split_' + str(num_splits) +'_D1-D6.csv')


# turn to numpy array
test_pair_nrn = label_table_test[['fc_id','em_id','label']].to_numpy()
train_pair_nrn = label_table_train[['fc_id','em_id','label']].to_numpy()





# %% data prerpare

'''
使用冠廷的檔案寫法，因冠廷的檔案全部混在同一個黃瓜中.
新寫法是使用和data_preprocess_annotator 相同的方法。
'''
if use_map_from == 'kt':
    # # 讀神經三視圖資料
    map_data_D1toD4 = load_pkl('./data/mapping_data_sn.pkl')
    # map_data(lst) 中每一项内容为: 'FC nrn','EM nrn ', Score, FC Array, EM Array

    map_data_D5 = load_pkl('./data/mapping_data_sn_D5_old.pkl')

    map_data = map_data_D1toD4 + map_data_D5
    del map_data_D1toD4, map_data_D5

    resolutions = map_data[0][3].shape
    print('Image shape: ', resolutions)

    def data_preprocess(map_data, pair_nrn):
        data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
        fc_nrn_lst, em_nrn_lst, score_lst, label_lst = [], [], [], []

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

    data_np_test, nrn_pair_test, test_not_found = data_preprocess(map_data, test_pair_nrn)
    data_np_train, nrn_pair_train, train_not_found = data_preprocess(map_data, train_pair_nrn)


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
    data_np_train, nrn_pair_train, train_not_found = data_preprocess(map_dict_folder, train_pair_nrn)



# %% Train Validation Split
data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.15, random_state=7)

print('\nOriginal Train data:', len(data_np_train),'\nValid data:', len(data_np_valid),'\nTest data:', len(data_np_test))


x_val = data_np_valid
X_test = data_np_test
y_val = np.array(nrn_pair_valid['label'])
y_test = np.array(nrn_pair_test['label'])




# %% 画图预览 map data
def imshow_pred_pair(predict_pair_df, pred_data_np):

    # 检查保存路径文件夹是否存在
    if not os.path.exists('./Figure/predict_3view/label_1'):
        os.makedirs('./Figure/predict_3view/label_1')
    
    if not os.path.exists('./Figure/predict_3view/label_0'):
        os.makedirs('./Figure/predict_3view/label_0')

    for p in range(len(predict_pair_df)):
        fc_img = pred_data_np[p,0,:]
        em_img = pred_data_np[p,1,:]

        fc_id = predict_pair_df.iloc[p]['fc_id']
        em_id = predict_pair_df.iloc[p]['em_id']
        label = predict_pair_df.iloc[p]['label']

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

        plt.suptitle(f'{fc_id}_{em_id}     Label={label}')

        if label == 1:
            plt.savefig(f'./Figure/predict_3view/label_1/{fc_id}_{em_id}.png', dpi=150, bbox_inches='tight')
        elif label == 0:
            plt.savefig(f'./Figure/predict_3view/label_0/{fc_id}_{em_id}.png', dpi=150, bbox_inches='tight')
        plt.close('all')

# imshow_pred_pair(nrn_pair_train, data_np_train)
# imshow_pred_pair(nrn_pair_test, data_np_test)



# %% Data Augmentation: Exchange 'fc' and 'em' data
x_train = data_np_train.copy()
y_train = np.array(nrn_pair_train['label'])

# 交換 FC/EM
x_train = np.vstack((x_train, np.flip(x_train, axis=1)))
y_train = np.hstack((y_train, y_train))

y_train_bin = np.array([1 if y > 0.5 else 0 for y in y_train])





# %% Balanced Weight
neg, pos = np.bincount(y_train_bin)     #label為0, label為1
print('\nTotal(After exchange): {}\nPositive: {} ({:.2f}% of total)\n'.format(neg + pos, pos, 100 * pos / (neg + pos)))
weight = compute_class_weight('balanced', classes=np.unique(y_train_bin), y=y_train_bin)
class_weights = {0:weight[0]*100, 1:weight[1]}
print('Balanced Weight in:\n', weight)


# UpSampling
X_train_add = np.zeros((abs(neg-pos), x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]))   # 製作需要增加的x_train 量
y_train_add = np.zeros(abs(neg-pos))

if neg > pos:
    add_idx = np.where(y_train_bin == 1)[0] #數據擴增在 label為1的 x_train

else:
    add_idx = np.where(y_train_bin == 0)[0]#數據擴增在 label為0的 x_train


k=0
for i in range(X_train_add.shape[0]):
    rotation_angle = 1  # 1*90 度旋轉
    X_train_add[i,0,:] = np.rot90(x_train[add_idx[k],0,:],rotation_angle) # FC img
    X_train_add[i,1,:] = np.rot90(x_train[add_idx[k],1,:],rotation_angle) # EM img

    y_train_add[i] = y_train[add_idx[k]]

    if k >= len(add_idx):
        k=0
        rotation_angle += 1
    else:
        k+=1


x_train = np.vstack((x_train, X_train_add))
y_train = np.hstack((y_train, y_train_add))

print('UpSampling: After label balancing:\nTrue Label/Total in x_train:\n',np.sum(y_train_bin),'/', len(x_train))

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


def augment_data(x_train, y_train, angle_range, resize_range, aug_seed):
    X_augmented, y_augmented = [], []

    for i in range(x_train.shape[0]):
        current_seed = aug_seed + i         #為每個循環定義一個種子。每張圖片旋轉角度因此不同
        rng = np.random.default_rng(current_seed)
        angle = rng.uniform(angle_range[0], angle_range[1])
        # scale = rng.random.uniform(resize_range[0], resize_range[1])

        rotate_pair = np.zeros(x_train.shape[1:])   # shape=(2,50,50,3)
        resize_pair = np.zeros(x_train.shape[1:])
        for j in range(x_train.shape[1]):
            rotate_pair[j] = rotate_and_pad(x_train[i, j], angle)

        X_augmented.append(rotate_pair)
        y_augmented.append(y_train[i])

    return np.array(X_augmented), np.array(y_augmented)

# 示例用法
angle_range = [-45, 45]  # 旋转角度范围（在 -10 到 10 之间）
resize_range = [0.8, 1.2]   # 縮放範圍（在 0.8 到 1.2 之間）
X_train_augmented, y_train_augmented = augment_data(x_train, y_train, angle_range, resize_range, seed)

x_train = np.vstack((x_train, X_train_augmented))
y_train = np.hstack((y_train, y_train_augmented))

# 再做一次
X_train_augmented, y_train_augmented = augment_data(x_train, y_train, angle_range, resize_range, seed+10000)

x_train = np.vstack((x_train, X_train_augmented))
y_train = np.hstack((y_train, y_train_augmented))

# # 再做一次
# X_train_augmented, y_train_augmented = augment_data(x_train, y_train, angle_range, resize_range, seed+10000)

# x_train = np.vstack((x_train, X_train_augmented))
# y_train = np.hstack((y_train, y_train_augmented))


# # 再做一次
# X_train_augmented, y_train_augmented = augment_data(x_train, y_train, angle_range, resize_range, seed+10000)

# x_train = np.vstack((x_train, X_train_augmented))
# y_train = np.hstack((y_train, y_train_augmented))


# 翻倍  All train data augmentation

X_train_aug1 = np.zeros_like(x_train)
for i in range(X_train_aug1.shape[0]):
    X_train_aug1[i,0,:] = np.fliplr(x_train[i,0,:])
    X_train_aug1[i,1,:] = np.fliplr(x_train[i,1,:])

x_train = np.vstack((x_train, X_train_aug1))
y_train = np.hstack((y_train, y_train))

del X_train_aug1

# 翻倍
X_train_aug2 = np.zeros_like(x_train)

for i in range(X_train_aug2.shape[0]):
    X_train_aug2[i,0,:] = np.flipud(x_train[i,0,:])
    X_train_aug2[i,1,:] = np.flipud(x_train[i,1,:])

x_train = np.vstack((x_train, X_train_aug2))
y_train = np.hstack((y_train, y_train))

del X_train_aug2

# 翻倍
X_train_aug3 = np.zeros_like(x_train)

for i in range(X_train_aug3.shape[0]):
    X_train_aug3[i,0,:] = np.flipud(np.rot90(x_train[i,0,:],1))
    X_train_aug3[i,1,:] = np.flipud(np.rot90(x_train[i,1,:],1))

x_train = np.vstack((x_train, X_train_aug3))
y_train = np.hstack((y_train, y_train))

del X_train_aug3


# FC/EM Split
x_train_FC = x_train[:,0,:]
x_train_EM = x_train[:,1,:]

del x_train

x_val_FC = x_val[:,0,:]
x_val_EM = x_val[:,1,:]

x_test_FC = X_test[:,0,:]
x_test_EM = X_test[:,1,:]

print('x_train shape:', x_train_FC.shape, x_train_EM.shape)
print('y_train shape:', len(y_train))
print('x_val shape:', x_val_FC.shape, x_val_EM.shape)
print('y_val shape:', len(y_val))
print('X_test shape:', x_test_FC.shape, x_test_EM.shape)
print('y_test shape:', len(y_test))



# %%

from model import CNN_best, CNN_deep, CNN_shared, CNN_focal, CNN_L2shared
# from tensorflow.keras.utils import plot_model

resolutions = x_train_FC.shape[1:]

cnn = CNN_shared((resolutions[0],resolutions[1],resolutions[2]))
# cnn = CNN_deep((resolutions[0],resolutions[1],resolutions[2]))

# plot_model(cnn, './Figure/Model_Structure.png', show_shapes=True)
if not scheduler_exp:
    cnn.compile(optimizer=AdamW(learning_rate=initial_lr), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[BinaryAccuracy(name='Bi-Acc')])

# Scheduler
def scheduler(epoch, lr): 

    min_lr=0.0000001
    total_epoch = train_epochs
    epoch_lr = lr*((1-epoch/total_epoch)**scheduler_exp)
    if epoch_lr<min_lr:
        epoch_lr = min_lr

    return epoch_lr


reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)

# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint('./Annotator_Model/' + save_model_name + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')


early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode="auto")


if scheduler_exp:
    callbacks = [checkpoint, reduce_lr]
else:
    callbacks = [checkpoint]
print('\nUse Callbacks:', callbacks)


# Model.fit

Annotator_history = cnn.fit({'FC':x_train_FC, 'EM':x_train_EM}, 
                            y_train, 
                            batch_size=128, 
                            validation_data=({'FC':x_val_FC, 'EM':x_val_EM}, y_val), 
                            epochs=train_epochs, 
                            shuffle=True, 
                            callbacks = callbacks, verbose=2)
                            # class_weight=class_weights)



plt.plot(Annotator_history.history['loss'], label='loss')
plt.plot(Annotator_history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('./Figure/Annotator_Train_Curve_'+str(num_splits)+'.png', dpi=150, bbox_inches="tight")
plt.show()
plt.close('all')

# cnn_train_loss = history.history['loss']
# cnn_valid_loss = history.history['val_loss']

# Save history to file
with open('./result/Train_History_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(Annotator_history.history, f)


# %%
model = load_model('./Annotator_Model/' + save_model_name + '.h5')

def binary(y_lst):
    y_binary = []
    for y in y_lst:
        if y > 0.5:
            y_binary.append(1)
        else:
            y_binary.append(0)
    return y_binary

def print_conf_martix(conf_matrix, name='0'):

    print('\nConfusion Matrix for ' + name)
    print('True Pos','False Neg')
    print(conf_matrix[0])
    print('False Pos','True Neg')
    print(conf_matrix[1])

def result_analysis(y_pred, y_test):
    y_pred_binary = binary(y_pred)
    y_test = binary(y_test) # for 软标签，统一格式

    conf_matrix = confusion_matrix(y_test, y_pred_binary, labels=[1,0])# 統一標籤格式

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
    return result, y_pred_binary


# predict validation dataset result
y_pred_val = model.predict({'FC':x_val_FC, 'EM':x_val_EM}, verbose=2)

print('Validation:')
val_result, val_pred_bin = result_analysis(y_pred_val, y_val)


# predict test dataset result
y_pred_test = model.predict({'FC':x_test_FC, 'EM':x_test_EM}, verbose=2)

print('Test:')
test_result, test_pred_binary = result_analysis(y_pred_test, y_test)

with open('./result/Test_Result_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(test_result, f)


# 保存對Testing data的label情況

# Save model prediction csv
pred_result_df = nrn_pair_test.copy()
pred_result_df['model_pred'] = y_pred_test
pred_result_df['model_pred_binary'] = test_pred_binary

# 将DataFrame存储为csv文件
pred_result_df.to_csv('./result/test_label_'+save_model_name+'.csv', index=False)
print('\nSaved')



# # %% 查詢特定層輸出情況
# fmap1_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac1').output)
# fmap1_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac1').output)

# fmap2_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac2').output)
# fmap2_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac2').output)

# fmap3_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac3').output)
# fmap3_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac3').output)

# fmap4_FC = Model(inputs=model.get_layer('FC').input, outputs=model.get_layer('fc_ac4').output)
# fmap4_EM = Model(inputs=model.get_layer('EM').input, outputs=model.get_layer('em_ac4').output)

# fmap1_test_FC = fmap1_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap1_test_EM = fmap1_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap1_val_FC = fmap1_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap1_val_EM = fmap1_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap1_train_FC = fmap1_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap1_train_EM = fmap1_EM.predict({'EM':x_train_EM}, verbose=2)

# fmap2_test_FC = fmap2_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap2_test_EM = fmap2_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap2_val_FC = fmap2_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap2_val_EM = fmap2_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap2_train_FC = fmap2_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap2_train_EM = fmap2_EM.predict({'EM':x_train_EM}, verbose=2)

# fmap3_test_FC = fmap3_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap3_test_EM = fmap3_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap3_val_FC = fmap3_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap3_val_EM = fmap3_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap3_train_FC = fmap3_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap3_train_EM = fmap3_EM.predict({'EM':x_train_EM}, verbose=2)

# fmap4_test_FC = fmap4_FC.predict({'FC':x_test_FC}, verbose=2)
# fmap4_test_EM = fmap4_EM.predict({'EM':x_test_EM}, verbose=2)
# fmap4_val_FC = fmap4_FC.predict({'FC':x_val_FC}, verbose=2)
# fmap4_val_EM = fmap4_EM.predict({'EM':x_val_EM}, verbose=2)
# # fmap4_train_FC = fmap4_FC.predict({'FC':x_train_FC}, verbose=2)
# # fmap4_train_EM = fmap4_EM.predict({'EM':x_train_EM}, verbose=2)




# def plot_feature(fmap):
#     f_num = fmap.shape[2]
#     while f_num > 0:
#         plt.figure(figsize=(20,5))
#         for i in range(min(4, fmap.shape[2])):
#             plt.subplot(1,4,i+1)
#             plt.imshow(fmap[:,:,-f_num+i], cmap='magma')
#             plt.xticks([])
#             plt.yticks([])
#         plt.show()
#         f_num -= 4


# plot_feature(fmap4_test_FC[3])
# plot_feature(fmap4_test_EM[3])

# %%
