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
# self-labeling pkl path
map_dict_folder = './data/statistical_results/pre_train_map'# pre-train使用的全部三视图位置

initial_lr = 0.001
train_epochs = 300


seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)


save_model_name  = 'pre_train_model_150K'

train_scale = 150000    #儘量偶數，因為要一半pos, 一半neg

# load train, test
label_table_train = pd.read_csv('./preTrain_label/preTrain_label.csv')
# 平衡 neg 和 pos 並優先選擇std小的
neg_idx = label_table_train[label_table_train['label']<0.5].index
pos_idx = label_table_train[label_table_train['label']>=0.5].index

label_table_pos = label_table_train.loc[pos_idx]
label_table_neg = label_table_train.loc[neg_idx]

label_table_pos = label_table_pos.iloc[:int(train_scale/2)]
label_table_neg = label_table_neg.iloc[:int(train_scale/2)]

label_table_train = pd.concat([label_table_pos, label_table_neg], axis=0)

# turn to numpy array
train_pair_nrn = label_table_train[['fc_id','em_id','label']].to_numpy()
#shuffle
idx = np.random.permutation(len(train_pair_nrn))
train_pair_nrn = train_pair_nrn[idx]




# %% data prerpare

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


data_np_train, nrn_pair_train, train_not_found = data_preprocess(map_dict_folder, train_pair_nrn)



# %% Train Validation Split
x_train, x_val, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.15, random_state=seed)
del data_np_train

print('\nOriginal Train data:', len(x_train),'\nValid data:', len(x_val))
y_train = np.array(nrn_pair_train['label'])
y_val = np.array(nrn_pair_valid['label'])


# 画图预览 map data
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
# 交換 FC/EM, enforcing symmetry in the input layer
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

# #翻轉 augment
# X_train_aug1 = np.zeros_like(x_train)
# for i in range(X_train_aug1.shape[0]):
#     X_train_aug1[i,0,:] = np.fliplr(x_train[i,0,:])
#     X_train_aug1[i,1,:] = np.fliplr(x_train[i,1,:])

# x_train = np.vstack((x_train, X_train_aug1))
# y_train = np.hstack((y_train, y_train))

# del X_train_aug1



# FC/EM Split
x_train_FC = x_train[:,0,:]
x_train_EM = x_train[:,1,:]

del x_train

x_val_FC = x_val[:,0,:]
x_val_EM = x_val[:,1,:]

print('x_train shape:', x_train_FC.shape, x_train_EM.shape)
print('y_train shape:', len(y_train))
print('x_val shape:', x_val_FC.shape, x_val_EM.shape)
print('y_val shape:', len(y_val))



# %%

from model import CNN_best, CNN_deep, CNN_shared, CNN_focal, CNN_L2shared
# from tensorflow.keras.utils import plot_model

resolutions = x_train_FC.shape[1:]

cnn = CNN_shared((resolutions[0],resolutions[1],resolutions[2]))
# cnn = CNN_deep((resolutions[0],resolutions[1],resolutions[2]))

cnn.compile(optimizer=AdamW(learning_rate=initial_lr), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[BinaryAccuracy(name='Bi-Acc')])


# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint('./preTrain_Model/' + save_model_name + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')



# Model.fit
Annotator_history = cnn.fit({'FC':x_train_FC, 'EM':x_train_EM}, 
                            y_train, 
                            validation_data=({'FC':x_val_FC, 'EM':x_val_EM}, y_val), 
                            epochs=train_epochs, 
                            shuffle=True, 
                            callbacks = [checkpoint], verbose=2)
                            # class_weight=class_weights)



plt.plot(Annotator_history.history['loss'], label='loss')
plt.plot(Annotator_history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('./Figure/'+save_model_name+'_train_curve.png', dpi=150, bbox_inches="tight")
plt.show()
plt.close('all')

# cnn_train_loss = history.history['loss']
# cnn_valid_loss = history.history['val_loss']

# Save history to file
with open('./result/'+save_model_name+'_train_history.pkl', 'wb') as f:
    pickle.dump(Annotator_history.history, f)


# %%
model = load_model('./preTrain_Model/' + save_model_name + '.h5')

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

