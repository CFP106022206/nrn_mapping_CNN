'''
Data_process_Train.py 的使用信心度版本

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
num_splits = 4 #0~9, or 99 for whole nBLAST testing' 

'''
使用冠廷的檔案寫法，因冠廷的檔案全部混在同一個黃瓜中.
新寫法是使用和data_preprocess_annotator 相同的方法。
'''
use_KT_map = True
grid75_path = './data/D1-D5_grid75_sn'


scheduler_exp = 1.5      #學習率調度器的約束力指數，越小約束越強
initial_lr = 0.00005
train_epochs = 50

add_low_score = False
low_score_neg_rate = 2

seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)


save_model_name  = 'Annotator_D1-D6_' +str(num_splits)

# load train, test
label_table_train = pd.read_csv('./data/train_split_' + str(num_splits) +'_D1-D6.csv')
label_table_test = pd.read_csv('./data/test_split_' + str(num_splits) +'_D1-D6.csv')


# turn to numpy array
test_pair_nrn = label_table_test[['fc_id','em_id','confidence']].to_numpy()
train_pair_nrn = label_table_train[['fc_id','em_id','confidence']].to_numpy()





# %% data prerpare

'''
使用冠廷的檔案寫法，因冠廷的檔案全部混在同一個黃瓜中.
新寫法是使用和data_preprocess_annotator 相同的方法。
'''
if use_KT_map:
    # 讀神經三視圖資料
    map_data_D1toD4 = load_pkl('./data/mapping_data_sn.pkl')
    # map_data(lst) 中每一项内容为: 'FC nrn','EM nrn ', Score, FC Array, EM Array

    map_data_D5 = load_pkl('./data/mapping_data_sn_D5_old.pkl')

    map_data = map_data_D1toD4 + map_data_D5
    del map_data_D1toD4, map_data_D5

    resolutions = map_data[0][3].shape
    print('Image shape: ', resolutions)

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
            print('Total:', len(not_found_data))


        # Normalization : x' = x - min(x) / max(x) - min(x)
        data_np = (data_np - np.min(data_np))/(np.max(data_np) - np.min(data_np))

        pair_df = pd.DataFrame({'fc_id':FC_nrn_lst, 'em_id':EM_nrn_lst, 'label':label_lst, 'score':score_lst})    # list of pairs

        return data_np, pair_df

    def data_preprocess_folder(file_path, pair_nrn):

            print('\nCollecting 3-View Data Numpy Array..')

            data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
            FC_nrn_lst, EM_nrn_lst, score_lst, label_lst = [], [], [], []


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
                print('Total:', len(not_found_data))


            # Normalization : x' = x - min(x) / max(x) - min(x)
            data_np = (data_np - np.min(data_np))/(np.max(data_np) - np.min(data_np))

            pair_df = pd.DataFrame({'fc_id':FC_nrn_lst, 'em_id':EM_nrn_lst, 'label':label_lst, 'score':score_lst})    # list of pairs

            return data_np, pair_df

    data_np_test, nrn_pair_test = data_preprocess(map_data, test_pair_nrn)
    data_np_train, nrn_pair_train = data_preprocess(map_data, train_pair_nrn)


    # data_np_test, nrn_pair_test = data_preprocess_folder('./data/all_pkl', test_pair_nrn)
    # data_np_train, nrn_pair_train = data_preprocess_folder('./data/all_pkl', train_pair_nrn)



else:
    resolutions = (3, 75, 75)

    def data_preprocess(file_path, pair_nrn):

        print('\nCollecting 3-View Data Numpy Array..')

        data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
        FC_nrn_lst, EM_nrn_lst, score_lst, label_lst = [], [], [], []


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


    data_np_test, nrn_pair_test = data_preprocess(grid75_path, test_pair_nrn)
    data_np_train, nrn_pair_train = data_preprocess(grid75_path, train_pair_nrn)




# Train Validation Split
data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.1, random_state=seed)

print('\nTrain data:', len(data_np_train),'\nValid data:', len(data_np_valid),'\nTest data:', len(data_np_test))


X_val = data_np_valid
X_test = data_np_test
y_val = np.array(nrn_pair_valid['label'])
y_test = np.array(nrn_pair_test['label'])

X_train = data_np_train
y_train = np.array(nrn_pair_train['label'])





# %% 找出各個信心等級的數量

def counter(y_train, generate=True):
    index_dict = {} # 保存各種類label與其對應的數據索引位置

    for i, num in enumerate(y_train):
        if num not in index_dict:
            index_dict[num] = [i]
        else:
            index_dict[num].append(i)

    y_train_label, label_num = [], []
    for k, v in index_dict.items():
        y_train_label.append(k)
        label_num.append(len(v))

    # 將兩個列表一起打包成元組，並排序
    sorted_pairs = sorted(zip(y_train_label, label_num))

    # 使用zip *操作符將打包的元組解開，得到兩個新的已排序列表
    y_train_label, label_num = zip(*sorted_pairs)

    print('Label', y_train_label)
    print('Number', label_num)
    
    if generate == True:
        return index_dict, y_train_label, label_num


print('Before Balancing:')
index_dict, y_train_label, label_num = counter(y_train, generate=True)

# 平衡個類別數據
augment_num = [(np.max(label_num) - num) for num in label_num]

# 旋轉產生擴增數據
for conf_idx, num in enumerate(augment_num):
    confidence = y_train_label[conf_idx]
    X_train_add = np.zeros((num, X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])) # 製作需要增加的x_train 量
    y_train_add = np.array([confidence] * num)

    x_add_idx = index_dict[confidence]  # 呼叫欲增加資料的對應索引位置

    k = 0
    for i in range(num):
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

print('After Balancing:')
counter(y_train, generate=False)







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

        X_augmented.append(rotate_pair)

        y_augmented.append(y_train[i])

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

del X_train_aug1

# 翻倍
X_train_aug2 = np.zeros_like(X_train)

for i in range(X_train_aug2.shape[0]):
    X_train_aug2[i,0,:] = np.flipud(X_train[i,0,:])
    X_train_aug2[i,1,:] = np.flipud(X_train[i,1,:])

X_train = np.vstack((X_train, X_train_aug2))
y_train = np.hstack((y_train, y_train))

del X_train_aug2

# 翻倍
X_train_aug3 = np.zeros_like(X_train)

for i in range(X_train_aug3.shape[0]):
    X_train_aug3[i,0,:] = np.flipud(np.rot90(X_train[i,0,:],1))
    X_train_aug3[i,1,:] = np.flipud(np.rot90(X_train[i,1,:],1))

X_train = np.vstack((X_train, X_train_aug3))
y_train = np.hstack((y_train, y_train))

del X_train_aug3


# FC/EM Split
X_train_FC = X_train[:,0,:]
X_train_EM = X_train[:,1,:]

del X_train

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

# 將信心度label轉換成one-hot標籤
def label_to_onehot(label_lst):
    onehot_lst = []
    label_class = np.unique(label_lst)   #信心度共有幾種 [0,0.3,0.6,0.9]
    for label in label_lst:
        onehot_label = [0]*len(label_class) #[0,0,0,0]
        idx = int(np.where(label_class == label)[0])
        onehot_label[idx] = 1               #[0,1,0,0] if label=0.3
        onehot_lst.append(onehot_label)
    
    return np.array(onehot_lst)

y_train = label_to_onehot(y_train)
y_val = label_to_onehot(y_val)

# %%

def CNN_shared(input_size=(50, 50, 3), output_size=4):
    inputs = [Input(shape=input_size, name="EM"), Input(shape=input_size, name="FC")]

    # 定义共享卷积层和池化层
    shared_conv1 = Conv2D(16, (3, 3), name="Shared_Conv1")
    shared_bn1 = BatchNormalization(name="Shared_BN1")
    shared_act1 = Activation("relu", name="Shared_Activation1")
    shared_conv2 = Conv2D(32, (3, 3), name="Shared_Conv2")
    shared_bn2 = BatchNormalization(name='Shared_BN2')
    shared_act2 = Activation("relu", name='Shared_Activation2')
    shared_pool1 = MaxPool2D(pool_size=(2, 2), name='Shared_pool1')
    
    shared_conv3 = Conv2D(48, (3, 3), name='Shared_Conv3')
    shared_bn3 = BatchNormalization(name='Shared_BN3')
    shared_act3 = Activation("relu", name='Shared_Activation3')
    shared_conv4 = Conv2D(64, (3, 3), name='Shared_Conv4')
    shared_bn4 = BatchNormalization(name='Shared_BN4')
    shared_act4 = Activation("relu", name='Shared_Activation4')
    shared_pool2 = MaxPool2D(pool_size=(2, 2), name='Shared_pool2')

    flattened_layers = []
    for input in inputs:
        conv_layer = shared_conv1(input)
        conv_layer = shared_bn1(conv_layer)
        conv_layer = shared_act1(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2, 2))(conv_layer)

        conv_layer = shared_conv2(conv_layer)
        conv_layer = shared_bn2(conv_layer)
        conv_layer = shared_act2(conv_layer)
        conv_layer = shared_pool1(conv_layer)

        conv_layer = shared_conv3(conv_layer)
        conv_layer = shared_bn3(conv_layer)
        conv_layer = shared_act3(conv_layer)

        conv_layer = shared_conv4(conv_layer)
        conv_layer = shared_bn4(conv_layer)
        conv_layer = shared_act4(conv_layer)
        conv_layer = shared_pool2(conv_layer)

        flattened_layers.append(Flatten()(conv_layer))

    concat_layer = concatenate(flattened_layers, axis=1)

    output = Dropout(0.5)(concat_layer)
    output = Dense(256)(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    # output = Dropout(0.2)(output)

    output = Dense(output_size, activation="softmax")(output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    model.summary()
    return model




cnn = CNN_shared(input_size=(50,50,3), output_size=len(y_train_label))




if not scheduler_exp:
    cnn.compile(optimizer=Adam(learning_rate=initial_lr), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[BinaryAccuracy(name='Bi-Acc')])

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

Annotator_history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, 
                            y_train, 
                            batch_size=128, 
                            validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), 
                            epochs=train_epochs, 
                            shuffle=True, 
                            callbacks = callbacks, verbose=2)
                            # class_weight=class_weights)



# plt.plot(Annotator_history.history['loss'], label='loss')
# plt.plot(Annotator_history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.savefig('./Figure/Annotator_Train_Curve_'+str(num_splits)+'.png', dpi=150, bbox_inches="tight")
# plt.show()
# plt.close('all')

# cnn_train_loss = history.history['loss']
# cnn_valid_loss = history.history['val_loss']

# Save history to file
with open('./result/Train_History_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(Annotator_history.history, f)


model = load_model('./Annotator_Model/' + save_model_name + '.h5')
y_pred = model.predict({'FC':X_test_FC, 'EM':X_test_EM}, verbose=2)


# 將one-hot 轉回信心度標籤, 需要額外提供信心度列表[0,0.3,0.6,0.9]
def onehot_to_label(onehot_lst, label_class):
    label_lst = []
    for onehot in onehot_lst:
        confidence = 0
        # 使用期望值方式計算模型輸出信心度
        for i, probability in enumerate(onehot):
            confidence += probability * label_class[i]

        label_lst.append(confidence)

    return np.array(label_lst)

y_pred = onehot_to_label(y_pred, y_train_label)

# 二分類化以計算 confusion matrix
def continuous_to_binary(y_pred):
    y_pred_binary = []
    for ans in y_pred:
        if ans >= 0.5:
            y_pred_binary.append(1)
        else:
            y_pred_binary.append(0)
    return y_pred_binary

y_pred_binary = continuous_to_binary(y_pred)
y_test_binary = continuous_to_binary(y_test)
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary, labels=[1,0])# 統一標籤格式

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
result_f1_score = f1_score(y_test_binary, y_pred_binary, average=None)
print('F1 Score for Neg:', result_f1_score[0])
print('F1 Score for Pos:', result_f1_score[1])

# save results
result = {'conf_matrix': conf_matrix, 'Precision': precision, 'Recall': recall, 'F1_pos':result_f1_score[1]}

with open('./result/Test_Result_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(result, f)


# 保存對Testing data的label情況

# Save model prediction csv
pred_result_df = nrn_pair_test.copy()
pred_result_df['model_pred'] = y_pred
pred_result_df['model_pred_binary'] = y_pred_binary

# 将DataFrame存储为csv文件
pred_result_df.to_csv('./result/test_label_'+save_model_name+'.csv', index=False)
print('\nSaved')


# %% 对新数据集进行标注
# unlabel_path = './data/all_pkl'

# # 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
# file_list = [file_name for file_name in os.listdir(unlabel_path) if file_name.endswith('.pkl')]

# # 计算label
# model = load_model('./Annotator_Model/' + save_model_name + '.h5')

# def annotator(model,fc_img, em_img):
#     # 使用transpose()将数组形状从(3, 50, 50)更改为(50, 50, 3)
#     fc_img = np.transpose(fc_img, (1, 2, 0))
#     em_img = np.transpose(em_img, (1, 2, 0))

#     # 将数据维度扩展至4维（符合CNN输入）
#     fc_img = np.expand_dims(fc_img, axis=0)
#     em_img = np.expand_dims(em_img, axis=0)
#     label = model.predict({'FC':fc_img, 'EM':em_img}, verbose=0)

#     label = label.flatten()[0]  #因為模型輸出是一個 numpy array

#     # binary label
#     if label > 0.5:
#         label_b = 1
#     else:
#         label_b = 0
#     return label, label_b



# # 初始化一个空lst，用于存储文件名和计算结果
# new_data_lst = []

# print('\nLabeling..')
# # 遍历母文件夹下的所有条目
# for file_name in tqdm(file_list, total=len(file_list)):
# # for file_name in file_list:     # No tqdm version

#     # 创建完整文件路径
#     file_path = os.path.join(unlabel_path, file_name)

#     # 读取pkl文件
#     data_lst = load_pkl(file_path)
#     for data in data_lst:
#         # 计算结果
#         result, result_b = annotator(model, data[3], data[4])

#         # 将文件名和计算结果添加到DataFrame
#         new_data = {'fc_id': data[0], 'em_id': data[1], 'score': data[2], 'label_c': result, 'label': result_b}

#         new_data_lst.append(new_data)

# label_df = pd.DataFrame(new_data_lst)

# # 将DataFrame存储为csv文件
# label_df.to_csv('./result/label_df_with_'+save_model_name+'.csv', index=False)
# print('\nSaved')
print('Program Completed.')

# %%
