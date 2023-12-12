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

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import pickle

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import load_model
from util import load_pkl
from tqdm import tqdm


num_splits = 2 #0~9
data_range = 'D5'   #D4 or D5


save_model_name  = 'model_D1-' + data_range + '_' +str(num_splits)


# %% 对新数据集进行标注
unlabel_path = './data/mapping_data_0.6-0.7+/'


# 计算label
model = load_model('./Final_Model/' + save_model_name + '.h5')

def annotator(model,fc_img, em_img):
    # 使用transpose()将数组形状从(3, 50, 50)更改为(50, 50, 3)
    fc_img = np.transpose(fc_img, (1, 2, 0))
    em_img = np.transpose(em_img, (1, 2, 0))

    # 将数据维度扩展至4维（符合CNN输入）
    fc_img = np.expand_dims(fc_img, axis=0)
    em_img = np.expand_dims(em_img, axis=0)
    label = model.predict({'FC':fc_img, 'EM':em_img}, verbose=0)
    # binary label
    if label > 0.5:
        label = 1
    else:
        label = 0
    return label



# 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
file_list = [file_name for file_name in os.listdir(unlabel_path) if file_name.endswith('.pkl')]

print('\nLabeling..')
# 遍历母文件夹下的所有条目

# 初始化一个list，用于存储文件名和计算结果
new_data_lst = []


for file_name in tqdm(file_list, total=len(file_list)):
    # 创建完整文件路径
    file_path = os.path.join(unlabel_path, file_name)

    # 读取pkl文件
    data_lst = load_pkl(file_path)
    for data in data_lst:
        # 计算结果
        result = annotator(model, data[3], data[4])

        # 将文件名和计算结果添加到DataFrame
        new_data = {'fc_id': data[0], 'em_id': data[1], 'score': data[2], 'label': result}
        new_data_lst.append(new_data)

# 将新数据转换为DataFrame
label_df = pd.DataFrame(new_data_lst)



# 将DataFrame存储为csv文件
label_df.to_csv('./Iterative_result/label_df_with_'+save_model_name+'.csv', index=False)
print('\nSaved')


# %% Train new model


# load annotator data
label_annotator = pd.read_csv('./Iterative_result/label_df_with_'+save_model_name+'.csv')

add_low_score = True
low_score_neg_rate = 2


seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'


# load train, test
label_table_train = pd.read_csv('./data/train_split_' + str(num_splits) +'_D1-' + data_range + '.csv')
label_table_test = pd.read_csv('./data/test_split_' + str(num_splits) +'_D1-' + data_range + '.csv')




# 删除label_table_train 在annotator中出现的数据
merged = pd.merge(label_annotator, label_table_train, how='left', indicator=True)
label_annotator = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)

# 删除label_table_test 在annotator中出现的数据
merged = pd.merge(label_annotator, label_table_test, how='left', indicator=True)
label_annotator = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)



#针对 label_annotator做 data_preprocess
train_pair_nrn = label_annotator[['fc_id','em_id','label']].to_numpy()


resolutions = [3,50,50]


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

    pair_df = pd.DataFrame({'FC':FC_nrn_lst, 'EM':EM_nrn_lst, 'label':label_lst, 'score':score_lst})    # list of pairs

    return data_np, pair_df

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

    pair_df = pd.DataFrame({'FC':FC_nrn_lst, 'EM':EM_nrn_lst, 'label':label_lst, 'score':score_lst})    # list of pairs

    return data_np, pair_df



data_np_train, nrn_pair_train = data_preprocess_annotator(unlabel_path, train_pair_nrn)

# Train Validation Split
data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.2, random_state=seed)

print('\nTrain data:', len(data_np_train),'\nValid data:', len(data_np_valid))


X_val = data_np_valid
y_val = np.array(nrn_pair_valid['label'],dtype=np.int32)

cross_expand = False

if cross_expand:
    true_label_pair_df = nrn_pair_train.loc[nrn_pair_train['label']==1]
    false_label_pair_df = nrn_pair_train.loc[nrn_pair_train['label']==0]

    def cluster_expansion(pair_df_pos, pair_df_neg):

        label_dict = {}
        for _, row in pair_df_pos.iterrows():
            if row['FC'] in label_dict:
                label_dict[row['FC']].append(row['EM'])
            else:
                label_dict[row['FC']] = [row['EM']]

        # 以FC為key的dictionary建立完成，接下來搜索可擴張項
        label_dict_new = label_dict.copy()
        for k in label_dict.keys():
            for v in label_dict[k]:
                for i in label_dict.keys():
                    if v in label_dict[i] and i != k:
                        label_dict_new[k] = list(set(label_dict[k]+label_dict[i]))

        pair_df_pos_expand = pd.DataFrame({'FC':[], 'EM':[], 'label':[]})
        for k in label_dict_new.keys():
            for v in label_dict_new[k]:
                pair_df_pos_expand = pair_df_pos_expand.append({'FC':k, 'EM':v, 'label':1}, ignore_index=True)


        # label=1資料擴張完畢，現在製造label=0的pair使資料量平衡  
        neg_dict = {}
        for _, row in pair_df_neg.iterrows():
            if row['FC'] in neg_dict:
                neg_dict[row['FC']].append(row['EM'])
            else:
                neg_dict[row['FC']] = [row['EM']]


        #擴張
        neg_dict_new = neg_dict.copy()
        for k in neg_dict.keys():
            for v in neg_dict[k]:
                for i in label_dict_new.keys():
                    if v in label_dict_new[i]:
                        neg_dict_new[k] = list(set(neg_dict_new[k]+label_dict_new[i]))


        # 從字典建立擴張後的 pair df
        pair_df_neg_expand = pd.DataFrame({'FC':[], 'EM':[], 'label':[]})
        for k in neg_dict_new.keys():
            for v in neg_dict_new[k]:
                pair_df_neg_expand = pair_df_neg_expand.append({'FC':k, 'EM':v, 'label':0}, ignore_index=True)


        return pair_df_pos_expand, pair_df_neg_expand

    pair_df_pos_expand, pair_df_neg_expand = cluster_expansion(true_label_pair_df, false_label_pair_df)

    def image_expand(data_np, pair_df_neg_expand, pair_df):
        # 增加 data_np, 更新 label lst
        data_np_add = np.zeros((len(pair_df_neg_expand), data_np.shape[1],data_np.shape[2], data_np.shape[3], data_np.shape[4]))
        for i in range(len(data_np_add)):
            fc_id = pair_df_neg_expand['FC'][i]
            em_id = pair_df_neg_expand['EM'][i]
            for j in range(len(data_np)):
                if pair_df['FC'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['EM'][k] == em_id:
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
            fc_id = pair_df_neg_expand['FC'][i]
            em_id = pair_df_neg_expand['EM'][i]
            for j in range(len(data_np)):
                if pair_df['FC'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['EM'][k] == em_id:
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

del X_train_aug1, X_train_aug2, X_train_add

# 翻倍
X_train_aug3 = np.zeros_like(X_train)

for i in range(X_train_aug3.shape[0]):
    X_train_aug3[i,0,:] = np.flipud(np.rot90(X_train[i,0,:],1))
    X_train_aug3[i,1,:] = np.flipud(np.rot90(X_train[i,1,:],1))

# 判斷x_train是否太大, 否則內存不足
if len(X_train) + len(X_train_aug3) > 100000:
    X_train_aug3 = X_train_aug3[:(100000-len(X_train))]
    y_train_aug3 = y_train[:(100000-len(y_train))]

X_train = np.vstack((X_train, X_train_aug3))
y_train = np.hstack((y_train, y_train_aug3))




del X_train_aug3, y_train_aug3

# FC/EM Split
X_train_FC = X_train[:,0,:]
X_train_EM = X_train[:,1,:]

X_val_FC = X_val[:,0,:]
X_val_EM = X_val[:,1,:]


print('X_train shape:', X_train_FC.shape, X_train_EM.shape)
print('y_train shape:', len(y_train))
print('X_val shape:', X_val_FC.shape, X_val_EM.shape)
print('y_val shape:', len(y_val))








from model import CNN_best

cnn = CNN_best((resolutions[1],resolutions[2],3))


train_epochs = 50
# Scheduler
def scheduler(epoch, lr): 

    min_lr=0.0000001
    total_epoch = train_epochs
    init_lr = 0.001
    epoch_lr = init_lr*((1-epoch/total_epoch)**2)
    if epoch_lr<min_lr:
        epoch_lr = min_lr

    return epoch_lr

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)

# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint('./Iterative_Second_Stage_Model/' + save_model_name + '.h5', verbose=2, monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode="auto")


# Scheduler
second_history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=128, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=train_epochs, shuffle=True, callbacks = [checkpoint, reduce_lr], verbose=2)

# plt.plot(second_history.history['loss'], label='loss')
# plt.plot(second_history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.savefig('./Figure/Second_Train_Curve_' + str(num_splits) + '.png', dpi=100)
# # plt.show()
# plt.close('all')

# cnn_train_loss = history.history['loss']
# cnn_valid_loss = history.history['val_loss']
with open('./Iterative_result/Train_History_Second_stage_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(second_history.history, f)

# %% Testing 1
# 清除内存
del X_train, X_train_FC, X_train_EM, y_train


# 讀神經三視圖資料
map_data_D1toD4 = load_pkl('./data/mapping_data_sn.pkl')
# map_data(lst) 中每一项内容为: 'FC nrn','EM nrn ', Score, FC Array, EM Array

map_data_D5 = load_pkl('./data/mapping_data_sn_D5_old.pkl')

map_data = map_data_D1toD4 + map_data_D5
del map_data_D1toD4, map_data_D5


# turn to numpy array
test_pair_nrn = label_table_test[['fc_id','em_id','label']].to_numpy()


data_np_test, nrn_pair_test = data_preprocess(map_data, test_pair_nrn)
X_test = data_np_test
y_test = np.array(nrn_pair_test['label'],dtype=np.int32)

X_test_FC = X_test[:,0,:]
X_test_EM = X_test[:,1,:]







model = load_model('./Iterative_Second_Stage_Model/' + save_model_name + '.h5')
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

with open('./Iterative_result/Second_stage_Result_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(result, f)





# %% 冻结前层参数
from keras.losses import BinaryFocalCrossentropy

for layer in model.layers[:-5]:
    layer.trainable = False

# # 确认模型状态
# for layer in model.layers:
#     print(layer, layer.trainable)
model.summary()

# 编译模型
model.compile(optimizer='rmsprop', loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=True), metrics = [tf.keras.metrics.BinaryAccuracy(name='Bi-Acc')])



# 准备人工标注的train data
train_pair_nrn = label_table_train[['fc_id','em_id','label']].to_numpy()
data_np_train, nrn_pair_train = data_preprocess(map_data, train_pair_nrn)


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
            if row['FC'] in label_dict:
                label_dict[row['FC']].append(row['EM'])
            else:
                label_dict[row['FC']] = [row['EM']]

        # 以FC為key的dictionary建立完成，接下來搜索可擴張項
        label_dict_new = label_dict.copy()
        for k in label_dict.keys():
            for v in label_dict[k]:
                for i in label_dict.keys():
                    if v in label_dict[i] and i != k:
                        label_dict_new[k] = list(set(label_dict[k]+label_dict[i]))

        pair_df_pos_expand = pd.DataFrame({'FC':[], 'EM':[], 'label':[]})
        for k in label_dict_new.keys():
            for v in label_dict_new[k]:
                pair_df_pos_expand = pair_df_pos_expand.append({'FC':k, 'EM':v, 'label':1}, ignore_index=True)


        # label=1資料擴張完畢，現在製造label=0的pair使資料量平衡  
        neg_dict = {}
        for _, row in pair_df_neg.iterrows():
            if row['FC'] in neg_dict:
                neg_dict[row['FC']].append(row['EM'])
            else:
                neg_dict[row['FC']] = [row['EM']]


        #擴張
        neg_dict_new = neg_dict.copy()
        for k in neg_dict.keys():
            for v in neg_dict[k]:
                for i in label_dict_new.keys():
                    if v in label_dict_new[i]:
                        neg_dict_new[k] = list(set(neg_dict_new[k]+label_dict_new[i]))


        # 從字典建立擴張後的 pair df
        pair_df_neg_expand = pd.DataFrame({'FC':[], 'EM':[], 'label':[]})
        for k in neg_dict_new.keys():
            for v in neg_dict_new[k]:
                pair_df_neg_expand = pair_df_neg_expand.append({'FC':k, 'EM':v, 'label':0}, ignore_index=True)


        return pair_df_pos_expand, pair_df_neg_expand

    pair_df_pos_expand, pair_df_neg_expand = cluster_expansion(true_label_pair_df, false_label_pair_df)

    def image_expand(data_np, pair_df_neg_expand, pair_df):
        # 增加 data_np, 更新 label lst
        data_np_add = np.zeros((len(pair_df_neg_expand), data_np.shape[1],data_np.shape[2], data_np.shape[3], data_np.shape[4]))
        for i in range(len(data_np_add)):
            fc_id = pair_df_neg_expand['FC'][i]
            em_id = pair_df_neg_expand['EM'][i]
            for j in range(len(data_np)):
                if pair_df['FC'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['EM'][k] == em_id:
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
            fc_id = pair_df_neg_expand['FC'][i]
            em_id = pair_df_neg_expand['EM'][i]
            for j in range(len(data_np)):
                if pair_df['FC'][j] == fc_id:
                    data_np_add[i,0,:] = data_np[j,0,:]
                    break
            for k in range(len(data_np)):
                if pair_df['EM'][k] == em_id:
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
    
    nrn_pair_test_np = nrn_pair_test[['FC','EM']].to_numpy()    # 以np格式獲取test中pair的神經名稱
    
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





reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=0)

# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint('./Iterative_Final_Model/' + save_model_name + '.h5', verbose=2, monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode="auto")


# Scheduler
final_history = model.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=128, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=train_epochs, shuffle=True, callbacks = [checkpoint, reduce_lr], verbose=2)

# plt.plot(final_history.history['loss'], label='loss')
# plt.plot(final_history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.savefig('./Figure/Final_Train_Curve_' + str(num_splits) + '.png', dpi=100)
# # plt.show()
# plt.close('all')
with open('./Iterative_result/Train_History_Final_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(final_history.history, f)

# %%
model = load_model('./Iterative_Final_Model/' + save_model_name + '.h5')
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

with open('./Iterative_result/Final_stage_Result_'+save_model_name+'.pkl', 'wb') as f:
    pickle.dump(result, f)

print('Result Saved.')
print('\nProgram Completed')
# %%

# %%