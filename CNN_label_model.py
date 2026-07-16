# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import tensorflow as tf
import seaborn as sns

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score
from util import load_pkl

# %% 此档案目的为 load 最佳参数模型, 然后对一帆那边画出的未标注三视图进行标注


seed = 10                       # Random Seed

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'
# %% Data Prepare

train_range_to = 'D5'   # 'D4' or 'D5'
save_model_name = 'Train_Test_In_D1-D5'

add_low_score = True
low_score_neg_rate = 2

use_focal_loss = True

# Load labeled csv
label_csv_D1 = './data/D1_20221230.csv'
label_csv_D2 = './data/D2_20221230.csv'
label_csv_D3 = './data/D3_20221230.csv'
label_csv_D4 = './data/D4_20221230.csv'
label_csv_D5 = './data/D5_20221230.csv'


D1 = pd.read_csv(label_csv_D1)     # FC, EM, label
D1.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D2 = pd.read_csv(label_csv_D2)     # FC, EM, label
D2.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D3 = pd.read_csv(label_csv_D3)     # FC, EM, label
D3.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

D4 = pd.read_csv(label_csv_D4)     # FC, EM, label
D4.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复


if train_range_to == 'D5':
    D5 = pd.read_csv(label_csv_D5)     # FC, EM, label
    D5.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

    label_table_all = pd.concat([D1, D2, D3, D4, D5])   # fc_id, em_id, score, rank, label

    # 讀神經三視圖資料
    map_data_D1toD4 = load_pkl('./data/mapping_data_sn.pkl')
    # map_data(lst) 中每一项内容为: 'FC nrn','EM nrn ', Score, FC Array, EM Array
    map_data_D5 = load_pkl('./data/mapping_data_sn_D5_old.pkl')
    map_data = map_data_D1toD4 + map_data_D5
    del map_data_D1toD4, map_data_D5
else:
    label_table_all = pd.concat([D1, D2, D3, D4])   # fc_id, em_id, score, rank, label
    
    map_data = load_pkl('./data/mapping_data_sn.pkl')


label_table_all.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

# shuffle label_table_all 
label_table_all = label_table_all.sample(frac=1, random_state=seed).reset_index(drop=True)

# 将 label_table_all 分成两份: train and test
test_ratio = 0.2
test_size = int(label_table_all.shape[0]*test_ratio)

label_table_test = label_table_all.iloc[:test_size]
label_table_train = label_table_all.iloc[test_size:]

# turn to numpy array
test_pair_nrn = label_table_test[['fc_id','em_id','label']].to_numpy()
train_pair_nrn = label_table_train[['fc_id','em_id','label']].to_numpy()

# save as csv
label_table_test.to_csv('./data/label_table_test.csv', index=False)
label_table_train.to_csv('./data/label_table_train.csv', index=False)


# %% train model 1 with best parameters(or load one from tuner)

# data prerpare
def data_preprocess(map_data, pair_nrn):
    data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
    FC_nrn_lst, EM_nrn_lst, score_lst, label_lst = [], [], [], []

    for i, row in enumerate(pair_nrn):
        for data in map_data:
            if str(row[0]) == str(data[0]) and str(row[1]) == str(data[1]):              
                FC_nrn_lst.append(str(data[0]))
                EM_nrn_lst.append(str(data[1]))
                score_lst.append(data[2]) 
                label_lst.append(row[2])
                for k in range(3):
                    data_np[i, 0, :, :, k] = data[3][k] # FC Image
                    data_np[i, 1, :, :, k] = data[4][k] # EM Image
                break

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


resolutions = map_data[0][3].shape
print('Image shape: ', resolutions)

data_np_test, nrn_pair_test = data_preprocess(map_data, test_pair_nrn)
data_np_train, nrn_pair_train = data_preprocess(map_data, train_pair_nrn)

# Train Validation Split
data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=0.2, random_state=seed)

print('\nTrain data:', len(data_np_train),'\nValid data:', len(data_np_valid),'\nTest data:', len(data_np_test))


X_val = data_np_valid
X_test = data_np_test
y_val = np.array(nrn_pair_valid['label'],dtype=np.int32)
y_test = np.array(nrn_pair_test['label'],dtype=np.int32)


# %% See the image

# # print_img = 4
# # def See_img(img_data, label_lst, img_idx):
# #     print('Show image index:', img_idx)
# #     print('Label: ', label_lst[img_idx])
# #     print('FC\nEM')
# #     plt.figure(figsize=(16,10))
# #     for i in range(3):
# #         plt.subplot(2,3,i+1)
# #         plt.imshow(img_data[img_idx,0,:,:,i])    # FC img
# #         plt.subplot(2,3,i+4)
# #         plt.imshow(img_data[img_idx,1,:,:,i])    # EM img
# #     plt.show()

# def Plot_NRN_Img(data_np, pair_df, idx):
#     plt.figure(figsize=(20,10))
#     for i in range(3):
#         plt.subplot(2,4,i+1)
#         plt.imshow(data_np[idx,0,:,:,i])    # FC img
#         plt.title('FC: ' + pair_df.iloc[idx][0] + '  Label: '+ str(pair_df.iloc[idx][2]))
#         plt.colorbar()
#         plt.subplot(2,4,i+5)
#         plt.imshow(data_np[idx,1,:,:,i])    # EM img
#         plt.title('EM: ' + pair_df.iloc[idx][1] + '  Label: '+ str(pair_df.iloc[idx][2]))
#         plt.colorbar()
    
#     plt.subplot(2,4,4)
#     plt.imshow(data_np[idx,0,:,:])  # FC
#     plt.title('FC: '+ pair_df.iloc[idx][0] + '     3 Channels')
#     plt.subplot(2,4,8)
#     plt.imshow(data_np[idx,1,:,:])  # EM
#     plt.title('EM: '+ pair_df.iloc[idx][1] + '     3 Channels')
#     plt.savefig('/home/ming/Project/Neural_Mapping_ZGT/Figure/SN/'+pair_df.iloc[idx][0]+'_'+pair_df.iloc[idx][1]+'.png', dpi=200)


# # See_img(data_np, label_lst, print_img)
# for i in range(len(data_np_train)):
#     Plot_NRN_Img(data_np_train, nrn_pair_train, i)
# for i in range(len(X_val)):
#     Plot_NRN_Img(X_val, nrn_pair_valid, i)
# for i in range(len(X_test)):
#     Plot_NRN_Img(X_test, nrn_pair_test, i)

# %%
#   Data Augmentation: 
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





# %%    All train data augmentation
# 翻倍
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



# X_train_add = np.zeros(((X_train.shape[0]-pos),X_train.shape[1],X_train.shape[2],X_train.shape[3], X_train.shape[4]))

# # 篩選出 label為1的 X_train
# true_label_idx = []
# for i in range(y_train.shape[0]):
#     if y_train[i] == 1:
#         true_label_idx.append(i)

# k=0
# for i in range(neg):
#     X_train_add[i] = X_train[true_label_idx[k]]

#     if k >= len(true_label_idx)-1:
#         k=0
#     else:
#         k+=1
# y_train_add = np.ones(X_train_add.shape[0])

# X_train = np.vstack((X_train, X_train_add))
# y_train = np.hstack((y_train, y_train_add))

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


# %%
from model import CNN_best

cnn = CNN_best((resolutions[1],resolutions[2],3))


train_epochs = 100
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
checkpoint = ModelCheckpoint('./CNN_best_' + save_model_name + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode="auto")


# Scheduler
history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=128, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=train_epochs, shuffle=True, callbacks = [checkpoint, reduce_lr])

# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.savefig('/home/ming/Project/nrn_mapping_package-master/Figure/Train_Curve.png', dpi=100)
# plt.show()
# plt.close('all')

cnn_train_loss = history.history['loss']
cnn_valid_loss = history.history['val_loss']

model = load_model('./CNN_best_' + save_model_name + '.h5')
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
print("Precision:", conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0]))
print("Recall:", conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[0,1]))

# F1 Score
result_f1_score = f1_score(y_test, y_pred_binary, average=None)
print('F1 Score for Neg:', result_f1_score[0])
print('F1 Score for Pos:', result_f1_score[1])




# # %% 对新数据集进行标注
# unlabel_path = './data/mapping_data_0.7/'

# # 将文件夹下文件名存入列表
# file_list = os.listdir(unlabel_path)
# # 计算label
# model = load_model('./CNN_best_' + save_model_name + '.h5')

# def annotator(model,fc_img, em_img):
#     # 使用transpose()将数组形状从(3, 50, 50)更改为(50, 50, 3)
#     fc_img = np.transpose(fc_img, (1, 2, 0))
#     em_img = np.transpose(em_img, (1, 2, 0))

#     # 将数据维度扩展至4维（符合CNN输入）
#     fc_img = np.expand_dims(fc_img, axis=0)
#     em_img = np.expand_dims(em_img, axis=0)
#     label = model.predict({'FC':fc_img, 'EM':em_img})
#     # binary label
#     if label > 0.5:
#         label = 1
#     else:
#         label = 0
#     return label



# # 初始化一个空的DataFrame，用于存储文件名和计算结果
# label_df = pd.DataFrame(columns=['fc_id', 'em_id', 'score', 'label'])

# # 遍历母文件夹下的所有条目
# for file_name in (file_list):
#     # 创建完整文件路径
#     file_path = os.path.join(unlabel_path, file_name)

#     # 检查是否为.pkl档案
#     if file_path.endswith('.pkl'):
#         # 读取pkl文件
#         data_lst = load_pkl(file_path)
#         for data in data_lst:
#             # 计算结果
#             result = annotator(model, data[3], data[4])

#             # 将文件名和计算结果添加到DataFrame
#             label_df = label_df.append({'fc_id': data[0], 'em_id': data[1], 'score': data[2], 'label':result}, ignore_index=True)

# # 将DataFrame存储为csv文件
# label_df.to_csv('./data/label_df_0.7.csv', index=False)