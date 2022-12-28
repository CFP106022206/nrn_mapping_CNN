# from multiprocessing import pool
# %%
from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import random
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score
from util import load_pkl


seed = 17                       # Random Seed
# csv_threshold = [0.6, 0.7, 0.8] # selected labeled csv's threshold
# map_data_type = 'rsn'           # 三视图的加权种类选择


os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DITERMINISTIC_OPS'] = '1'

# %% Data Preprocessing 1 -------------------------------------------------------------------------------------------

use_D2_only = False
add_low_score = True
low_score_neg_rate = 2

test_in_all_data = True

use_focal_loss = True

# Load labeled csv
label_csv_all = '/home/ming/Project/Neural_Mapping_ZGT/data/D1-D5.csv'
label_csv_D2 = '/home/ming/Project/Neural_Mapping_ZGT/data/D2_data_1013.csv'
label_table_all = pd.read_csv(label_csv_all)   # fc_id, em_id, score, rank, label
label_table_D2 = pd.read_csv(label_csv_D2)     # FC, EM, label

# D2_data_1013 label 錯誤修正
relabel_lst = []
for i in range(len(label_table_D2)):
    fc_id = label_table_D2['FC'][i]
    em_id = label_table_D2['EM'][i]
    old_label = label_table_D2['label'][i]
    for j in range(len(label_table_all)):
        if fc_id == label_table_all['fc_id'][j] and em_id == label_table_all['em_id'][j]:
            relabel_lst.append(label_table_all['label'][j])
            break

        if j == len(label_table_all)-1:
            relabel_lst.append(old_label)   #找不到新label

label_table_D2['label'] = relabel_lst


if test_in_all_data:
    use_D2_only = False
    #Testing data 從全部裡面挑選10%
    train_pair_nrn, test_pair_nrn = train_test_split(label_table_all[['fc_id','em_id','label']].to_numpy(), test_size=0.1, random_state=seed)

else:
    #Testing data 從D2裡面挑選20%
    train_pair_nrn, test_pair_nrn = train_test_split(label_table_D2.to_numpy(), test_size=0.2, random_state=seed)


if not use_D2_only:
    if not test_in_all_data:
        # train data 使用D1～D4
        train_pair_nrn = label_table_all[['fc_id','em_id','label']].to_numpy()      # 如果不是test in all data, train pair就要重做

        # 將D1~D4中 test data 的部分刪除
        test_data_row = []
        for i, row in enumerate(train_pair_nrn):
            fc_id = str(row[0])
            em_id = str(row[1])
            for j in test_pair_nrn:
                if fc_id == str(j[0]) and em_id == str(j[1]):
                    test_data_row.append(i)
                    break
        train_pair_nrn = np.delete(train_pair_nrn, test_data_row, axis=0)

    # 讀神經三視圖資料
    map_data = load_pkl('/home/ming/Project/nrn_mapping_package-master/data/statistical_results/mapping_data_sn.pkl')
    # map_data(lst) 中每一项内容为: 'FC nrn','EM nrn ', Score, FC Array, EM Array



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

else:
    map_data = load_pkl('/home/ming/Project/nrn_mapping_package-master/data/statistical_results/res_sn.pkl')
    # map_data(lst) 中每一项内容为: 'FC nrn','EM nrn ', FC Array, EM Array, Score



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
                        data_np[i, 0, :, :, k] = data[2][k] # FC Image
                        data_np[i, 1, :, :, k] = data[3][k] # EM Image
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

        pair_df = pd.DataFrame({'FC':FC_nrn_lst, 'EM':EM_nrn_lst, 'label':label_lst})    # list of pairs

        return data_np, pair_df


resolutions = map_data[0][3].shape
print('Image shape: ', resolutions)

data_np_test, nrn_pair_test = data_preprocess(map_data, test_pair_nrn)
data_np_train, nrn_pair_train = data_preprocess(map_data, train_pair_nrn)

# Train Validation Split
test_ratio = 0.1
data_np_train, data_np_valid, nrn_pair_train, nrn_pair_valid = train_test_split(data_np_train, nrn_pair_train, test_size=test_ratio, random_state=seed)

print('\nTrain data:', len(data_np_train),'\nValid data:', len(data_np_valid),'\nTest data:', len(data_np_test))


# 讀取腦科重標記label的csv檔案(因為後續腦科那邊可能會更改原本的label)

# def relabel(pair_df, path='/home/ming/Project/Neural_Mapping_ZGT/data/D1-D4.csv'):
#     relabel_table = pd.read_csv(path)

#     #執行label修正
#     relabel_fc, relabel_em, old_label, new_label = [],[],[],[]
#     for i in range(pair_df.shape[0]):
#         fc_id = str(pair_df.iloc[i,0])      #'FC'
#         em_id = str(pair_df.iloc[i,1])      #'EM'
#         label_o = str(pair_df.iloc[i,2])  #'label'
#         for j in range(relabel_table.shape[0]):
#             new_inform = relabel_table.iloc[j]
#             if str(new_inform['fc_id']) == fc_id and str(new_inform['em_id']) == em_id and str(new_inform['label']) != label_o:
#                 pair_df.iloc[i,2] = new_inform['label']
#                 relabel_fc.append(fc_id)
#                 relabel_em.append(em_id)
#                 old_label.append(label_o)
#                 new_label.append(str(new_inform['label']))


#     relabel_df = pd.DataFrame({'rFC':relabel_fc, 'rEM':relabel_em, 'old_l':old_label, 'new_l':new_label})
#     label_lst = pair_df.iloc[:,2]

#     return pair_df, relabel_df

# nrn_pair_train, relabel_train = relabel(nrn_pair_train)
# nrn_pair_valid, relabel_valid = relabel(nrn_pair_valid)
# nrn_pair_test, relabel_test = relabel(nrn_pair_test)

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
    
    k=0
    for i in range(X_train_add.shape[0]):
        for j in range(k, len(map_data)):
            if map_data[j][2] < 0.5:
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


# 保存 training validation testing data 為 pickle
# RAM 不夠，保存結束關閉kernel 重啟，load 回 data

# model_train_data = {'FC':X_train_FC, 'EM':X_train_EM, 'y': y_train}
# model_valid_data = {'FC':X_val_FC, 'EM':X_val_EM, 'y': y_val}
# model_test_data = {'FC':X_test_FC, 'EM': X_test_EM, 'y': y_test}

# model_data = [model_train_data, model_valid_data, model_test_data, class_weights]

# with open('/home/ming/Project/Neural_Mapping_ZGT/data/model_data.pkl', 'wb') as f:
#     pickle.dump(model_data, f)

# %%
from model import CNN_small, CNN_focal, CNN_big

if use_focal_loss:
    cnn = CNN_focal((50,50,3))
else:
    cnn = CNN_small((50, 50, 3))

plot_model(cnn, '/home/ming/Project/nrn_mapping_package-master/Figure/cnn_small_structure.png', show_shapes=True)

# %% Load pkl data
# train_data = load_pkl('/home/ming/Project/Neural_Mapping_ZGT/data/training.pkl')
# valid_data = load_pkl('/home/ming/Project/Neural_Mapping_ZGT/data/valid.pkl')
# test_data = load_pkl('/home/ming/Project/Neural_Mapping_ZGT/data/test.pkl')
# class_weights = [0.52363299, 11.07843137]

train_epochs = 150
# Scheduler
def scheduler(epoch, lr): 

    min_lr=0.0000001
    total_epoch = train_epochs
    init_lr = 0.001
    epoch_lr = init_lr*((1-epoch/total_epoch)**0.9)
    if epoch_lr<min_lr:
        epoch_lr = min_lr

    return epoch_lr

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)

# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint('/home/ming/Project/nrn_mapping_package-master/CNN_small_Checkpoint.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode="auto")


# Scheduler
history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=128, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=train_epochs, shuffle=True, callbacks = [checkpoint, reduce_lr])


# No Scheduler
# history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=128, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=train_epochs, shuffle=True, callbacks = [checkpoint])    # No Schedular


#  No Callback
# history = cnn.fit({'FC':train_data['FC'], 'EM':train_data['EM']}, train_data['y'], batch_size=32, validation_data=({'FC':valid_data['FC'], 'EM':valid_data['EM']}, valid_data['y']), class_weight=class_weights, epochs=50, shuffle=True)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('/home/ming/Project/nrn_mapping_package-master/Figure/Train_Curve.png', dpi=100)
plt.show()

cnn_small_train_loss = history.history['loss']
cnn_small_valid_loss = history.history['val_loss']
# %%
model = load_model('/home/ming/Project/nrn_mapping_package-master/CNN_small_Checkpoint.h5')
y_pred = model.predict({'FC':X_test_FC, 'EM':X_test_EM})
pred_test_compare = np.hstack((y_pred, y_test.reshape(len(y_test), 1)))
y_pred_binary = []
for ans in y_pred:
    if ans >= 0.5:
        y_pred_binary.append(1)
    else:
        y_pred_binary.append(0)

conf_matrix = confusion_matrix(y_test.tolist(), y_pred_binary, labels=[1,0])# 統一標籤格式


# 可视化混淆矩阵
# group_names = ['True Neg','False Pos','False Neg','True Pos']
group_names = ['True Pos','False Neg','False Pos','True Neg']
group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
group_percent_false = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()[:2]/np.sum(conf_matrix[0])]
group_percent_true = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()[2:]/np.sum(conf_matrix[1])]
group_percent = group_percent_false + group_percent_true
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percent)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Purples')
plt.savefig('/home/ming/Project/nrn_mapping_package-master/Figure/ConfuseMatric.png', dpi=100)

# Precision and recall 
print("Precision:", conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0]))
print("Recall:", conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[0,1]))
# F1 Score
result_f1_score = f1_score(y_test, y_pred_binary, average=None)
print('F1 Score for Neg:', result_f1_score[0])
print('F1 Score for Pos:', result_f1_score[1])

# %% ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

fig = plt.figure()
lw = 2
plt.plot(fpr, tpr, '-o', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#fig.savefig('/tmp/roc.png')
plt.show()

# %% Precision with different threshold
# def plot_conf_matrix(y_test, y_pred_binary, threshold):
#     conf_matrix = confusion_matrix(y_test, y_pred_binary, labels=[1,0])
#     # 可视化混淆矩阵
#     # group_names = ['True Neg','False Pos','False Neg','True Pos']
#     group_names = ['True Pos','False Neg','False Pos','True Neg']
#     group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
#     group_percent_false = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()[:2]/np.sum(conf_matrix[0])]
#     group_percent_true = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()[2:]/np.sum(conf_matrix[1])]
#     group_percent = group_percent_false + group_percent_true
#     labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percent)]
#     labels = np.asarray(labels).reshape(2,2)
#     sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Purples')
#     plt.title('Threshold = ' + str(np.round(threshold,1)))
#     plt.savefig('/home/ming/Project/nrn_mapping_package-master/Figure/ConfuseMatric_tr_'+str(threshold)+'.png', dpi=100)
#     plt.show()

# def binary_label(y_pred, threshold):
#     y_binary = y_pred.copy()
#     for i in range(len(y_pred)):
#         if y_pred[i] < threshold:
#             y_binary[i] = 0
#         else:
#             y_binary[i] = 1
#     return y_binary

# threshold = np.linspace(0,1,11)

# pricision_score, recall_score, f1_pos_score = [], [], []
# for t in threshold:
#     y_binary = binary_label(y_pred, t)
#     conf_matrix = confusion_matrix(y_test, y_binary.flatten(), labels=[0,1])
#     plot_conf_matrix(y_test, y_binary.flatten(), t)
#     result_f1_score = f1_score(y_test, y_binary, average=None)
#     print('threshold = ', t, '\nf1 score = ', result_f1_score)
#     f1_pos_score.append(result_f1_score[1])
#     pricision = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[0,1])
#     recall = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
#     pricision_score.append(pricision)
#     recall_score.append(recall)

# pricision_recall_np = np.zeros((len(threshold),3))
# pricision_recall_np[:,0] = threshold
# pricision_recall_np[:,1] = pricision_score
# pricision_recall_np[:,2] = recall_score

# print(pricision_recall_np)
# plt.plot(threshold[:-1], pricision_score[:-1],'o-',label='Pricision')
# plt.plot(threshold[:-1], recall_score[:-1],'o-', label='Recall')
# plt.plot(threshold[:-1], f1_pos_score[:-1], 'o-', label='f1_score')

# plt.legend()
# plt.xlabel('Threshold')
# plt.savefig('/home/ming/Project/nrn_mapping_package-master/Figure/Pricision_Recall.png', dpi=150)
# plt.show()
# confusion_matrix(y_test, binary_label(y_pred, 0.4).flatten(), labels=[0,1])
# %% compare big model

# cnn2 = CNN_big((50, 50, 3))
# plot_model(cnn2, '/home/ming/Project/Neural_Mapping_ZGT/cnn_big.png', show_shapes=True)

# # %% Load pkl data
# # train_data = load_pkl('/home/ming/Project/Neural_Mapping_ZGT/data/training.pkl')
# # valid_data = load_pkl('/home/ming/Project/Neural_Mapping_ZGT/data/valid.pkl')
# # test_data = load_pkl('/home/ming/Project/Neural_Mapping_ZGT/data/test.pkl')
# # class_weights = [0.52363299, 11.07843137]

# # 設定模型儲存條件(儲存最佳模型)
# checkpoint = ModelCheckpoint('/home/ming/Project/Neural_Mapping_ZGT/CNN_big_Checkpoint.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode="auto")
# # history = cnn.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=64, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), class_weight=class_weights, epochs=50, shuffle=True, callbacks = [checkpoint, early_stopping])   #ClassWeight
# history = cnn2.fit({'FC':X_train_FC, 'EM':X_train_EM}, y_train, batch_size=64, validation_data=({'FC':X_val_FC, 'EM':X_val_EM}, y_val), epochs=50, shuffle=True, callbacks = [checkpoint])

# # %% No Callback
# # history = cnn.fit({'FC':train_data['FC'], 'EM':train_data['EM']}, train_data['y'], batch_size=32, validation_data=({'FC':valid_data['FC'], 'EM':valid_data['EM']}, valid_data['y']), class_weight=class_weights, epochs=50, shuffle=True)

# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.savefig('/home/ming/Project/Neural_Mapping_ZGT/Figure/Train_Curve', dpi=100)
# plt.show()

# cnn_big_train_loss = history.history['loss']
# cnn_big_valid_loss = history.history['val_loss']
# # %%
# model = load_model('/home/ming/Project/Neural_Mapping_ZGT/CNN_Checkpoint.h5')
# y_pred = model.predict({'FC':X_test_FC, 'EM':X_test_EM})
# y_pred_binary = []
# for ans in y_pred:
#     if ans >= 0.5:
#         y_pred_binary.append(1)
#     else:
#         y_pred_binary.append(0)
# conf_matrix = confusion_matrix(y_test.tolist(), y_pred_binary, labels=[0,1])
# # 可视化混淆矩阵
# group_names = ['True Neg','False Pos','False Neg','True Pos']
# group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
# group_percent_false = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()[:2]/np.sum(conf_matrix[0])]
# group_percent_true = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()[2:]/np.sum(conf_matrix[1])]
# group_percent = group_percent_false + group_percent_true
# labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percent)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Purples')
# plt.savefig('/home/ming/Project/Neural_Mapping_ZGT/Figure/ConfuseMatric', dpi=100)
# # F1 Score
# result_f1_score = f1_score(y_test, y_pred_binary, average=None)
# print('F1 Score for Neg:', result_f1_score[0])
# print('F1 Score for Pos:', result_f1_score[1])

# # %%
# plt.plot(cnn_small_train_loss, label='small_model_train_loss')
# plt.plot(cnn_big_train_loss, label='big_model_train_loss')
# plt.legend()
# plt.show()

# plt.plot(cnn_small_valid_loss, label='small_model_valid_loss')
# plt.plot(cnn_big_valid_loss, label='big_model_valid_loss')
# plt.legend()
# plt.show()

# %%
