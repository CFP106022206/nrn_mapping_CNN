
# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from util import load_pkl

resolutions=(3,50,50)

def data_preprocess(file_path, pair_nrn):

        print('\nCollecting 3-View Data Numpy Array..')

        data_np = np.zeros((len(pair_nrn), 2, resolutions[1], resolutions[2], resolutions[0]))  #pair, FC/EM, 图(三维)
        fc_nrn_lst, em_nrn_lst, score_lst = [], [], []


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
                fc_nrn_lst.append(data[0])
                em_nrn_lst.append(data[1])
                score_lst.append(row[2])
        


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

        pair_df = pd.DataFrame({'fc_id':fc_nrn_lst, 'em_id':em_nrn_lst, 'score':score_lst})    # list of pairs

        return data_np, pair_df


def Plot_NRN_Img(data_np, pair_df, idx, cmap='magma'):
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.subplot(2,4,i+1)
        plt.imshow(data_np[idx,0,:,:,i],cmap=cmap)    # FC img
        plt.title('FC: ' + pair_df.iloc[idx][0])
        plt.colorbar()
        plt.subplot(2,4,i+5)
        plt.imshow(data_np[idx,1,:,:,i],cmap=cmap)    # EM img
        plt.title('EM: ' + pair_df.iloc[idx][1])
        plt.colorbar()
    
    plt.subplot(2,4,4)
    plt.imshow(data_np[idx,0,:,:])  # FC
    plt.title('FC: '+ pair_df.iloc[idx][0] + '  score: '+ str(pair_df.iloc[idx][2]))
    plt.subplot(2,4,8)
    plt.imshow(data_np[idx,1,:,:])  # EM
    plt.title('EM: '+ pair_df.iloc[idx][1] + '  score: '+ str(pair_df.iloc[idx][2]))
    plt.savefig('./Figure/SN/'+pair_df.iloc[idx][0]+'_'+pair_df.iloc[idx][1]+'.png', dpi=150, bbox_inches='tight')
    plt.close('all')
# %% Load 图
mapping_data_path = './data/mapping_data_0.7+'
pair_nrn_path = './data/D5_nblast_score.csv'

# 读取csv,没有index
pair_nrn = pd.read_csv(pair_nrn_path, index_col=0).to_numpy()
data_np, nrn_pair = data_preprocess(mapping_data_path, pair_nrn)
nrn_pair.to_csv('./Figure/SN/nrn_pair_D5.csv')
#画图]\
for i in tqdm(range(len(data_np))):
    Plot_NRN_Img(data_np, nrn_pair, i)
# %%
