# %%
# 此版本為串接冠廷畫圖程式之後的調用模型評分版本
# v2版本是因為冠廷程式修改成ranking_process不運行，因此檔案讀取位置不同，此版本增加一個預處理步驟
#Warning: 以下所有folder路徑必須以"/"結尾
import os
import pickle
import numpy as np
import pandas as pd
from keras.models import *
import time
from collections import defaultdict
from tqdm import tqdm

def load_pkl(path):
    if path[-4:] != '.pkl':
        # print('Check the file type')
        path += '.pkl'
    with open(path,'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data

def collect_img_from(id_lst, map_folder, map_type='sn'):
    img_lst = []
    for i in id_lst:
        img = load_pkl(map_folder + str(i) + '.pkl')[map_type]
        img_lst.append(np.transpose(img, (1, 2, 0)))
    return np.array(img_lst)



model_path = './Fine_Tune_Model/Fine_Tune_Model_150K_1.h5'         # 模型存放路徑
save_folder = './result/predict_result/'                            # 模型預測結果存放路徑

unlabel_pairs_path = './data/statistical_results/match_list.csv'   # 冠廷程式初篩後的pairs csv
unlabel_map_folder = './data/mapping_data2/'                       # 使用者上傳的神經做圖資料夾

map_type = 'sn'

# 檢查路徑存在
if not os.path.exists(model_path):
    print('Model not found:', model_path)
    exit()

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

unlabel_pairs_df = pd.read_csv(unlabel_pairs_path)






# %%
# Inference only: avoid restoring optimizer state (and related warnings)
model = load_model(model_path, compile=False)   # 模型存放資料夾

# Initialize dictionaries with default types to store file names and results
source_img_lst, target_img_lst = [], []
st = time.time()
print('Start collecting...')

# v2
# 去除重複
source_id_set = list(set(unlabel_pairs_df['source_id']))
target_id_set = list(set(unlabel_pairs_df['target_id']))
# source_id_set = list(set(unlabel_pairs_df['fc_id']))
# target_id_set = list(set(unlabel_pairs_df['em_id']))

source_img = collect_img_from(source_id_set, unlabel_map_folder, map_type)
target_img = collect_img_from(target_id_set, unlabel_map_folder, map_type)

for row in tqdm(unlabel_pairs_df.iterrows(), total=len(unlabel_pairs_df)):
    source_id = row[1].iloc[0]
    target_id = row[1].iloc[1]
    source_index = source_id_set.index(source_id)
    target_index = target_id_set.index(target_id)
    source_img_lst.append(source_img[source_index])
    target_img_lst.append(target_img[target_index])


print('End collecting.')
print('Collected Time used:', time.time()-st)

# %%

source_id_lst = unlabel_pairs_df['source_id'].tolist()
target_id_lst = unlabel_pairs_df['target_id'].tolist()
# source_id_lst = unlabel_pairs_df['fc_id'].tolist()
# target_id_lst = unlabel_pairs_df['em_id'].tolist()


# 產生模型輸入numpy array
source_img = np.array(source_img_lst)
target_img = np.array(target_img_lst)

st = time.time()
predict_result = model.predict({'FC':source_img, 'EM':target_img}, verbose=0)
time_used = time.time() - st
print('\nTotal time used:', time_used, f'For {len(source_img)} pairs.')
print('\nAverage time per pair:', time_used/len(source_img))


# 将文件名和计算结果添加到DataFrame
label_df = pd.DataFrame({'source_id': source_id_lst, 'target_id': target_id_lst, 'score': predict_result.flatten()})# online version

# 排序
label_df = label_df.sort_values(by=['source_id', 'score'], ascending=[True, False])


# 将DataFrame存储为csv文件
label_df.to_csv(save_folder+'model_predict.csv', index=False)
print('\nSaved')
print('Program Completed.')
# %%
