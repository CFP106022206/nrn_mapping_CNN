# %%
import os
import numpy as np
import pandas as pd
from util import load_pkl
from keras.models import *


model_path = './Annotator_Model/Annotator_D1-D6_0.h5'   # 模型存放路徑
save_folder_path = './result/'                       # 模型預測結果存放路徑
unlabel_path_01 = './data/statistical_results/three_view_pic_rk10' # 使用者上傳的神經做圖資料夾

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)





# %% 

def annotator(model,fc_img, em_img):
    # 使用transpose()将数组形状从(3, 50, 50)更改为(50, 50, 3)
    fc_img = np.transpose(fc_img, (1, 2, 0))
    em_img = np.transpose(em_img, (1, 2, 0))

    # 将数据维度扩展至4维 (1,50,50,3)（符合CNN输入）
    fc_img = np.expand_dims(fc_img, axis=0)
    em_img = np.expand_dims(em_img, axis=0)
    label = model.predict({'FC':fc_img, 'EM':em_img}, verbose=0)

    label = label.flatten()[0]  #因為模型輸出是一個 numpy array

    return label



# 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
file_list_01 = [file_name for file_name in os.listdir(unlabel_path_01) if file_name.endswith('.pkl')]

file_path = [os.path.join(unlabel_path_01, file_name) for file_name in file_list_01]

# 載入模型
model = load_model(model_path)   # 模型存放資料夾






# %%
# 初始化一个空lst，用于存储文件名和计算结果
new_data_lst = []

# 遍历母文件夹下的所有条目
for pkl_file in file_path:
    # 读取pkl文件
    data_lst = load_pkl(pkl_file)
    for data in data_lst:
        # 计算结果
        result = annotator(model, data[3], data[4])

        # 計算二元標籤
        result_bin = 1 if result > 0.5 else 0

        # 将文件名和计算结果添加到DataFrame
        # new_data = {'fc_id': data[0], 'em_id': data[1], 'KT_score': data[2], 'model_predict': result, 'binary_label': result_bin}
        new_data = {'fc_id': data[0], 'em_id': data[1], 'score': result} # online version

        new_data_lst.append(new_data)

label_df = pd.DataFrame(new_data_lst)

# 将DataFrame存储为csv文件
label_df.to_csv(save_folder_path+'model_predict.csv', index=False)
# print('\nSaved')
# print('Program Completed.')
# %%
