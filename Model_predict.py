# %%
import os
import numpy as np
import pandas as pd
from util import load_pkl
from keras.models import *
from tqdm import tqdm


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



cross_num = 10 #int, 用了幾個模型做cross 訓練0~9

# model_name  = 'Annotator_D1-D6_' +str(num_splits)
model_file = './Annotator_Model/'   #改成計算多個模型的平均值並分析標準差
save_folder_path = './result/unlabel_data_predict/'

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)


# %% 对新数据集进行标注
unlabel_path_01 = './data/statistical_results/three_view_pic_rk10'
unlabel_path_02 = './data/statistical_results/three_view_pic_rk10to20'

# 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
file_list_01 = [file_name for file_name in os.listdir(unlabel_path_01) if file_name.endswith('.pkl')]
file_list_02 = [file_name for file_name in os.listdir(unlabel_path_02) if file_name.endswith('.pkl')]

file_path_01 = [os.path.join(unlabel_path_01, file_name) for file_name in file_list_01]
file_path_02 = [os.path.join(unlabel_path_02, file_name) for file_name in file_list_02]
file_path = file_path_01 + file_path_02


# 计算label

model_lst = []
for i in range(cross_num):
    model_name = 'Annotator_D1-D6_' +str(i)
    model = load_model(model_file + model_name + '.h5')
    model_lst.append(model)




# %%

# 分段完成
sub_length = 2000

if len(file_path) > sub_length:
    
    num = 0
    while num * sub_length < len(file_path):
        start_idx = int(num * sub_length)
        end_idx = int(min((num + 1) * sub_length, len(file_path)))

        file_path_predict = file_path[start_idx:end_idx]
        print('\nProcess on Num.', start_idx, '~', end_idx)

        new_data_lst = []
        # 遍历母文件夹下的所有条目
        for pkl_file in tqdm(file_path_predict, total=len(file_path_predict)):
            # 读取pkl文件
            data_lst = load_pkl(pkl_file)
            for data in data_lst:
                # 计算结果
                # 計算各模型結果
                result_lst = []
                for model in model_lst:
                    result = annotator(model, data[3], data[4])
                    result_lst.append(result)
                # 計算平均值
                result_avg = np.mean(result_lst)

                # 計算二元標籤
                result_bin = 1 if result_avg > 0.5 else 0

                # 計算標準差
                result_std = np.std(result_lst)

                # 将文件名和计算结果添加到DataFrame
                new_data = {'fc_id': data[0], 'em_id': data[1], 'KT_score': data[2], 'model_predict': result_avg, 'binary_label': result_bin, 'pred_std': result_std}
                new_data_lst.append(new_data)

        label_df = pd.DataFrame(new_data_lst)

        # 将DataFrame存储为csv文件
        label_df.to_csv(save_folder_path+'labeled_'+str(num)+'_'+model_name+'.csv', index=False)
        print('\nSave Num:', num)

        num += 1
        # del label_df, data_lst, new_data_lst
    
    print('Program Completed.')

else:
    print('\nLabeling..')

    # 初始化一个空lst，用于存储文件名和计算结果
    new_data_lst = []

    # 遍历母文件夹下的所有条目
    for pkl_file in tqdm(file_path, total=len(file_path)):
        # 读取pkl文件
        data_lst = load_pkl(pkl_file)
        for data in data_lst:
            # 计算结果
            # 計算各模型結果
            result_lst = []
            for model in model_lst:
                result = annotator(model, data[3], data[4])
                result_lst.append(result)
            # 計算平均值
            result_avg = np.mean(result_lst)

            # 計算二元標籤
            result_bin = 1 if result_avg > 0.5 else 0

            # 計算標準差
            result_std = np.std(result_lst)

            # 将文件名和计算结果添加到DataFrame
            new_data = {'fc_id': data[0], 'em_id': data[1], 'KT_score': data[2], 'model_predict': result_avg, 'binary_label': result_bin, 'pred_std': result_std}
            new_data_lst.append(new_data)

    label_df = pd.DataFrame(new_data_lst)

    # 将DataFrame存储为csv文件
    label_df.to_csv(save_folder_path+model_name+'.csv', index=False)
    print('\nSaved')
    print('Program Completed.')
# %%
