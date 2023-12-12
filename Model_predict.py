# %%
import os
import numpy as np
import pandas as pd
from util import load_pkl
from keras.models import *
from tqdm import tqdm


num_splits = 0 #0~9

save_model_name  = 'Annotator_D1-D6_' +str(num_splits)
save_folder_path = './result/unlabel_data_predict/'
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)


# %% 对新数据集进行标注
unlabel_path = './data/statistical_results/three_view_pic_rk10to20/'

# 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
file_list = [file_name for file_name in os.listdir(unlabel_path) if file_name.endswith('.pkl')]

# 计算label
model = load_model('./Annotator_Model/' + save_model_name + '.h5')

def annotator(model,fc_img, em_img):
    # 使用transpose()将数组形状从(3, 50, 50)更改为(50, 50, 3)
    fc_img = np.transpose(fc_img, (1, 2, 0))
    em_img = np.transpose(em_img, (1, 2, 0))

    # 将数据维度扩展至4维 (1,50,50,3)（符合CNN输入）
    fc_img = np.expand_dims(fc_img, axis=0)
    em_img = np.expand_dims(em_img, axis=0)
    label = model.predict({'FC':fc_img, 'EM':em_img}, verbose=0)

    label = label.flatten()[0]  #因為模型輸出是一個 numpy array

    # binary label
    if label > 0.5:
        label_b = 1
    else:
        label_b = 0
    return label, label_b


# %%

# 分段完成
sub_length = 2000

if len(file_list) > sub_length:
    
    num = 0
    while num * sub_length < len(file_list):
        start_idx = int(num * sub_length)
        end_idx = int(min((num + 1) * sub_length, len(file_list)))

        file_list_predict = file_list[start_idx:end_idx]
        print('\nProcess on Num.', start_idx, '~', end_idx)

        new_data_lst = []
        # 遍历母文件夹下的所有条目
        for file_name in tqdm(file_list_predict, total=len(file_list_predict)):
            # 创建完整文件路径
            file_path = os.path.join(unlabel_path, file_name)

            # 读取pkl文件
            data_lst = load_pkl(file_path)
            for data in data_lst:
                # 计算结果
                result, result_b = annotator(model, data[3], data[4])

                # 将文件名和计算结果添加到DataFrame
                new_data = {'fc_id': data[0], 'em_id': data[1], 'KT_score': data[2], 'model_predict': result, 'binary_label': result_b}
                new_data_lst.append(new_data)

        label_df = pd.DataFrame(new_data_lst)

        # 将DataFrame存储为csv文件
        label_df.to_csv(save_folder_path+'labeled_'+str(num)+'_'+save_model_name+'.csv', index=False)
        print('\nSave Num.:', num)

        num += 1
        del label_df, data_lst, new_data_lst
    
    print('Program Completed.')



else:
    print('\nLabeling..')

    # 初始化一个空lst，用于存储文件名和计算结果
    new_data_lst = []

    # 遍历母文件夹下的所有条目
    for file_name in tqdm(file_list, total=len(file_list)):
    # for file_name in file_list:     # No tqdm version
        # 创建完整文件路径
        file_path = os.path.join(unlabel_path, file_name)

        # 读取pkl文件
        data_lst = load_pkl(file_path)
        for data in data_lst:
            # 计算结果
            result, result_b = annotator(model, data[3], data[4])

            # 将文件名和计算结果添加到DataFrame
            new_data = {'fc_id': data[0], 'em_id': data[1], 'score': data[2], 'label_c': result, 'label': result_b}

            new_data_lst.append(new_data)



    label_df = pd.DataFrame(new_data_lst)

    # 将DataFrame存储为csv文件
    label_df.to_csv(save_folder_path+save_model_name+'.csv', index=False)
    print('\nSaved')
    print('Program Completed.')
# %%
