import numpy as np
import pandas as pd
import os
import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score
from util import load_pkl

save_model_name = 'Train_Test_In_D1-D5'
# %% 对新数据集进行标注

unlabel_path = './data/mapping_data_0.7/'

# 将文件夹下文件名存入列表
file_list = os.listdir(unlabel_path)
# 计算label
model = load_model('./CNN_best_' + save_model_name + '.h5')

def annotator(model, fc_img, em_img):
    # 使用transpose()将数组形状从(3, 50, 50)更改为(50, 50, 3)
    fc_img = np.transpose(fc_img, (1, 2, 0))
    em_img = np.transpose(em_img, (1, 2, 0))

    # 将数据维度扩展至4维（符合CNN输入）
    fc_img = np.expand_dims(fc_img, axis=0)
    em_img = np.expand_dims(em_img, axis=0)
    label = model.predict({'FC':fc_img, 'EM':em_img})
    # binary label
    if label > 0.5:
        label = 1
    else:
        label = 0
    return label



# 初始化一个空的DataFrame，用于存储文件名和计算结果
label_df = pd.DataFrame(columns=['fc_id', 'em_id', 'score', 'label'])

# 遍历母文件夹下的所有条目
for file_name in (file_list):
    # 创建完整文件路径
    file_path = os.path.join(unlabel_path, file_name)

    # 检查是否为.pkl档案
    if file_path.endswith('.pkl'):
        # 读取pkl文件
        data_lst = load_pkl(file_path)
        for data in data_lst:
            # 计算结果
            result = annotator(model, data[3], data[4])

            # 将文件名和计算结果添加到DataFrame
            label_df = label_df.append({'fc_id': data[0], 'em_id': data[1], 'score': data[2], 'label':result}, ignore_index=True)

# 将DataFrame存储为csv文件
label_df.to_csv('./data/label_df_0.7.csv', index=False)