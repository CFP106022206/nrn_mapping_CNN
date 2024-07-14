# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.optimizers import RMSprop, AdamW
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from model import CNN_shared
import os


seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)

# %%
model_save_path = './mnist_preTrain/pre_train_0.h5'   # 模型存放路徑
train_epochs = 300


# 讀取 mnist 資料
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

train_scale = 50000

def pair_data(x, y, sample_num=1000):
    x_0, x_1, y_merge = [], [], []
    for i in range(10):
        pair = np.random.permutation(x[y==i])
        x_0.append(pair[:sample_num-1:2])
        x_1.append(pair[1:sample_num:2])

    x_0 = np.concatenate(x_0)
    x_1 = np.concatenate(x_1)
    y_merge = np.ones(len(x_0))
    
    # 錯位pair
    x_n_0, x_n_1, y_n_merge = [], [], []
    for i in range(10):
        pair = np.random.permutation(x[y!=i])
        x_n_0.append(pair[:sample_num-1:2])
        x_n_1.append(pair[1:sample_num:2])
    
    x_n_0 = np.concatenate(x_n_0)
    x_n_1 = np.concatenate(x_n_1)
    y_n_merge = np.zeros(len(x_n_0))

    x_0 = np.concatenate([x_0, x_n_0])
    x_1 = np.concatenate([x_1, x_n_1])
    y_merge = np.concatenate([y_merge, y_n_merge])
    
    # reshape to (50, 50)
    add_pixel = (50 - x_0.shape[1])//2
    x_0 = np.pad(x_0, ((0, 0), (add_pixel, add_pixel), (add_pixel, add_pixel)), mode='constant')
    x_1 = np.pad(x_1, ((0, 0), (add_pixel, add_pixel), (add_pixel, add_pixel)), mode='constant')

    return x_0, x_1, y_merge

x_train_0, x_train_1, y_train_merge = pair_data(x_train, y_train, sample_num=train_scale//10)
x_val_0, x_val_1, y_val_merge = pair_data(x_test, y_test, sample_num=1000)

del x_train, y_train, x_test, y_test

# shuffle
idx = np.random.permutation(len(x_train_0))
x_train_0, x_train_1, y_train_merge = x_train_0[idx], x_train_1[idx], y_train_merge[idx]

# 創造圖片的另外兩個視角，產生偽三視圖
def second_view(x):
    x_2 = np.rot90(x, axes=(1,2))
    # 上下拉長左右壓扁
    x_2_zoom = zoom(x_2, [1, 2, 0.5])
    up = (x_2_zoom.shape[1] - x_2.shape[1])//2
    down = up + x_2.shape[1]
    x_2_zoom = x_2_zoom[:, up:down, :]
    add_pixel = x_2.shape[2] - x_2_zoom.shape[2]
    x_2_zoom = np.pad(x_2_zoom, ((0,0), (0,0), (add_pixel//2, add_pixel-add_pixel//2)))
    # clip在0~1之間
    x_2_zoom = np.clip(x_2_zoom, 0, 1)
    return x_2_zoom
def third_view(x):
    x_3 = np.rot90(x,3, axes=(1,2))
    # 上下壓扁左右拉長
    x_3_zoom = zoom(x_3, [1, 0.5, 2])
    left = (x_3_zoom.shape[2] - x_3.shape[2])//2
    right = left + x_3.shape[2]
    x_3_zoom = x_3_zoom[:, :, left:right]
    add_pixel = x_3.shape[1] - x_3_zoom.shape[1]
    x_3_zoom = np.pad(x_3_zoom, ((0,0), (add_pixel//2, add_pixel-add_pixel//2), (0,0)))
    # clip在0~1之間
    x_3_zoom = np.clip(x_3_zoom, 0, 1)
    return x_3_zoom

def gen_3view(x):
    x_3view = np.zeros((x.shape[0], 50, 50, 3))
    x_3view[..., 0] = x
    x_3view[..., 1] = second_view(x)
    x_3view[..., 2] = third_view(x)
    return x_3view

x_train_0 = gen_3view(x_train_0)
x_train_1 = gen_3view(x_train_1)

x_val_0 = gen_3view(x_val_0)
x_val_1 = gen_3view(x_val_1)

# 交換 x_train_0, x_train_1
x_train_0_copy = x_train_0.copy()
x_train_1_copy = x_train_1.copy()

x_train_0 = np.concatenate([x_train_0, x_train_1_copy])
x_train_1 = np.concatenate([x_train_1, x_train_0_copy])
y_train_merge = np.concatenate([y_train_merge, y_train_merge])

del x_train_0_copy, x_train_1_copy

# %% Load model

cnn = CNN_shared((50,50,3))
cnn.compile(optimizer=AdamW(learning_rate=0.001), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[BinaryAccuracy(name='Bi-Acc')])

# 儲存最佳模型
checkpoint = ModelCheckpoint(model_save_path, verbose=1, monitor='val_loss', save_best_only=True, mode='min')


history = cnn.fit({'FC':x_train_0, 'EM':x_train_1}, y_train_merge, 
                validation_data=({'FC':x_val_0, 'EM':x_val_1}, y_val_merge), 
                epochs=train_epochs, 
                shuffle=True, 
                callbacks = [checkpoint], verbose=2)


plt.plot(history.history['loss'], label='loss', color='red')
plt.plot(history.history['val_loss'], label='val_loss', color='orchid')
plt.legend()
plt.savefig('./Figure/Pre-Train_model_loss.png', dpi=100, bbox_inches="tight")
plt.close('all')

# %%

def binary(y_lst):
    y_binary = []
    for y in y_lst:
        if y > 0.5:
            y_binary.append(1)
        else:
            y_binary.append(0)
    return y_binary

def print_conf_martix(conf_matrix, name='0'):

    print('\nConfusion Matrix for ' + name)
    print('True Pos','False Neg')
    print(conf_matrix[0])
    print('False Pos','True Neg')
    print(conf_matrix[1])

def result_analysis(y_pred, y_test):
    y_pred_binary = binary(y_pred)
    y_test = binary(y_test) # for 软标签，统一格式

    conf_matrix = confusion_matrix(y_test, y_pred_binary, labels=[1,0])# 統一標籤格式

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
    return result, y_pred_binary


best_model = tf.keras.models.load_model(model_save_path)

# predict validation dataset result
y_pred_val = best_model.predict({'FC':x_val_0, 'EM':x_val_1}, verbose=2)

print('\nValidation:')
val_result, val_pred_bin = result_analysis(y_pred_val, y_val_merge)
