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
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

import os


seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)
# %%
result_path = './mnist_test/'

train_scale = 15000  #拿多少筆資料來訓練
epoch = 100

# 讀取 mnist 資料
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# %% 將y相同的資料組合
def pair_data(x, y, sample_num=100):
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
    return x_0, x_1, y_merge

x_train_0, x_train_1, y_train_merge = pair_data(x_train, y_train, sample_num=train_scale//10)
x_test_0, x_test_1, y_test_merge = pair_data(x_test, y_test)

# shuffle and take validation
idx = np.random.permutation(len(x_train_0))
x_train_0, x_train_1, y_train_merge = x_train_0[idx], x_train_1[idx], y_train_merge[idx]

#train test split
x_train_0, x_val_0, x_train_1, x_val_1, y_train_merge, y_val_merge = train_test_split(x_train_0, x_train_1, y_train_merge, test_size=0.2, random_state=seed)

# 交換 x_train_0, x_train_1
x_train_0_copy = x_train_0.copy()
x_train_1_copy = x_train_1.copy()

x_train_0 = np.concatenate([x_train_0, x_train_1_copy])
x_train_1 = np.concatenate([x_train_1, x_train_0_copy])
y_train_merge = np.concatenate([y_train_merge, y_train_merge])

def plot_pairs(x_0, x_1, y, idx):
    plt.subplot(1,2,1)
    plt.imshow(x_0[idx], cmap='magma')
    plt.subplot(1,2,2)
    plt.imshow(x_1[idx], cmap='magma')
    print(y[idx])
    plt.show()

# 畫圖
# for i in range(5):
#     plot_pairs(x_train_0, x_train_1, y_train_merge, i)
#     plot_pairs(x_train_0, x_train_1, y_train_merge, i+len(x_train_0)//2)

def augment_data(x_0, x_1, y):
    x_0_copy = x_0.copy()
    x_1_copy = x_1.copy()
    y_aug = [y.copy()]
    for j in range(3):
        x_0_aug = np.zeros_like(x_0)
        x_1_aug = np.zeros_like(x_1)
        y_aug.append(y.copy())
        for i in range(len(x_0)):
            x_0_aug[i] = np.rot90(x_0[i], j)
            x_1_aug[i] = np.rot90(x_1[i], j)
        x_0_copy = np.concatenate([x_0_copy, x_0_aug])
        x_1_copy = np.concatenate([x_1_copy, x_1_aug])
    return x_0_copy, x_1_copy, np.concatenate(y_aug)

# x_train_0, x_train_1, y_train_merge = augment_data(x_train_0, x_train_1, y_train_merge)

# %% 建立模型
def CNN_shared(input_size=(28, 28, 1)):
    inputs = [Input(shape=input_size, name="FC"), Input(shape=input_size, name="EM")]

    # 定义共享卷积层和池化层
    shared_conv1 = Conv2D(32, (3, 3), padding='same', name="conv1")
    shared_bn1 = BatchNormalization(name="bn1")
    shared_act1 = Activation("gelu", name="ac1")
    shared_conv2 = Conv2D(32, (3, 3), padding='same', name="conv2")
    shared_bn2 = BatchNormalization(name='bn2')
    shared_act2 = Activation("gelu", name='ac2')
    shared_pool1 = MaxPool2D(pool_size=(2, 2), name='pool1')
    
    shared_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3')
    shared_bn3 = BatchNormalization(name='bn3')
    shared_act3 = Activation("gelu", name='ac3')
    shared_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4')
    shared_bn4 = BatchNormalization(name='bn4')
    shared_act4 = Activation("gelu", name='ac4')
    shared_pool2 = MaxPool2D(pool_size=(2, 2), name='pool2')

    flattened_layers = []
    for input in inputs:
        conv_layer = shared_conv1(input)
        conv_layer = shared_bn1(conv_layer)
        conv_layer = shared_act1(conv_layer)
        # conv_layer = MaxPool2D(pool_size=(2, 2))(conv_layer)

        conv_layer = shared_conv2(conv_layer)
        conv_layer = shared_bn2(conv_layer)
        conv_layer = shared_act2(conv_layer)
        conv_layer = shared_pool1(conv_layer)

        conv_layer = shared_conv3(conv_layer)
        conv_layer = shared_bn3(conv_layer)
        conv_layer = shared_act3(conv_layer)

        conv_layer = shared_conv4(conv_layer)
        conv_layer = shared_bn4(conv_layer)
        conv_layer = shared_act4(conv_layer)
        conv_layer = shared_pool2(conv_layer)

        conv_layer = Dropout(0.5)(conv_layer)

        flattened_layers.append(Flatten()(conv_layer))

    concat_layer = concatenate(flattened_layers, axis=1)

    output = Dropout(0.5)(concat_layer)
    output = Dense(128)(output)
    output = BatchNormalization()(output)
    output = Activation("gelu")(output)
    # output = Dropout(0.5)(output)

    output = Dense(1, activation="sigmoid")(output)

    model = Model(inputs=inputs, outputs=output)

    return model


def CNN_deep(input_size=(28, 28, 1)):
    inputs = [Input(shape=input_size, name='FC'), Input(shape=input_size, name='EM')]
    part_lst = ['fc', 'em']
    flattened_layers = []
    for i, input in enumerate(inputs):
        conv_layer = Conv2D(16, (3,3), padding='same')(input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac1')(conv_layer)

        conv_layer = Conv2D(32, (3,3), padding='same')(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac2')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        # conv_layer = Dropout(0.2)(conv_layer)

        conv_layer = Conv2D(32, (3,3), padding='same')(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac3')(conv_layer)

        conv_layer = Conv2D(64, (3,3), padding='same')(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac4')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Dropout(0.5)(conv_layer)
        
        flattened_layers.append(Flatten()(conv_layer))
    
    concat_layer = concatenate(flattened_layers, axis=1)

    output = Dropout(0.5)(concat_layer)
    output = Dense(128)(output)
    output = BatchNormalization()(output)
    output = Activation('gelu')(output)

    # output = Dropout(0.2)(output)
    # output = Dense(4)(output)
    # output = BatchNormalization()(output)
    # output = Activation('relu')(output)

    output = Dense(1, activation='sigmoid')(output)
    
    
    model = Model(inputs=inputs, outputs=output)
    return model



model = CNN_shared()
# model = CNN_deep()
model.compile(optimizer=AdamW(learning_rate=0.001), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(name="Bi-Acc")])
model.summary()

# plot model
# tf.keras.utils.plot_model(model, to_file=result_path + 'model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

# %% 訓練模型
x_train_0 = np.expand_dims(x_train_0, axis=-1)  # 讓圖片滿足CNN輸入形狀
x_train_1 = np.expand_dims(x_train_1, axis=-1)

# 設定模型儲存條件(儲存最佳模型)
checkpoint = ModelCheckpoint(result_path + 'model/saved_model.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')

history = model.fit({'FC':x_train_0, 'EM':x_train_1}, y_train_merge, validation_data=({'FC':x_val_0, 'EM':x_val_1}, y_val_merge), epochs=epoch, callbacks=[checkpoint], shuffle=True, verbose=2)

plt.plot(history.history['loss'], label='loss', color='mediumblue')
plt.plot(history.history['val_loss'], label='val_loss', color='orchid')
plt.legend()
plt.savefig(result_path + f'loss_scale{train_scale}.png', dpi=120, bbox_inches='tight')
plt.close()

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


best_model = tf.keras.models.load_model(result_path + 'model/saved_model.h5')

# predict validation dataset result
y_pred_val = best_model.predict({'FC':x_val_0, 'EM':x_val_1}, verbose=2)

print('\nValidation:')
val_result, val_pred_bin = result_analysis(y_pred_val, y_val_merge)


# predict test dataset result
y_pred_test = best_model.predict({'FC':x_test_0, 'EM':x_test_1}, verbose=2)

print('\nTest:')
test_result, test_pred_binary = result_analysis(y_pred_test, y_test_merge)

# save test result
result_df = pd.DataFrame({'scale': train_scale, 'precision': [test_result['Precision']], 'recall': [test_result['Recall']], 'f1_pos': [test_result['F1_pos']]})
result_df.to_csv(result_path + f'res/result{train_scale}.csv', index=False)

# plot predict
# 提取预测值中属于每个类别的部分
y_pred_label0 = y_pred_test[y_test_merge == 0].flatten()
y_pred_label1 = y_pred_test[y_test_merge == 1].flatten()

# 计算两组数据的最小值和最大值
min_val = min(y_pred_label0.min(), y_pred_label1.min())
max_val = max(y_pred_label0.max(), y_pred_label1.max())

# 计算bin的边界
bins = np.linspace(min_val, max_val, 50)

plt.style.use('default')

sns.histplot(y_pred_label0, label="Label 0", color='blue', lw=0.5, alpha=0.6, bins=bins)   
sns.histplot(y_pred_label1, label="Label 1", color='red', lw=0.5, alpha=0.6, bins=bins)

# 设置图标题和坐标轴标签
plt.tick_params(axis='both', which='major', labelsize=12)
plt.minorticks_on()
# 显示图例
plt.legend()
plt.savefig(result_path+f'predict_scale{train_scale}.png', dpi=120, bbox_inches='tight')
plt.close()



# %% 整合全部的結果分析（需完成所有參數的訓練後）
result_folder = result_path + 'res/'
result_file_lst = os.listdir(result_folder)

# 分開有交換和沒交換的結果。
result_lst_origin, result_lst_exchange = [], []
for file_name in result_file_lst:
    if file_name[-5:]=='2.csv':
        result_lst_exchange.append(file_name)
    else:
        result_lst_origin.append(file_name)

# 整合結果
result_df_origin = pd.DataFrame()
for file_name in result_lst_origin:
    result_df_origin = pd.concat([result_df_origin, pd.read_csv(result_folder+file_name)], axis=0)
result_df_origin = result_df_origin.sort_values(by='scale')
result_df_origin = result_df_origin.iloc[:7]

result_df_exchange = pd.DataFrame()
for file_name in result_lst_exchange:
    result_df_exchange = pd.concat([result_df_exchange, pd.read_csv(result_folder+file_name)], axis=0)
result_df_exchange = result_df_exchange.sort_values(by='scale')
result_df_exchange = result_df_exchange.iloc[:7]

# 繪製結果
def plot_score_for(df_column_name='f1_pos'):
    plt.plot(result_df_origin['scale'], result_df_origin[df_column_name], 'o--', label='No Augmentation', color='burlywood')
    plt.plot(result_df_exchange['scale'], result_df_exchange[df_column_name], 'o--', label='Augment by Rotate', color='aquamarine')
    plt.legend()
    plt.title(df_column_name)
    plt.xlabel('Scale of Training Data')
    plt.ylabel(df_column_name)
    plt.savefig(result_path+df_column_name+'.png', dpi=120, bbox_inches='tight')
    plt.minorticks_on()
    plt.show()

plot_score_for('precision')
plot_score_for('recall')
plot_score_for('f1_pos')




# %%
