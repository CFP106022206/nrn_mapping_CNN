# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from util import load_pkl
from tqdm import tqdm
from keras import layers, models
from keras.layers import Input, Conv2D, concatenate, MaxPooling2D, Lambda, Flatten
from keras.utils import plot_model


# %%

data_range = 'D5'   #D4 or D5

map_path = './data/all_pkl'

data_aug = False
encoder_mode = 'sep'

use_unet = True     #使用具有跳躍連接的Unet結構，若True則下面contractive_loss要是False
use_contractive_loss = False # 打開會使得相似的輸入擁有相似的lv。注意此功能內部有調整權重的超參數

cae_batch_size = 2     # 如果GPU內存不足，調小

steps_gradient_accumulate = 64 #是否使用梯度累加，通常在極小batch size中使用. 設為0如果不使用

train_epochs = 400

seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)



# %% data prerpare for CAE

def data_gen_CAE(map_path):

    print('\nCollecting 3-View Data Numpy Array..')
    # 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
    file_list = [file_name for file_name in os.listdir(map_path) if file_name.endswith('.pkl')]

    #使用字典存储有三視圖数据, 以 FC_EM 作为键, 使用字典来查找相应的数据, 减少查找时间
    fc_dict, em_dict = {}, {}
    nrn=0
    for file_name in file_list:
        file_path = os.path.join(map_path, file_name)
        data_lst = load_pkl(file_path)
        for data in data_lst:
            fc_name = data[0]
            em_name = data[1]

            fc_data = data[3]
            em_data = data[4]

            # # 懒得写判断是否存在，直接覆盖，效率低再说
            # fc_dict[fc_name] = fc_data
            # em_dict[em_name] = em_data
            fc_dict[nrn] = fc_data
            em_dict[nrn] = em_data

            nrn += 1
    

    resolutions = data[3].shape # (3,50,50)
    
    fc_np = np.zeros((len(fc_dict), resolutions[1], resolutions[2], resolutions[0]))  #pair, 图(三维)
    em_np = np.zeros((len(em_dict), resolutions[1], resolutions[2], resolutions[0]))  #pair, 图(三维)
    fc_nrn_lst, em_nrn_lst = [], []

    # 将字典中的数据搬到numpy中
    for i, (nrn, image) in enumerate(fc_dict.items()):
        fc_nrn_lst.append(nrn)

        # 三視圖填入 data_np
        fc_np[i] = image.transpose((1,2,0)) # FC Image

    for i, (nrn, image) in enumerate(em_dict.items()):
        em_nrn_lst.append(nrn)

        # 三視圖填入 data_np
        em_np[i] = image.transpose((1,2,0))

    # Normalization : x' = x - min(x) / max(x) - min(x)
    fc_np = (fc_np - np.min(fc_np))/(np.max(fc_np) - np.min(fc_np))
    em_np = (em_np - np.min(em_np))/(np.max(em_np) - np.min(em_np))

    return fc_np, em_np, fc_nrn_lst, em_nrn_lst

fc_np, em_np, fc_nrn_lst, em_nrn_lst = data_gen_CAE(map_path)

'''
原输入图片大小为(50,50,3)
然而经过两次CAE中的池化后由于50不是8(对应三次池化)的倍数，会遇到奇数向上取整
此时解码器上采样回来导致图片大小不一致，因此将图片大小通过兩側縮小方式改为(48,48,3)
'''

def add_padding(image_np):

    padded_np = np.zeros((image_np.shape[0], image_np.shape[1]+6, image_np.shape[2]+6, image_np.shape[3]))
    
    for i in range(len(padded_np)):
        padded_np[i] = np.pad(image_np[i], pad_width=((3, 3), (3, 3), (0, 0)), mode='constant', constant_values=0)
    
    
    return padded_np

def remove_pixels(image_np):
    cropped_np = np.zeros((image_np.shape[0], image_np.shape[1]-2, image_np.shape[2]-2, image_np.shape[3]))
    
    for i in range(len(cropped_np)):
        cropped_np[i] = image_np[i, 1:-1, 1:-1, :]
    
    return cropped_np

fc_np = remove_pixels(fc_np)
em_np = remove_pixels(em_np)

# Train Validation Split
fc_np_train_ini, fc_np_valid = train_test_split(fc_np, test_size=0.2, random_state=seed)
em_np_train_ini, em_np_valid = train_test_split(em_np, test_size=0.2, random_state=seed)



# data augmentation for CAE
if data_aug:
    # 镜像翻倍
    def mirror_aug(X_train):
        X_train_aug1 = np.zeros_like(X_train)
        for i in range(X_train_aug1.shape[0]):
            X_train_aug1[i] = np.fliplr(X_train[i])

        X_train = np.vstack((X_train, X_train_aug1))

        # 翻倍
        X_train_aug2 = np.zeros_like(X_train)

        for i in range(X_train_aug2.shape[0]):
            X_train_aug2[i] = np.flipud(X_train[i])

        X_train = np.vstack((X_train, X_train_aug2))
        return X_train

    # 圖片旋轉任一角度
    def rotate_and_pad(image, angle, border_value=(0, 0, 0)):
        # 获取图像尺寸
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # 计算旋转矩阵
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算新图像的尺寸
        new_w = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
        new_h = int(h * abs(np.cos(np.radians(angle))) + w * abs(np.sin(np.radians(angle))))

        # 更新旋转矩阵
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        # 应用旋转和填充
        rotated_image = cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=border_value)

        # 裁剪或填充旋转后的图像以保持原始尺寸
        if new_h > h and new_w > w:
            y_offset = (new_h - h) // 2
            x_offset = (new_w - w) // 2
            rotated_image = rotated_image[y_offset:y_offset + h, x_offset:x_offset + w]
        else:
            y_padding_top = (h - new_h) // 2
            y_padding_bottom = h - new_h - y_padding_top
            x_padding_left = (w - new_w) // 2
            x_padding_right = w - new_w - x_padding_left
            rotated_image = cv2.copyMakeBorder(rotated_image, y_padding_top, y_padding_bottom, x_padding_left, x_padding_right, cv2.BORDER_CONSTANT, value=border_value)

        return rotated_image


    def augment_CAE(X_train, angle_range, aug_seed):
        X_augmented = np.zeros_like(X_train)

        for i in range(X_train.shape[0]):
            current_seed = aug_seed + i         #為每個循環定義一個種子。每張圖片旋轉角度因此不同
            rng = np.random.default_rng(current_seed)
            angle = rng.uniform(angle_range[0], angle_range[1])

            X_augmented[i] = rotate_and_pad(X_train[i], angle)

        return X_augmented


    fc_np_train = mirror_aug(fc_np_train_ini)
    em_np_train = mirror_aug(em_np_train_ini)


    angle_range = [-45, 45]  # 旋转角度范围（在 -45 到 45 之间）
    fc_augmented = augment_CAE(fc_np_train, angle_range, seed)
    em_augmented = augment_CAE(em_np_train, angle_range, seed)

    fc_np_train = np.vstack((fc_np_train, fc_augmented))
    em_np_train = np.vstack((em_np_train, em_augmented))


    # 再做一次
    fc_augmented = augment_CAE(fc_np_train, angle_range, seed)
    em_augmented = augment_CAE(em_np_train, angle_range, seed)

    fc_np_train = np.vstack((fc_np_train, fc_augmented))
    em_np_train = np.vstack((em_np_train, em_augmented))


else:
    fc_np_train = fc_np_train_ini[:50000]
    em_np_train = em_np_train_ini[:50000]


print('Data shape for model input')
print('FC after aug', fc_np_train.shape)




# %%print('EM after aug', em_np_train.shape)


# 定义卷积自编码器
def create_cae(input_shape=(48, 48, 3)):
    # 编码器
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Flatten(name='encode_output')(x)

    # 解码器
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    decoded = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    # 构建模型
    autoencoder = models.Model(encoder_input, decoded)
    encoder = models.Model(encoder_input, encoded)

    return autoencoder, encoder


def upsample_bilinear(inputs, scale):
    # 使用 tf.image.resize 创建一个上采样层
    # 'bilinear' 是双线性插值
    # 注意，tf.image.resize 要求输入的尺寸是 (batch_size, height, width, channels)，并且要求尺寸是浮点数
    return Lambda(lambda x: tf.image.resize(x, tf.cast(tf.shape(x)[1:3] * scale, tf.int32), method='bilinear'))(inputs)


def unet(input_size=(48,48,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    encoded = Flatten()(conv3)

    up4 = concatenate([upsample_bilinear(conv3, 2), conv2], axis=3)  # 使用双线性上采样层
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(up4)

    up5 = concatenate([upsample_bilinear(conv4, 2), conv1], axis=3)  # 使用双线性上采样层
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(up5)

    conv6 = Conv2D(1, 1, activation='sigmoid')(conv5)

    unet = models.Model(inputs=inputs, outputs=conv6)
    encoder = models.Model(inputs=inputs, outputs=encoded)
    return unet, encoder

# 创建模型
if use_unet:
    cae_FC, encoder_FC = unet(input_size=(48,48,3))
    cae_EM, encoder_EM = unet(input_size=(48,48,3))
else:
    cae_FC, encoder_FC = create_cae(input_shape=(48, 48, 3))
    cae_EM, encoder_EM = create_cae(input_shape=(48, 48, 3))

#在損失函數加上針對輸入的雅可比矩陣，做到將相似的輸入擁有相似的lv
def contractive_loss(encoder):
    def loss(y_pred, y_true):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Compute the Jacobian
        with tf.GradientTape() as tape:
            tape.watch(y_pred)
            encoded = encoder(y_pred)

        jacobian = tape.jacobian(encoded, y_pred)
        contractive = tf.reduce_sum(tf.square(jacobian))

        return mse + 1e-4 * contractive  # 1e-4 is a hyperparameter, you can change it
    return loss


# 编译模型
if use_contractive_loss:
    cae_FC.compile(optimizer='adam', loss=contractive_loss(encoder_FC))
    cae_EM.compile(optimizer='adam', loss=contractive_loss(encoder_EM))
else:
    cae_FC.compile(optimizer='adam', loss='mse')
    cae_EM.compile(optimizer='adam', loss='mse')

# 梯度累加器，在batch size被迫很小的時候使用
def train_with_gradient_accumulation(model, train_data, val_data, epochs, accumulation_steps=10, callbacks=None):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        grads = [g/accumulation_steps for g in grads]
        return loss_value, grads

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        val_loss_value = loss_fn(y, logits)
        return val_loss_value

    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        print(f'Start of epoch {epoch+1}')
        losses = []
        for step, (x, y) in enumerate(train_data):
            loss_value, grads = train_step(x, y)
            if step % accumulation_steps == 0:
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                # print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                losses.append(loss_value)
        
        avg_loss = np.mean(losses)
        train_loss_history.append(avg_loss)
        print('loss at epoch= ',epoch, ':', avg_loss)
        # Compute validation loss at the end of the epoch.
        val_losses = []
        for x, y in val_data:
            val_loss = val_step(x, y)
            val_losses.append(val_loss)

        avg_val_loss = np.mean(val_losses)
        val_loss_history.append(avg_val_loss)
        print('va_loss at epoch= ',epoch, ':', avg_val_loss)

        # 在每個epoch結束時調用callbacks的on_epoch_end方法
        logs = {'loss': float(loss_value), 'val_loss': float(avg_val_loss)}
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

    return {'loss': train_loss_history, 'val_loss': val_loss_history}

# 為保存最佳編碼器設定回調
class SaveEncoderCallback(Callback):
    def __init__(self, encoder, filepath, monitor='val_loss', mode='min'):
        super().__init__()
        self.encoder = encoder
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_val_loss = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        if val_loss is not None:
            if self.mode == 'min' and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.encoder.save(self.filepath)
            elif self.mode == 'max' and val_loss > self.best_val_loss:
                self.best_val_loss = val_loss
                self.encoder.save(self.filepath)

class CustomSaveModelCallback(Callback):
    def __init__(self, model, filepath, monitor='val_loss', mode='min'):
        super().__init__()
        self.model = model
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_val_loss = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        if val_loss is not None:
            if self.mode == 'min' and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # tf.saved_model.save(self.model, self.filepath)
                self.model.save(self.filepath)
            elif self.mode == 'max' and val_loss > self.best_val_loss:
                self.best_val_loss = val_loss
                # tf.saved_model.save(self.model, self.filepath)
                self.model.save(self.filepath)


# 設定模型儲存條件(儲存最佳模型)
checkpoint_FC_encoder = SaveEncoderCallback(encoder_FC, './CAE_FC/encoder_FC_best.h5', monitor='val_loss', mode='min')
checkpoint_FC = ModelCheckpoint('./CAE_FC/FC_CAE02.h5', verbose=1, monitor='val_loss', save_best_only=True, save_weight_only=False, mode='min')

checkpoint_EM_encoder = SaveEncoderCallback(encoder_EM, './CAE_EM/encoder_EM_best.h5', monitor='val_loss', mode='min')
checkpoint_EM = ModelCheckpoint('./CAE_EM/EM_CAE02.h5', verbose=1, monitor='val_loss', save_best_only=True, save_weight_only=False, mode='min')



print("Training FC CAE...")
if steps_gradient_accumulate == 0:


    history_CAE_FC = cae_FC.fit(fc_np_train, fc_np_train, 
            epochs=train_epochs, batch_size=cae_batch_size, shuffle=True, 
            validation_data=(fc_np_valid, fc_np_valid),
            callbacks = [checkpoint_FC, checkpoint_FC_encoder])


    plt.plot(history_CAE_FC.history['loss'], label='train')
    plt.plot(history_CAE_FC.history['val_loss'], label='valid')
    plt.legend()
    plt.title('CAE_FC')
    plt.savefig('./CAE_FC/FC_CAE02.png', dpi=150, bbox_inches="tight")
    # plt.show()


else:
    #自帶回調因為使用了自定義loss不可用，所以這裡使用自定義回調
    checkpoint_FC = CustomSaveModelCallback(model=cae_FC, filepath='./CAE_FC/FC_CAE02.h5', monitor='val_loss', mode='min')



    #手動將訓練數據建立為一個批次化的數據集
    train_data = tf.data.Dataset.from_tensor_slices((fc_np_train, fc_np_train))
    train_data = train_data.shuffle(buffer_size=1024).batch(cae_batch_size) #shuffle

    val_data = tf.data.Dataset.from_tensor_slices((fc_np_valid, fc_np_valid))
    val_data = val_data.shuffle(buffer_size=128).batch(cae_batch_size)

    # 創建callback列表
    callbacks = [checkpoint_FC, checkpoint_FC_encoder]

    history_CAE_FC = train_with_gradient_accumulation(cae_FC, train_data, val_data, epochs=train_epochs, accumulation_steps=steps_gradient_accumulate, callbacks=callbacks)    

    plt.plot(history_CAE_FC['loss'], label = 'loss')
    plt.plot(history_CAE_FC['val_loss'], label='valid')
    plt.legend()
    plt.title('CAE_FC')
    plt.sacefig('./CAE_FC/FC_CAE02.png', dpi=150, bbox_inches='tight')

plt.close('all')

print("Training EM CAE...")
if steps_gradient_accumulate == 0:
    history_CAE_EM = cae_EM.fit(em_np_train, em_np_train, 
            epochs=train_epochs, batch_size=cae_batch_size, shuffle=True, 
            validation_data=(em_np_valid, em_np_valid),
            callbacks = [checkpoint_EM, checkpoint_EM_encoder])


    plt.plot(history_CAE_EM.history['loss'], label='train')
    plt.plot(history_CAE_EM.history['val_loss'], label='valid')
    plt.legend()
    plt.title('CAE_EM')
    plt.savefig('./CAE_EM/EM_CAE02.png', dpi=150, bbox_inches="tight")
    # plt.show()

else:
    checkpoint_EM = CustomSaveModelCallback(model=cae_EM, filepath='./CAE_EM/EM_CAE02.h5', monitor='val_loss', mode='min')


    #手動將訓練數據建立為一個批次化的數據集
    train_data = tf.data.Dataset.from_tensor_slices((em_np_train, em_np_train))
    train_data = train_data.shuffle(buffer_size=1024).batch(cae_batch_size) #shuffle

    val_data = tf.data.Dataset.from_tensor_slices((em_np_valid, em_np_valid))
    val_data = val_data.shuffle(buffer_size=128).batch(cae_batch_size)

    # 創建callback列表
    callbacks = [checkpoint_EM, checkpoint_EM_encoder]

    history_CAE_EM = train_with_gradient_accumulation(cae_EM, train_data, val_data, epochs=train_epochs, accumulation_steps=steps_gradient_accumulate, callbacks=callbacks)    

    plt.plot(history_CAE_EM['loss'], label='train')
    plt.plot(history_CAE_EM['val_loss'], label='valid')
    plt.legend()
    plt.title('CAE_EM')
    plt.savefig('./CAE_EM/EM_CAE02.png', dpi=150, bbox_inches='tight')

plt.close('all')



if encoder_mode == 'mix':
    cae_mix, encoder_mix = create_cae(input_shape=(48, 48, 3))
    cae_mix.compile(optimizer='adam', loss='mse')

    checkpoint_mix_encoder = SaveEncoderCallback(encoder_mix, './CAE_mix/encoder_mix_best.h5', monitor='val_loss', mode='min')
    checkpoint_mix = ModelCheckpoint('./CAE_mix/mix_CAE01.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')

    print("Training mix CAE...")
    mix_np_train = np.concatenate((fc_np_train, em_np_train), axis=0)
    mix_np_valid = np.concatenate((fc_np_valid, em_np_valid), axis=0)

    history_CAE_FC = cae_FC.fit(mix_np_train, mix_np_train, 
            epochs=train_epochs, batch_size=128, shuffle=True, 
            validation_data=(mix_np_valid, mix_np_valid),
            callbacks = [checkpoint_mix, checkpoint_mix_encoder])

    plt.plot(history_CAE_FC.history['loss'], label='train')
    plt.plot(history_CAE_FC.history['val_loss'], label='valid')
    plt.legend()
    plt.title('CAE_mix')
    plt.savefig('./CAE_mix/mix_CAE01.png', dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close('all')




# 驗證CAE成果
test_num = 5

if steps_gradient_accumulate == 0:
    cae_FC = models.load_model('./CAE_FC/FC_CAE02.h5')
    cae_EM = models.load_model('./CAE_EM/EM_CAE02.h5')
    # cae_mix = models.load_model('./CAE_mix/mix_CAE01.h5')

else:

    # 创建一个包含你的自定义损失函数的字典
    custom_objects = {'loss': tf.keras.losses.MeanSquaredError()}

    cae_FC = models.load_model('./CAE_FC/FC_CAE02.h5', custom_objects=custom_objects)
    cae_EM = models.load_model('./CAE_EM/EM_CAE02.h5', custom_objects=custom_objects)



def plot_CAE_predict(cae_model, test_num, np_train, save_path):
    predict_img = cae_model.predict(np_train[:test_num])
    fig = plt.figure(figsize=(test_num*2, 4))
    for i in range(test_num):
        plt.subplot(2, test_num, i+1)
        plt.imshow(np_train[i])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, test_num, i+1+test_num)
        plt.imshow(predict_img[i])
        plt.xticks([])
        plt.yticks([])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()

plot_CAE_predict(cae_FC, test_num, fc_np_train, './CAE_FC/FC_CAE01_predict.png')
plot_CAE_predict(cae_EM, test_num, em_np_train, './CAE_EM/EM_CAE01_predict.png')
# plot_CAE_predict(cae_mix, test_num, mix_np_train, './CAE_mix/mix_CAE01_predict.png')

# %%
