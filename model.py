import os
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
# import keras_tuner as kt
from keras.regularizers import l2
from keras.losses import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.metrics import BinaryAccuracy
from keras import backend as K
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras import backend as keras


def unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # merge6 = concatenate([drop4,up6], axis = 3)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    # merge7 = concatenate([conv3,up7], axis = 3)
    # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['acc'])
    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['acc'])

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def autoencoder(input_size=256*256):
    input_img = Input(shape=(input_size,))
    encoded = Dense(128,activation = 'relu')(input_img)
    encoded = Dense(32,activation = 'relu')(encoded)
    # encoded = Dense(128,activation = 'relu')(encoded)
    # encoded = LeakyReLU(alpha=0.5)(encoded)

    decoded = Dense(128,activation = 'relu')(encoded)
    # decoded = Dense(2048,activation = 'relu')(encoded)
    decoded = Dense(input_size, activation='sigmoid')(decoded)#最後輸出的數字一定要介於0～1之間，所以用sigmoid

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])#adadelta不行
    autoencoder.summary()
    return autoencoder

def CNN_small(input_size=(256,256,3)):
    inputs = [Input(shape=input_size, name='EM'), Input(shape=input_size, name='FC')]
    flattened_layers = []
    for input in inputs:
        # conv_layer = Conv2D(32, 3, activation = 'relu')(input)
        conv_layer = Conv2D(32, (3,3))(input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Conv2D(32, (3,3))(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        # # conv_layer = GlobalAvgPool2D()(conv_layer)
        # # conv_layer = Dropout(0.2)(conv_layer)
        flattened_layers.append(Flatten()(conv_layer))
    
    concat_layer = concatenate(flattened_layers, axis=1)
    # subtracted = Subtract()(flattened_layers)
    # subtracted = Add()(flattened_layers)
    output = Dropout(0.5)(concat_layer)
    output = Dense(8)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    # output = Dropout(0.5)(output)
    # output = Dense(8, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)
    
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model


def CNN_focal(input_size=(50,50,3)):
    inputs = [Input(shape=input_size, name='EM'), Input(shape=input_size, name='FC')]
    flattened_layers = []
    for input in inputs:
        # conv_layer = Conv2D(32, 3, activation = 'relu')(input)
        conv_layer = Conv2D(32, (3,3))(input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Conv2D(32, (3,3))(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        # # conv_layer = GlobalAvgPool2D()(conv_layer)
        # # conv_layer = Dropout(0.2)(conv_layer)
        flattened_layers.append(Flatten()(conv_layer))
    
    concat_layer = concatenate(flattened_layers, axis=1)
    # subtracted = Subtract()(flattened_layers)
    # subtracted = Add()(flattened_layers)
    output = Dropout(0.5)(concat_layer)
    output = Dense(8)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    # output = Dropout(0.5)(output)
    # output = Dense(8, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)
    
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics = [tf.keras.metrics.BinaryAccuracy(name='Bi-Acc')])
    model.summary()
    return model

def CNN_tuner(hp, input_size=(50,50,3)):
    inputs = [Input(shape=input_size, name='EM'), Input(shape=input_size, name='FC')]
    flattened_layers = []
    for input in inputs:
        # 寻找超参数
        conv_layer = Conv2D(filters=hp.Choice('conv_filters_1', values=[32, 48, 56, 64, 72]), kernel_size=hp.Choice('kernel_size_1', values=[3, 5, 7]))(input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Conv2D(filters=hp.Choice('conv_filters_2', values=[24, 48, 64, 80, 96, 128]), kernel_size=hp.Choice('kernel_size_2', values=[3, 5, 7]))(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Conv2D(filters=hp.Choice('conv_filters_3', values=[48, 64, 96, 128, 256]), kernel_size=hp.Choice('kernel_size_3', values=[3, 5, 7]))(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        flattened_layers.append(Flatten()(conv_layer))
    
    concat_layer = concatenate(flattened_layers, axis=1)
    # subtracted = Subtract()(flattened_layers)
    # subtracted = Add()(flattened_layers)
    output = Dropout(hp.Choice('drop_out_1', values=[0.4, 0.5, 0.6]))(concat_layer)
    # output = Dense(8)(output)
    output = Dense(units=hp.Int('dense_units_1', min_value=64, max_value=512, step=64))(output)    #寻找超参数
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    # output = Dropout(0.5)(output)
    # output = Dense(8, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)
    
    model = Model(inputs=inputs, outputs=output)

    # 写 F1 score的matric

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam", "sgd"])

    model.compile(optimizer=optimizer, loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics = [BinaryAccuracy(name='accuracy'), f1])
    # model.summary()
    return model


def CNN_best(input_size=(50,50,3)):
    inputs = [Input(shape=input_size, name='EM'), Input(shape=input_size, name='FC')]
    flattened_layers = []
    for input in inputs:
        conv_layer = Conv2D(56, (5,5))(input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Conv2D(24, (3,3))(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        # # conv_layer = GlobalAvgPool2D()(conv_layer)
        # # conv_layer = Dropout(0.2)(conv_layer)
        flattened_layers.append(Flatten()(conv_layer))
    
    concat_layer = concatenate(flattened_layers, axis=1)
    # subtracted = Subtract()(flattened_layers)
    # subtracted = Add()(flattened_layers)
    output = Dropout(0.5)(concat_layer)
    output = Dense(256)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = Dense(1, activation='sigmoid')(output)
    
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='rmsprop', loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics = [tf.keras.metrics.BinaryAccuracy(name='Bi-Acc')])
    model.summary()
    return model


def CNN_deep(input_size=(50,50,3)):
    inputs = [Input(shape=input_size, name='FC'), Input(shape=input_size, name='EM')]
    part_lst = ['fc', 'em']
    flattened_layers = []
    for i, input in enumerate(inputs):
        conv_layer = Conv2D(16, (3,3), padding='same')(input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac1')(conv_layer)

        conv_layer = Conv2D(16, (3,3), padding='same')(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac2')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        # conv_layer = Dropout(0.2)(conv_layer)

        conv_layer = Conv2D(32, (3,3), padding='same')(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac3')(conv_layer)

        conv_layer = Conv2D(32, (3,3), padding='same')(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('gelu', name=part_lst[i]+'_ac4')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Dropout(0.3)(conv_layer)
        
        flattened_layers.append(Flatten()(conv_layer))
    
    concat_layer = concatenate(flattened_layers, axis=1)

    output = Dropout(0.5)(concat_layer)
    # output = Dense(128)(output)
    # output = BatchNormalization()(output)
    # output = Activation('gelu')(output)

    # output = Dropout(0.2)(output)
    # output = Dense(4)(output)
    # output = BatchNormalization()(output)
    # output = Activation('relu')(output)

    output = Dense(1, activation='sigmoid')(output)
    
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='rmsprop', loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics = [tf.keras.metrics.BinaryAccuracy(name='Bi-Acc')])
    model.summary()
    return model


def CNN_shared(input_size=(50, 50, 3)):
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
    output = Dense(256)(output)
    output = BatchNormalization()(output)
    output = Activation("gelu")(output)
    # output = Dropout(0.5)(output)

    output = Dense(1, activation="sigmoid")(output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(name="Bi-Acc")])
    model.summary()
    return model

def CNN_L2shared(input_size=(50, 50, 3), l2_reg=1e-6):
    inputs = [Input(shape=input_size, name="EM"), Input(shape=input_size, name="FC")]

    # 定义共享卷积层和池化层
    shared_conv1 = Conv2D(24, (3, 3), kernel_regularizer=l2(l2_reg), name="Shared_Conv1")
    shared_bn1 = BatchNormalization(name="Shared_BN1")
    shared_act1 = Activation("relu", name="Shared_Activation1")

    shared_conv2 = Conv2D(32, (3, 3), kernel_regularizer=l2(l2_reg), name="Shared_Conv2")
    shared_bn2 = BatchNormalization(name='Shared_BN2')
    shared_act2 = Activation("relu", name='Shared_Activation2')
    shared_pool1 = MaxPool2D(pool_size=(2, 2), name='Shared_pool1')
    
    shared_conv3 = Conv2D(48, (3, 3), kernel_regularizer=l2(l2_reg), name='Shared_Conv3')   
    shared_bn3 = BatchNormalization(name='Shared_BN3')
    shared_act3 = Activation("relu", name='Shared_Activation3')

    shared_conv4 = Conv2D(64, (3, 3), kernel_regularizer=l2(l2_reg), name='Shared_Conv4')
    shared_bn4 = BatchNormalization(name='Shared_BN4')
    shared_act4 = Activation("relu", name='Shared_Activation4')
    shared_pool2 = MaxPool2D(pool_size=(2, 2), name='Shared_pool2')

    flattened_layers = []
    for input in inputs:
        conv_layer = shared_conv1(input)
        conv_layer = shared_bn1(conv_layer)
        conv_layer = shared_act1(conv_layer)

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
        
        flattened_layers.append(Flatten()(conv_layer))

    concat_layer = concatenate(flattened_layers, axis=1)

    output = Dropout(0.5)(concat_layer)
    output = Dense(256, kernel_regularizer=l2(l2_reg))(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    # output = Dropout(0.2)(output)
    output = Dense(1, activation="sigmoid")(output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="rmsprop", loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(name="Bi-Acc")])
    model.summary()
    return model

def CNN_big(input_size=(256,256,3)):
    inputs = [Input(shape=input_size, name='EM'), Input(shape=input_size, name='FC')]
    flattened_layers = []
    for input in inputs:
        conv_layer = Conv2D(64, 3, activation = 'relu')(input)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Conv2D(64, 3, activation = 'relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer) 
        
        conv_layer = Conv2D(64, 3, activation = 'relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        #add layer
        # conv_layer = GlobalAvgPool2D()(conv_layer)
        flattened_layers.append(Flatten()(conv_layer))
    
    concat_layer = concatenate(flattened_layers, axis=1)
    # subtracted = Add()(flattened_layers)

    output = Dense(512, activation='relu')(concat_layer)
    output = Dense(128, activation='relu')(output)
    output = Dense(32, activation='relu')(output)
    output = Dense(8, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)
    
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model




# %% 2025/09/13 嘗試 MVCNN

def MVCNN_Siamese(input_size=(50, 50, 3), pool_type="max"):
    """
    MVCNN + Siamese Network
    输入: 两组图片 (FC, EM)，每组是50x50x3的图像（三个视角叠在RGB通道）
    输出: 二分类结果（是否同类）
    """
    
    # 定义两个输入
    inputs = [Input(shape=input_size, name="FC"), 
              Input(shape=input_size, name="EM")]
    
    # 定义共享的卷积层（用于处理每个视角）
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
    
    def process_single_view(view_input):
        """处理单个视角的函数"""
        x = shared_conv1(view_input)
        # x = shared_bn1(x)
        x = shared_act1(x)
        x = shared_conv2(x)
        # x = shared_bn2(x)
        x = shared_act2(x)
        x = shared_pool1(x)
        x = shared_conv3(x)
        # x = shared_bn3(x)
        x = shared_act3(x)
        x = shared_conv4(x)
        # x = shared_bn4(x)
        x = BatchNormalization()(x) # 针对每一个视角独立一个batch normalization层，而不使用共享参数
        x = shared_act4(x)
        x = shared_pool2(x)
        x = Dropout(0.1)(x)
        return Flatten()(x)
    
    def extract_multiview_features(input_image):
        """
        从输入图像中提取多视角特征
        输入: (50, 50, 3) - 三个视角叠在RGB通道
        输出: 融合后的特征向量
        """
        # 分离三个视角 (每个channel代表一个视角)
        views = [Lambda(lambda x: tf.expand_dims(x[..., i], axis=-1))(input_image) 
                 for i in range(3)]
        
        # 每个视角通过共享的CNN处理
        view_features = [process_single_view(v) for v in views]
        
        # 堆叠视角特征: (batch, num_views, feature_dim)
        stacked_features = Lambda(lambda x: tf.stack(x, axis=1))(view_features)
        
        # View Pooling: 融合多视角信息
        if pool_type == "max":
            pooled_features = Lambda(lambda x: tf.reduce_max(x, axis=1))(stacked_features)
        elif pool_type == "mean":
            pooled_features = Lambda(lambda x: tf.reduce_mean(x, axis=1))(stacked_features)
        else:  # concatenate
            pooled_features = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1)))(stacked_features)
        
        return pooled_features
    
    # 处理两组输入
    features_FC = extract_multiview_features(inputs[0])
    features_EM = extract_multiview_features(inputs[1])
    
    # Siamese结构：合并两个特征
    # 方法1: 使用绝对差值（对称性好）
    # merged = Lambda(lambda x: tf.abs(x[0] - x[1]))([features_FC, features_EM])
    
    # 方法2: 拼接特征（保留更多信息）
    merged = concatenate([features_FC, features_EM], axis=1)
    
    # 分类头
    output = Dropout(0.3)(merged)
    output = Dense(256)(output)
    output = BatchNormalization()(output)
    output = Activation("gelu")(output)
    output = Dense(1, activation="sigmoid")(output)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=output)
    
    # # 编译模型
    # try:
    #     # 如果有focal loss库
    #     model.compile(
    #         optimizer=RMSprop(learning_rate=0.001),
    #         loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
    #         metrics=[tf.keras.metrics.BinaryAccuracy(name="Bi-Acc")]
    #     )
    # except:
    #     # 使用标准的二元交叉熵
    #     model.compile(
    #         optimizer=RMSprop(learning_rate=0.001),
    #         loss='binary_crossentropy',
    #         metrics=[tf.keras.metrics.BinaryAccuracy(name="Bi-Acc")]
    #     )
    
    # model.summary()
    return model

def MVCNN_Siamese_Advanced(input_size=(50, 50, 3), pool_type="max"):
    """
    增强版的MVCNN + Siamese Network
    包含更多的特征融合选项和注意力机制
    """
    
    # 定义两个输入
    inputs = [Input(shape=input_size, name="FC"), 
              Input(shape=input_size, name="EM")]
    
    # 定义共享的卷积层
    def create_view_encoder():
        """创建处理单个视角的编码器"""
        view_input = Input(shape=(50, 50, 1))
        
        x = Conv2D(32, (3, 3), padding='same')(view_input)
        x = BatchNormalization()(x)
        x = Activation("gelu")(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("gelu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("gelu")(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("gelu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("gelu")(x)
        x = Dropout(0.5)(x)
        
        x = Flatten()(x)
        
        return Model(view_input, x)
    
    # 创建共享的视角编码器
    view_encoder = create_view_encoder()
    
    def extract_multiview_features_with_attention(input_image):
        """
        带注意力机制的多视角特征提取
        """
        # 分离三个视角
        views = [Lambda(lambda x: tf.expand_dims(x[..., i], axis=-1))(input_image) 
                 for i in range(3)]
        
        # 每个视角通过共享的编码器
        view_features = [view_encoder(v) for v in views]
        
        # 学习视角权重（简单的注意力机制）
        stacked_features = Lambda(lambda x: tf.stack(x, axis=1))(view_features)
        
        # 计算注意力权重
        attention_scores = Dense(3, activation='softmax')(
            Lambda(lambda x: tf.reduce_mean(x, axis=-1))(stacked_features)
        )
        attention_scores = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention_scores)
        
        # 应用注意力权重
        weighted_features = Lambda(lambda x: x[0] * x[1])([stacked_features, attention_scores])
        
        # 融合
        if pool_type == "max":
            pooled = Lambda(lambda x: tf.reduce_max(x, axis=1))(weighted_features)
        elif pool_type == "mean":
            pooled = Lambda(lambda x: tf.reduce_mean(x, axis=1))(weighted_features)
        else:  # weighted sum
            pooled = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_features)
        
        return pooled
    
    # 处理两组输入
    features_FC = extract_multiview_features_with_attention(inputs[0])
    features_EM = extract_multiview_features_with_attention(inputs[1])
    
    # Siamese融合：组合多种相似度度量
    # 绝对差值
    diff_features = Lambda(lambda x: tf.abs(x[0] - x[1]))([features_FC, features_EM])
    # 哈达玛积（元素积）
    hadamard_features = Lambda(lambda x: x[0] * x[1])([features_FC, features_EM])
    # 拼接所有特征
    merged = concatenate([features_FC, features_EM, diff_features, hadamard_features], axis=1)
    
    # 更深的分类头
    output = Dense(512)(merged)
    output = BatchNormalization()(output)
    output = Activation("gelu")(output)
    output = Dropout(0.5)(output)
    
    output = Dense(256)(output)
    output = BatchNormalization()(output)
    output = Activation("gelu")(output)
    output = Dropout(0.3)(output)
    
    output = Dense(1, activation="sigmoid")(output)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=output)
    
    # 编译模型
    try:
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="Bi-Acc")]
        )
    except:
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.BinaryAccuracy(name="Bi-Acc")]
        )
    
    model.summary()
    return model