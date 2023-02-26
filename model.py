import os
import numpy as np
import tensorflow_addons as tfa
import keras_tuner as kt

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
    model.compile(optimizer='Adam', loss = tfa.losses.SigmoidFocalCrossEntropy(), metrics = [tf.keras.metrics.BinaryAccuracy(name='Bi-Acc')])
    model.summary()
    return model

def CNN_tuner(hp, input_size=(50,50,3)):
    inputs = [Input(shape=input_size, name='EM'), Input(shape=input_size, name='FC')]
    flattened_layers = []
    for input in inputs:
        # 寻找超参数
        conv_layer = Conv2D(filters=hp.Choice('num_filters_1', values=[16, 32, 48]), kernel_size=hp.Choice('kernel_size_1', values=[3, 5]))(input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
        
        conv_layer = Conv2D(filters=hp.Choice('num_filters_2', values=[16, 32]), kernel_size=hp.Choice('kernel_size_2', values=[3, 5]))(conv_layer)
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
    # output = Dense(units=hp.Int('units_3', min_value=2, max_value=8, step=2))(output)    #寻找超参数
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



    model.compile(optimizer='Adam', loss = tfa.losses.SigmoidFocalCrossEntropy(), metrics = [BinaryAccuracy(name='accuracy'), f1])
    # model.summary()
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

# def CNN(input_size=(50,50,3)):
#     inputs = Input(shape=input_size)
#     conv_layer = Conv2D(64, 3, activation = 'relu')(input)
#     conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
    
#     conv_layer = Conv2D(32, 3, activation = 'relu')(conv_layer)
#     conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
#     #add layer
#     # conv_layer = GlobalAvgPool2D()(conv_layer)
#     conv_layer = Dropout(0.2)(conv_layer)
    
#     concat_layer = concatenate(flattened_layers, axis=1)
#     # subtracted = Subtract()(flattened_layers)
#     # subtracted = Add()(flattened_layers)

#     output = Dense(32, activation='relu')(concat_layer)
#     # output = Dense(512, activation='relu')(output)
#     output = Dense(8, activation='relu')(output)
#     output = Dense(1, activation='sigmoid')(output)
    
    
#     model = Model(inputs=inputs, outputs=output)
#     model.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#     model.summary()
#     return model