# keras tuner test
# %% Test Tensorflow GPU
import tensorflow as tf
tf.test.is_gpu_available()

# %%
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input

# def build_model(hp):
#     units = hp.Int(name="units", min_value=16, max_value=32, step=16)
#     model = keras.Sequential([
#         Dense(units, activation="relu"),
#         Dense(10, activation="softmax")
#     ])
#     optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"])
#     return model

# def build_model(hp):
#     model = keras.Sequential()
#     #   model.add(keras.layers.Flatten(input_shape=(28, 28)))

#     # Tune the number of units in the first Dense layer
#     # Choose an optimal value between 32-512
#     hp_units = hp.Int('units', min_value=16, max_value=64, step=32)
#     model.add(Dense(units=hp_units, activation='relu'))
#     model.add(Dense(10, activation="softmax"))

#     # Tune the learning rate for the optimizer
#     # Choose an optimal value from 0.01, 0.001, or 0.0001
#     # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

#     model.compile(optimizer=hp.Choice(name="optimizer", values=["rmsprop", "adam"]),
#                     loss='sparse_categorical_crossentropy',
#                     metrics=['accuracy'])

#     return model

# def build_model(hp):
#     model = keras.Sequential()
#     model.add(Dense(units=hp.Int('units', min_value=16, max_value=64, step=32), activation='relu'))
#     model.add(Dense(10, activation='softmax'))
#     model.compile(
#         optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"]),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy'])
#     return model

def build_model(hp):
    inputs = Input(shape=(784,))

    x = Dense(units=hp.Int('units_1', min_value=32, max_value=64, step=16),
              activation=hp.Choice('activation_1', values=['relu', 'sigmoid']))(inputs)

    outputs = Dense(units=10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs, name='mnist_model')
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# %%
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=5,              #指定超参数搜索的最大尝试次数
    executions_per_trial=2,
    directory="mnist_kt_test",  #指定了用于存储调谐器状态和结果的目录路径
    overwrite=True,
)

# %%
tuner.search_space_summary()

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255
x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255
x_train_full = x_train[:]
y_train_full = y_train[:]
num_val_samples = 10000
x_train, x_val = x_train[:-num_val_samples], x_train[-num_val_samples:]
y_train, y_val = y_train[:-num_val_samples], y_train[-num_val_samples:]
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
]
tuner.search(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=2,      #verbose=2: 在每个 epoch 结束时输出一条记录，包括训练和验证指标的平均值。
)

# 查看搜索结果
tuner.results_summary()
# %% 查詢最佳的超參數配置
top_n = 4
best_hps = tuner.get_best_hyperparameters(top_n)


# 查看搜索结果
tuner.results_summary(top_n)
# %% 将超参数引入后将原先的测试资料也纳入训练中，最大化训练资料时找出最低的epochs
def get_best_epoch(hp):
    model = build_model(hp)
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10)
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks)
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"\nBest epoch: {best_epoch}")
    return best_epoch

# %%    将测试集纳入训练集重新训练最佳超参数组合
def get_best_trained_model(hp):
    best_epoch = get_best_epoch(hp)
    model = build_model(hp)
    model.fit(
        x_train_full, y_train_full,
        batch_size=128, epochs=int(best_epoch * 1.2))   # 因為現在使用了更多資料來訓練, 訓練 epoch 數要比剛剛找到的最佳 epoch 數多 1.2 倍
    return model

# %%
best_models = []
for hp in best_hps:
    model = get_best_trained_model(hp)
    model.evaluate(x_test, y_test)
    best_models.append(model)
# %% 保存模型
for _i, model in enumerate(best_models):
    model.save('Tuner_MNIST_' + str(_i + 1)+ '.h5')
# %% 验证保存的模型
model = load_model('Tuner_MNIST_3.h5')
model.evaluate(x_test, y_test)
# %%
import pandas as pd
import numpy as np
import os
import copy

def recoTxt(text_name):
    """
    reconstruct the text to the form we'd like to use

    :param text_name:
    :return:
    """
    f = open(text_name + ".swc", "r")
    lis = []
    start = True
    for line in f:
        if start:
            start = False
            if line == '#n T x y z R P\n':
                return None
        if '#' in line:
            continue
        elif line[0] == "\n":
            continue
        else:
            line = line.strip()
            line = line.replace("\t", " ")
            line = line.replace("   ", " ")
            line = line.replace("  ", " ")
            line = line + "\n"
            lis.append(line)
    lis.insert(0, '#n T x y z R P\n')
    f.close()
    f = open(text_name + ".swc", "w")
    for i in lis:
        f.write(i)
    f.close()
# %%
path = 'data/selected_data/test/FC/'
name = '5-HT1B-F-000000'
length_th = 2.5

def tree_builder(path, name, length_th):
    # 假设recoTxt函数已经被定义，用于读取和预处理SWC文件
    recoTxt(path + name)
    
    # 读取SWC文件到DataFrame
    nrn_df = pd.read_csv(path + name + ".swc", sep=" ", header=0, names=["ID", "type", "x", "y", "z", "r", "parent_ID"])
    nrn_df.sort_values(by="ID", inplace=True)
    
    # 检查ID连续性，如果不连续返回错误
    if nrn_df["ID"].iloc[-1] != len(nrn_df):
        return "error"
    
    # 删除不需要的列并初始化新列
    nrn_df.drop(columns=["type", "r"], inplace=True)
    nrn_df["CN"] = 0
    nrn_df["distance"] = 0.0
    nrn_df["Strahler_order"] = 0
    
    # 将ID列设置为索引
    nrn_df.index = nrn_df['ID'].values
    
    # 创建CN列
    for i in nrn_df.index:
        parent_id = nrn_df.at[i, "parent_ID"]
        if parent_id != -1:
            nrn_df.at[parent_id, "CN"] += 1

            # 创建distance列
            nrn_df.at[i, "distance"] = np.sqrt((nrn_df.at[i, "x"] - nrn_df.at[parent_id, "x"])**2 +
                                               (nrn_df.at[i, "y"] - nrn_df.at[parent_id, "y"])**2 +
                                               (nrn_df.at[i, "z"] - nrn_df.at[parent_id, "z"])**2)
    


    # 计算Strahler order
    leaf_nodes = nrn_df[nrn_df['CN'] == 0].index.tolist()
    nrn_df.loc[leaf_nodes, 'Strahler_order'] = 1
    while leaf_nodes:
        new_leaf_nodes = []
        for node in leaf_nodes:
            parent = nrn_df.at[node, 'parent_ID']
            if parent == -1:
                continue
            children = nrn_df[nrn_df['parent_ID'] == parent]
            if all(children['Strahler_order'] > 0):
                max_order = children['Strahler_order'].max()
                if sum(children['Strahler_order'] == max_order) > 1:
                    nrn_df.at[parent, 'Strahler_order'] = max_order + 1
                else:
                    nrn_df.at[parent, 'Strahler_order'] = max_order
                new_leaf_nodes.append(parent)
        leaf_nodes = new_leaf_nodes
    
    # 插值
    interpolated_rows = []
    for index, row in nrn_df.iterrows():
        parent_id = row['parent_ID']
        if parent_id != -1 and row['distance'] > length_th:
            parent_row = nrn_df.loc[parent_id]
            num_points = int(row['distance'] // length_th + 1)
            delta_x = (row['x'] - parent_row['x']) / num_points
            delta_y = (row['y'] - parent_row['y']) / num_points
            delta_z = (row['z'] - parent_row['z']) / num_points
            for i in range(1, num_points):
                interpolated_row = {
                    'ID': index,
                    'x': parent_row['x'] + delta_x * i,
                    'y': parent_row['y'] + delta_y * i,
                    'z': parent_row['z'] + delta_z * i,
                    'parent_ID': parent_id,
                    'CN': row['CN'], 
                    'distance': row['distance'], 
                    'Strahler_order': row['Strahler_order']
                }
                interpolated_rows.append(interpolated_row)

    # 将插值行添加到DataFrame中
    if interpolated_rows:
        interpolated_df = pd.DataFrame(interpolated_rows)
        nrn_df = pd.concat([nrn_df, interpolated_df], ignore_index=True)
        nrn_df.sort_values(by=['ID'], inplace=True)  # Assuming sorting by coordinates or another logic
        nrn_df.reset_index(drop=True, inplace=True)
    #rename column
    nrn_df.rename(columns={'distance':'l', 'Strahler_order':'sn'}, inplace=True)
    
    return nrn_df
