# keras tuner test
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
