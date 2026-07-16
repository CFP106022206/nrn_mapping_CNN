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
from pathlib import Path
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
# %%
import pandas as pd
import numpy as np

# 读取全部名单
label_path = './labeled_info/'
df_filename_lst = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

df_lst = []
for f in df_filename_lst:
    df_lst.append(pd.read_csv(label_path + f + '_conf.csv').drop_duplicates(subset=['fc_id','em_id'])) # 删除重复
df_total = pd.concat(df_lst, ignore_index=True)
df_total.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复
df_total.to_csv(label_path + 'D1-D6_total_conf.csv', index=False)

# %%

# === 找出 (fc_id, em_id) 重複但 label 不一致的行 ===
# 注意：上面的 merge 沒有指定 on=...，會用“共同欄位的交集”當 join key，
# 因此 diff 不等同於「(fc_id, em_id) 重複被刪掉的行」。下面用 duplicated/groupby 做精確檢查。

key_cols = ['fc_id', 'em_id']
if not all(c in df_total.columns for c in key_cols):
    raise KeyError(f"Missing key columns: {key_cols}")
if 'label' not in df_total.columns:
    raise KeyError("Missing column: label")

df_check = df_total.copy()

# 如果上面做過 merge，可能會帶入 '_merge' 欄；不影響 key/label 檢查，但輸出 CSV 時先去掉會更乾淨
if '_merge' in df_check.columns:
    df_check = df_check.drop(columns=['_merge'])

# 先把 key 欄位與 label 統一成乾淨的字串，避免 '123' vs 123、尾空白等造成“看起來重複但判定不重複”
df_check['fc_id'] = df_check['fc_id'].astype(str).str.strip()
df_check['em_id'] = df_check['em_id'].astype(str).str.strip()
df_check['label'] = df_check['label'].astype(str).str.strip()

mask_key_dup_any = df_check.duplicated(subset=key_cols, keep=False)

# 每一組 (fc_id, em_id) 的 label 種類數 > 1 代表 label 有衝突
label_nunique = df_check.groupby(key_cols)['label'].transform(lambda s: s.nunique(dropna=False))
mask_label_conflict = label_nunique > 1

conflict_rows = (
    df_check.loc[mask_key_dup_any & mask_label_conflict]
    .sort_values(key_cols)
    .reset_index(drop=True)
)

conflict_keys = (
    conflict_rows[key_cols]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("\n[check] key-duplicate rows:", int(mask_key_dup_any.sum()))
print("[check] conflict rows:", len(conflict_rows))
print("[check] conflict key groups:", len(conflict_keys))

out_conflict = Path(label_path) / 'conflict_fc_em_label.csv'
conflict_rows.to_csv(out_conflict, index=False)
print(f"[save] {out_conflict}")

# 進一步：列出“被 drop_duplicates(subset=key) 會丟掉的那些行”，並比對它們與保留行的 label
kept = df_check.drop_duplicates(subset=key_cols, keep='first')[key_cols + ['label']].rename(columns={'label': 'label_keep'})
dropped = df_check[df_check.duplicated(subset=key_cols, keep='first')].copy()
dropped = dropped.merge(kept, on=key_cols, how='left')

dropped_label_diff = dropped[dropped['label'] != dropped['label_keep']].sort_values(key_cols).reset_index(drop=True)
out_dropped = Path(label_path) / 'dropped_pairs_label_diff.csv'
dropped_label_diff.to_csv(out_dropped, index=False)
print(f"[save] {out_dropped}  rows={len(dropped_label_diff)}")

print("\nExample conflict rows (top 10):")
print(conflict_rows.head(10)[key_cols + ['label']])

# %%
from pathlib import Path
import sys

p = Path("data/standard_views/FC/5HT1A-F-200016_views.npz")
# p = Path("data/standard_views/EM/203253253_views.npz")

import numpy as np
import matplotlib.pyplot as plt


with np.load(p, allow_pickle=False) as z:
    print("keys:", list(z.files))
    for k in z.files:
        a = z[k]
        print(f"- {k}: shape={a.shape}, dtype={a.dtype}")

    if "views" in z.files:
        v = z["views"]
        print("views stats: min=", int(v.min()), "max=", int(v.max()))
        if v.ndim == 3 and v.shape[0] == 3:
            for i in range(3):
                vi = v[i]
                print(f"  view[{i}]: shape={vi.shape}, min={int(vi.min())}, max={int(vi.max())}, nonzero={(vi>0).sum()}")
plt.imshow(v[0], cmap='magma')
plt.show()
# %%
from pathlib import Path
import csv
import numpy as np
import pandas as pd
PAIRS_CSV = Path("data/pairs_label/D1-D6_total_conf.csv")
FC_DIR    = Path("data/standard_views/FC")
EM_DIR    = Path("data/standard_views/EM")

OUT_CSV   = Path("data/pairs_label/pairs_views_size_report.csv")
MAX_PAIRS = 0   # 0=全跑；比如先测 200 对就写 200

def read_npz_size(npz_path: Path):
    """
    返回 (exists, grid_size, h, w, err)
    - 优先读 grid_size（不会把 views 整个数组读进内存）
    - 若没有 grid_size 才读 views.shape
    """
    if not npz_path.exists():
        return (False, None, None, None, "missing_file")

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "grid_size" in z.files:
                gs = z["grid_size"]
                try:
                    gs = int(gs)
                except Exception:
                    gs = int(np.asarray(gs).reshape(-1)[0])
                return (True, gs, gs, gs, "")
            elif "views" in z.files:
                v = z["views"]
                if v.ndim >= 2:
                    h, w = int(v.shape[-2]), int(v.shape[-1])
                    gs = h if h == w else None
                    return (True, gs, h, w, "")
                else:
                    return (True, None, None, None, "views_ndim_too_small")
            else:
                return (True, None, None, None, f"no_expected_keys:{z.files}")
    except Exception as e:
        return (True, None, None, None, f"load_error:{repr(e)}")

# 读 pairs
with PAIRS_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header: {PAIRS_CSV}")

    header = [h.strip() for h in reader.fieldnames]
    if "fc_id" not in header or "em_id" not in header:
        raise KeyError(f"CSV must contain fc_id/em_id. header={header}")

    rows = list(reader)

if MAX_PAIRS and MAX_PAIRS > 0:
    rows = rows[:MAX_PAIRS]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

abs_diffs = []
pair_max_sizes = []
missing_fc = 0
missing_em = 0
ok_pairs = 0

with OUT_CSV.open("w", encoding="utf-8", newline="") as fo:
    w = csv.writer(fo)
    w.writerow([
        "fc_id",
        "em_id",
        "fc_grid_size",
        "em_grid_size",
        "abs_diff_grid",
        "label",
    ])

    for i, r in enumerate(rows, start=1):
        fc_id = str(r.get("fc_id", "")).strip()
        em_id = str(r.get("em_id", "")).strip()
        label = str(r.get("label", "")).strip()

        fc_npz = FC_DIR / f"{fc_id}_views.npz"
        em_npz = EM_DIR / f"{em_id}_views.npz"

        fc_exists, fc_gs, fc_h, fc_w, fc_err = read_npz_size(fc_npz)
        em_exists, em_gs, em_h, em_w, em_err = read_npz_size(em_npz)

        if not fc_exists:
            missing_fc += 1
        if not em_exists:
            missing_em += 1

        abs_diff = ""
        if fc_gs is not None and em_gs is not None:
            d = abs(int(fc_gs) - int(em_gs))
            abs_diff = d
            abs_diffs.append(d)
            pair_max_sizes.append(max(int(fc_gs), int(em_gs)))
            if fc_exists and em_exists:
                ok_pairs += 1

        w.writerow([
            fc_id,
            em_id,
            fc_gs,
            em_gs,
            abs_diff,
            label,
        ])

print("saved:", OUT_CSV, "rows:", len(rows))
print("missing fc views:", missing_fc)
print("missing em views:", missing_em)
print("ok pairs (both exist & grid_size present):", ok_pairs)

if abs_diffs:
    a = np.asarray(abs_diffs, dtype=np.int32)
    m = np.asarray(pair_max_sizes, dtype=np.int32)
    print("abs_diff_grid: min/median/p90/p99/max =",
          int(a.min()), float(np.median(a)), float(np.percentile(a, 90)),
          float(np.percentile(a, 99)), int(a.max()))
    print("pair_max_grid: min/median/p90/p99/max =",
          int(m.min()), float(np.median(m)), float(np.percentile(m, 90)),
          float(np.percentile(m, 99)), int(m.max()))
else:
    print("No valid grid_size pairs to summarize.")
# %% pseudo labeling前將所有人類標註資料剔除
import pandas as pd
import numpy as np
from pathlib import Path
# %%
label_path = Path("data/pairs_label/D1-D6_total_conf.csv")
predict_label_path = Path("data/pairs_label/EMxFC_all.csv")

predict_df = pd.read_csv(predict_label_path)
label_df = pd.read_csv(label_path)
# 以 fc_id 和 em_id 為鍵，從 predict_df 中剔除 label_df 中存在的行
merged_df = predict_df.merge(label_df[['fc_id', 'em_id']], on=['fc_id', 'em_id'], how='left', indicator=True)
filtered_predict_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
# 进一步：剔除 em_swc 缺失名单中的 neuron
missing_em_txt = Path("data/pairs_label/missing_em_swc_ids.txt")
if missing_em_txt.exists():
    missing_em_ids: set[str] = set()
    with missing_em_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # allow comma/space separated tokens
            for tok in s.replace(",", " ").split():
                t = tok.strip()
                if not t:
                    continue
                if t.endswith(".swc"):
                    t = t[:-4]
                missing_em_ids.add(t)

    if missing_em_ids:
        before_n = len(filtered_predict_df)
        filtered_predict_df = filtered_predict_df[
            ~filtered_predict_df["em_id"].astype(str).str.strip().isin(missing_em_ids)
        ]
        after_n = len(filtered_predict_df)
        print(f"Removed {before_n - after_n} rows by missing EM SWC list: {missing_em_txt}")
else:
    print(f"[warn] missing list not found, skip: {missing_em_txt}")

# 保存剔除後的 DataFrame 到新的 CSV 文件
filtered_predict_df.to_csv("data/pairs_label/EMxFC_all_0_rk20_filtered.csv", index=False)

# %%
# 批量修改文件名
import os
folder_path = './result/unlabel_data_predict/'
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 末尾添加001
        new_filename = filename[:-4] + '02.csv'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        print(f'Renamed: {filename} -> {new_filename}')

# %%
