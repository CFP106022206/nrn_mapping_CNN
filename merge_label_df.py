# %%
import pandas as pd
import os


# %%
file_path = './result/unlabel_data_predict'
file_list = os.listdir(file_path)
# 只保留csv檔案
file_list = [f for f in file_list if f.endswith('.csv')]

df_lst = []

for file_name in file_list:
    df = pd.read_csv(os.path.join(file_path, file_name))

    df = df.drop(columns=['KT_score', 'binary_label'])
    df_lst.append(df)

# 合并所有df
df = pd.concat(df_lst, ignore_index=True)

# 针对df['fc_id']相同者, 用['model_predict']列从大到小排
df = df.sort_values(by=['fc_id', 'model_predict'], ascending=[True, False])

# 添加排名, 整数
df['rank'] = df.groupby('fc_id')['model_predict'].rank(ascending=False).astype(int)

df.to_csv('./result/unlabel_data_predict/merge_predict_Rank20.csv', index=False)
# %%
