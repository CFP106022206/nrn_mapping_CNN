# %%
import pandas as pd
import os


# %%
find_from = 'em_id' # fc找em用'fc_id', em找fc用'em_id'

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

# 针对df['fc_id']/['em_id']相同者, 用['model_predict']列从大到小排
df = df.sort_values(by=[find_from, 'model_predict'], ascending=[True, False])

# 添加排名, 整数
df['rank'] = df.groupby(find_from)['model_predict'].rank(ascending=False).astype(int)

# 保留排名前5的数据
df_reduced = df[df['rank'] <= 10]

df_reduced.to_csv('./result/unlabel_data_predict/merge_predict_Rank10.csv', index=False)
# %%
