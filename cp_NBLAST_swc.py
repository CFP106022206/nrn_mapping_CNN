# %%
import pandas as pd
import os
import shutil

# %% Rebuild output path(clean)
output_path = 'NBLAST_swc_0607/'
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path+'FC/')
os.makedirs(output_path+'EM/')

# %%
# create the csv file
target_lst = ['labeled_info/D2_conf.csv', 'labeled_info/D5_conf.csv', 'labeled_info/D6_conf.csv']
target_df = []
for target in target_lst:
    new_df = pd.read_csv(target)
    target_df.append(new_df[['fc_id', 'em_id']])

df = pd.concat(target_df, axis=0)
df.drop_duplicates(inplace=True)
df.to_csv('nblast_All_list.csv', index=False)

# # 若已經做好現成的list: Load the csv file
# csv_lst = ['nblast_conflict.csv','nblast_missing.csv']

# df_lst = []
# for csv in csv_lst:
#     df = pd.read_csv(csv)
#     df_lst.append(df)

# df = pd.concat(df_lst, axis=0)

fc_lst = df['fc_id'].drop_duplicates().astype(str).to_list()
em_lst = df['em_id'].drop_duplicates().astype(str).to_list()

fc_swc_path = 'data/selected_data/FC'
em_swc_path = 'data/selected_data/EM'
# 將每個fc對應的檔案路徑存成list
fc_path_lst = [os.path.join(fc_swc_path,fc_id+'.swc') for fc_id in fc_lst]
em_path_lst = [os.path.join(em_swc_path,em_id+'.swc') for em_id in em_lst]
# 判斷是否有不存在的路徑
fc_notfound = [path.split('/')[-1] for path in fc_path_lst if not os.path.exists(path)]
fc_path_lst = [path for path in fc_path_lst if os.path.exists(path)]
print('Not Found fc:', fc_notfound)

# FC未找到可能在另一個位置
fc_swc_path_add = 'data/selected_data/FC_add'
fc_path_lst_add = [os.path.join(fc_swc_path_add,fc_id) for fc_id in fc_notfound]
fc_StillNotFound = [path.split('/')[-1] for path in fc_path_lst_add if not os.path.exists(path)]
print('Still Not Found fc:', fc_StillNotFound)

fc_path_lst += fc_path_lst_add

em_notfound = [path for path in em_path_lst if not os.path.exists(path)]
em_path_lst = [path for path in em_path_lst if os.path.exists(path)]
print('Not Found em:', em_notfound)

#將檔案複製到指定位置
for path in fc_path_lst:
    shutil.copy2(path, output_path+'FC/')
for path in em_path_lst:
    shutil.copy2(path, output_path+'EM/')

# %%
