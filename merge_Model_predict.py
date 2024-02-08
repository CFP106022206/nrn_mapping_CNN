# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# %%
find_from = 'em_id' # fc找em用'fc_id', em找fc用'em_id'

file_path = './result/unlabel_data_predict'
file_list = os.listdir(file_path)
# 只保留csv檔案
file_list = [f for f in file_list if f.endswith('.csv')]

df_lst = []

for file_name in file_list:
    df = pd.read_csv(os.path.join(file_path, file_name))

    df = df.drop(columns=['binary_label'])
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
# 分析冠廷分数和模型分数的相关性
from scipy.stats import pearsonr
from util import load_pkl
import numpy as np

plot_pics_num = 10

unique_id_lst = df[find_from].unique()
r_lst = []

for em in unique_id_lst:
    model_pred = df[df[find_from] == em]['model_predict']
    kt_score = df[df[find_from] == em]['KT_score']
    r_lst.append(pearsonr(model_pred, kt_score)[0])

r_np = np.array(r_lst)

#
singular_idx = np.where(r_np < -0.5)[0]
singular_r = r_np[singular_idx]


data_path = './data/statistical_results/EMxFC_rk0-20'
data_lst = os.listdir(data_path)

def plot_3view(em_id):
    fc_lst = df[df[find_from] == em_id][:5]['fc_id'].tolist()
    for fc_id in fc_lst:
        for data in data_lst:
            if fc_id in data:
                pair_lst = load_pkl(os.path.join(data_path, data))
                for pair in pair_lst:
                    if pair[1] == str(em_id):
                        fc_pics = pair[3]
                        em_pics = pair[4]

                        
                        plt.figure(figsize=(12,6))
                        plt.subplot(2,3,1)
                        plt.imshow(fc_pics[0],cmap='magma')

                        plt.subplot(2,3,2)
                        plt.imshow(fc_pics[1],cmap='magma')

                        plt.subplot(2,3,3)
                        plt.imshow(fc_pics[2],cmap='magma')

                        plt.subplot(2,3,4)
                        plt.imshow(em_pics[0],cmap='magma')

                        plt.subplot(2,3,5)
                        plt.imshow(em_pics[1],cmap='magma')

                        plt.subplot(2,3,6)
                        plt.imshow(em_pics[2],cmap='magma')

                        plt.suptitle(fc_id+'_'+str(em_id))
                        plt.show()

                        break
                break
    

for i in range(plot_pics_num):
    model_pred = df[df[find_from] == unique_id_lst[singular_idx[i]]]['model_predict']
    kt_score = df[df[find_from] == unique_id_lst[singular_idx[i]]]['KT_score']
    
    # Normalized kt_score
    kt_score = (kt_score - kt_score.min()) / (kt_score.max() - kt_score.min())

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=model_pred, y=kt_score)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.ylabel = 'KT_score(Norm)'

    plt.title(str(unique_id_lst[singular_idx[i]])+'  r='+str(pearsonr(model_pred, kt_score)[0]))
    plt.show()

for i in range(20, 20+plot_pics_num):
    em_id = unique_id_lst[singular_idx[i]]
    print('EM', str(em_id))
    plot_3view(em_id)
# %%
