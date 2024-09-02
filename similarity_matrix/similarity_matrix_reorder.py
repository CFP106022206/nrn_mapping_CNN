# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pandas as pd
import os


folder_path = '/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/'

#load csv
similarity = pd.read_csv(os.path.join(folder_path, 'info_list_unit_YC.csv'))

# 构建字典，将fc_id映射到唯一的数字
# fc_ids = similarity['fc_id'].unique()
# fc_id2index = {fc_id: index for index, fc_id in enumerate(fc_ids)}
# 讀預設順序
lpu_lst = ['ALLN','PN','KC','FB']
fc_id_lst, lpu_num = [], []
for order in lpu_lst:
    fc_id_lst.append(pd.read_csv(os.path.join(folder_path, order+'.txt'), header=None, names=['fc_id']))
    lpu_num.append(len(fc_id_lst[-1]))
fc_ids = pd.concat(fc_id_lst, ignore_index=True)

fc_id2index = {fc_id: index for index, fc_id in enumerate(fc_ids['fc_id'])}
# 過濾掉不在預設順序的
similarity = similarity[similarity['fc_id'].isin(fc_ids['fc_id'])]
similarity = similarity[similarity['em_id'].isin(fc_ids['fc_id'])]
# 获取矩阵的大小
matrix_size = len(fc_ids)
# 创建一个空的矩阵
similarity_matrix = np.zeros((matrix_size, matrix_size))
# 填充矩阵
for _, row in similarity.iterrows():
    row_index = fc_id2index[row['fc_id']]
    col_index = fc_id2index[row['em_id']]
    similarity_matrix[row_index, col_index] = row['score']

idx=0   # print 各个脑区在图中的范围
for i, lpu in enumerate(lpu_lst):
    print(lpu, ': '+str(idx) + '~' +str(idx+lpu_num[i]))
    idx += lpu_num[i]

plt.figure(figsize=(10, 7))
plt.imshow(similarity_matrix, cmap='magma')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Similarity Matrix')
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_single.png', dpi=300, bbox_inches='tight')
plt.show()

# 对行进行层次聚类
linkage_matrix = sch.linkage(1 - similarity_matrix, method='complete')

# 根据层次聚类的结果生成排序顺序
dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
order = dendrogram['leaves']

# 使用相同的顺序对行和列进行排序
reordered_matrix = similarity_matrix[order, :]
reordered_matrix = reordered_matrix[:, order]


# 可视化原始相似度矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# 原始相似度矩阵
cax1 = ax1.imshow(similarity_matrix, cmap='magma')
ax1.set_title('Original Similarity Matrix')
ax1.set_xticks(np.linspace(0, matrix_size, 9))
ax1.set_yticks(np.linspace(0, matrix_size, 9))
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(reordered_matrix, cmap='magma')
ax2.set_title('Reordered Similarity Matrix')
ax2.set_xticks(np.linspace(0, matrix_size, 9))
ax2.set_yticks(np.linspace(0, matrix_size, 9))
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_compare.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
