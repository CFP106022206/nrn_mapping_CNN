# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os


folder_path = '/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/'

#load csv
similarity = pd.read_csv(os.path.join(folder_path, 'FTmodel_predict.csv'))

# 构建字典，将fc_id映射到唯一的数字
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

# 歸一化
matrix_min = np.min(similarity_matrix)
matrix_max = np.max(similarity_matrix)
similarity_matrix = (similarity_matrix - matrix_min) / (matrix_max - matrix_min)

# 計算和理想(每類內部都為1)的差距
ideal_matrix = np.zeros((matrix_size, matrix_size))
idx = 0
for num in lpu_num:
    end_idx = idx + num
    ideal_matrix[idx:end_idx, idx:end_idx] = 1
    idx = end_idx
# plt.imshow(ideal_matrix, cmap='magma')
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Ideal Matrix')
# plt.xticks([0,45,145,216,316])
# plt.yticks([0,45,145,216,316])
# plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/ideal_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

# 计算均方误差（MSE）
mse_model = np.mean((similarity_matrix - ideal_matrix) ** 2)
mae_model = np.mean(np.abs(similarity_matrix - ideal_matrix))
frobenius_model = np.linalg.norm(similarity_matrix - ideal_matrix)
print('MSE:', mse_model)
print('MAE:', mae_model)
print('Frobenius Norm:', frobenius_model)

idx=0   # print 各个脑区在图中的范围
for i, lpu in enumerate(lpu_lst):
    print(lpu, ': '+str(idx) + '~' +str(idx+lpu_num[i]))
    idx += lpu_num[i]

plt.figure(figsize=(10, 7))
plt.imshow(similarity_matrix, cmap='magma')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title(f'Similarity Matrix (MSE: {mse_model:.2f}, MAE: {mae_model:.2f}, Frobenius Norm: {frobenius_model:.2f})')
plt.xticks([45,145,216,316])
plt.yticks([45,145,216,316])
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_single.png', dpi=300, bbox_inches='tight')
plt.show()

# 对行进行层次聚类
def reorder_matrix(matrix, fc_id_lst):
    linkage_matrix = sch.linkage(1 - matrix, method='complete')

    # 根据层次聚类的结果生成排序顺序
    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
    order = dendrogram['leaves']


    # 生成新的fc_id2index字典
    reordered_fc_id2index = {fc_id_lst[i]: idx for idx, i in enumerate(order)}

    # 使用相同的顺序对行和列进行排序
    reordered_matrix = similarity_matrix[order, :]
    reordered_matrix = reordered_matrix[:, order]
    return reordered_matrix, reordered_fc_id2index

reordered_matrix, reordered_fc_id2index = reorder_matrix(similarity_matrix, fc_ids['fc_id'].tolist())

# 可视化原始相似度矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

cax1 = ax1.imshow(similarity_matrix, cmap='magma')
ax1.set_title('Original Similarity Matrix')
ax1.set_xticks([0,45,145,216,316])
ax1.set_yticks([0,45,145,216,316])
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(reordered_matrix, cmap='magma')
ax2.set_title('Reordered Similarity Matrix')
ax2.set_xticks(np.arange(0, matrix_size, 50))
ax2.set_yticks(np.arange(0, matrix_size, 50))
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_compare.png', dpi=300, bbox_inches='tight')
plt.show()


# %%    重排序之後 原本人類標註的四大類是否分散？是否有更多有意義的結構
alln = pd.read_csv(os.path.join(folder_path, 'ALLN.txt'), header=None, names=['fc_id'])
pn = pd.read_csv(os.path.join(folder_path, 'PN.txt'), header=None, names=['fc_id'])
kc = pd.read_csv(os.path.join(folder_path, 'KC.txt'), header=None, names=['fc_id'])
fb = pd.read_csv(os.path.join(folder_path, 'FB.txt'), header=None, names=['fc_id'])

def location_matrix(fc_id_lst, reordered_fc_id2index, matrix_size):
    location_matrix = np.zeros((matrix_size, matrix_size))
    for x in fc_id_lst:
        for y in fc_id_lst:
            location_matrix[reordered_fc_id2index[x], reordered_fc_id2index[y]] = 1
    return location_matrix

alln_matrix = location_matrix(alln['fc_id'].tolist(), reordered_fc_id2index, matrix_size)
pn_matrix = location_matrix(pn['fc_id'].tolist(), reordered_fc_id2index, matrix_size)
kc_matrix = location_matrix(kc['fc_id'].tolist(), reordered_fc_id2index, matrix_size)
fb_matrix = location_matrix(fb['fc_id'].tolist(), reordered_fc_id2index, matrix_size)

fig, axes = plt.subplots(2, 2, figsize=(11, 11))
(ax1, ax2), (ax3, ax4) = axes

cax1 = ax1.imshow(alln_matrix, cmap='magma')
ax1.set_title('ALLN')
cax2 = ax2.imshow(pn_matrix, cmap='magma')
ax2.set_title('PN')
cax3 = ax3.imshow(kc_matrix, cmap='magma')
ax3.set_title('KC')
cax4 = ax4.imshow(fb_matrix, cmap='magma')
ax4.set_title('FB')
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/location_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %% 讀取NBLAST那邊的計算分數
nblast_matrix = np.load(os.path.join(folder_path, 'NBLAST_316.npy'))

# 歸一化
nblast_matrix = (nblast_matrix - np.min(nblast_matrix)) / (np.max(nblast_matrix) - np.min(nblast_matrix))

# 计算均方误差（MSE）
mse_nblast = np.mean((nblast_matrix - ideal_matrix) ** 2)
mae_nblast = np.mean(np.abs(nblast_matrix - ideal_matrix))
frobenius_nblast = np.linalg.norm(nblast_matrix - ideal_matrix)

print('MSE:', mse_nblast)
print('MAE:', mae_nblast)
print('Frobenius Norm:', frobenius_nblast)

plt.figure(figsize=(10, 7))
plt.imshow(nblast_matrix, cmap='magma')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title(f'NBLAST Matrix (MSE: {mse_nblast:.2f}, MAE: {mae_nblast:.2f}, Frobenius Norm: {frobenius_nblast:.2f})')
plt.xticks([45,145,216,316])
plt.yticks([45,145,216,316])
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_NBLAST.png', dpi=300, bbox_inches='tight')
plt.show()


# 畫NBLAST和模型的結果對比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

cax1 = ax1.imshow(nblast_matrix, cmap='magma')
ax1.set_title('NBLAST Similarity Matrix')
ax1.set_xticks([0,45,145,216,316])
ax1.set_yticks([0,45,145,216,316])
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(similarity_matrix, cmap='magma')
ax2.set_title('Model Similarity Matrix')
ax2.set_xticks([0,45,145,216,316])
ax2.set_yticks([0,45,145,216,316])
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_NBLAST_Model_compare.png', dpi=300, bbox_inches='tight')
plt.show()

# # 计算SSIM 结构相似性
# ssim_index, ssim_map = ssim(nblast_matrix, ideal_matrix, full=True,  data_range=1)

# # 打印SSIM指数
# print(f"SSIM index between the two matrices: {ssim_index}")

# # 显示SSIM map
# plt.imshow(ssim_map, cmap='magma')
# plt.colorbar()
# plt.title("SSIM Map")
# plt.show()

# # 计算Frobenius Norm
# frobenius_norm_nblast = np.linalg.norm(nblast_matrix - ideal_matrix)
# print(f"Frobenius Norm of NBLAST-Ideal: {frobenius_norm_nblast}")

# frobenius_norm_model = np.linalg.norm(similarity_matrix - ideal_matrix)
# print(f"Frobenius Norm of Model-Ideal: {frobenius_norm_model}")

# %% 數值分佈圖
def plot_distribution(matrix, title='Distribution'):
    plt.hist(matrix.flatten(), bins=50)
    plt.title(title)
    # plt.xlabel('Similarity Score')
    # plt.ylabel('Frequency')
    # plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/'+title+'.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_distribution(nblast_matrix, 'NBLAST Distribution')
plot_distribution(similarity_matrix, 'Model Distribution')

# %% binary化相似度矩陣
def binarize(matrix, threshold):
    return (matrix > threshold).astype(int)

nblast_binary = binarize(nblast_matrix, 0.5)
model_binary = binarize(similarity_matrix, 0.5)

mse_nblast_binary = np.mean((nblast_binary - ideal_matrix) ** 2)
mse_model_binary = np.mean((model_binary - ideal_matrix)**2)

print('MSE NBLAST Binary:', mse_nblast_binary)
print('MSE Model Binary:', mse_model_binary)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
cax1 = ax1.imshow(nblast_binary, cmap='magma')
ax1.set_title('NBLAST Binary Matrix')
ax1.set_xticks([0,45,145,216,316])
ax1.set_yticks([0,45,145,216,316])
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(model_binary, cmap='magma')
ax2.set_title('Model binary Matrix')
ax2.set_xticks([0,45,145,216,316])
ax2.set_yticks([0,45,145,216,316])
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
plt.show()