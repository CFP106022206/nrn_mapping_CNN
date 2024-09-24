# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os


folder_path = '/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/'

# 全部資料
similarity = pd.read_csv(os.path.join(folder_path, 'FTmodel_predict.csv'))

# 建构总数据表
df = pd.DataFrame()

# 构建字典，将fc_id映射到唯一的数字
# 讀預設順序
lpu_lst = ['ALLN','PN','KC','FB']
fc_id_lst, lpu_num = [], []
for lpu in lpu_lst:
    fc_id_lst.append(pd.read_csv(os.path.join(folder_path, lpu+'.txt'), header=None, names=['fc_id']))
    lpu_num.append(len(fc_id_lst[-1]))
fc_ids = pd.concat(fc_id_lst, ignore_index=True)
fc_id2index = {fc_id: index for index, fc_id in enumerate(fc_ids['fc_id'])}

df['fc_id'] = fc_ids['fc_id']
df['index'] = df['fc_id'].map(fc_id2index)

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

# 因為模型交換輸入的結果會有細微差異，也可以使用平均讓矩陣對稱
similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

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
# plt.title(f'Similarity Matrix (MSE: {mse_model:.2f}, MAE: {mae_model:.2f}, Frobenius Norm: {frobenius_model:.2f})')
axis_sep = [0]
for i in lpu_num:
    axis_sep.append(i+axis_sep[-1])
# axis_sep=[0,45,145,216,316]
plt.xticks(axis_sep)
plt.yticks(axis_sep)
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_single.png', dpi=300, bbox_inches='tight')
plt.show()

# 对行进行层次聚类
def reorder_matrix(matrix, fc_id_lst):
    linkage_matrix = sch.linkage(1 - matrix, method='complete')

    # 根据层次聚类的结果生成排序顺序
    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
    lpu = dendrogram['leaves']


    # 生成新的fc_id2index字典
    reordered_fc_id2index = {fc_id_lst[i]: idx for idx, i in enumerate(lpu)}

    # 使用相同的顺序对行和列进行排序
    reordered_matrix = matrix[lpu, :]
    reordered_matrix = reordered_matrix[:, lpu]
    return reordered_matrix, reordered_fc_id2index

# # 直接对完整的矩阵进行重排序
# reordered_matrix, reordered_fc_id2index = reorder_matrix(similarity_matrix, fc_ids['fc_id'].tolist())
# df['reordered_index'] = df['fc_id'].map(reordered_fc_id2index)

# 分别对每个脑区进行重排序
lpu_matrix_lst = [] # 分离每个脑区部分的矩阵
idx=0
for num in lpu_num:
    lpu_matrix_lst.append(similarity_matrix[idx:idx+num, idx:idx+num])
    idx += num
# 排序
reordered_matrix_lst, reordered_fc_id2index_lst = [], []
for i, lpu_matrix in enumerate(lpu_matrix_lst):
    reordered_matrix, reordered_fc_id2index = reorder_matrix(lpu_matrix, fc_id_lst[i]['fc_id'].tolist())
    reordered_matrix_lst.append(reordered_matrix)
    reordered_fc_id2index_lst.append(reordered_fc_id2index)

# 合并 fc_id2index
reordered_fc_id2index = {}
# 合并时需要考虑reordered index 在整个矩阵中的位置需要加上脑区原本位置的偏移
start_pos = [0]
for num in lpu_num[:-1]:
    start_pos.append(start_pos[-1] + num)
for i, reordered_index in enumerate(reordered_fc_id2index_lst):
    reordered_fc_id2index.update({fc_id: idx+start_pos[i] for fc_id, idx in reordered_index.items()})

# 将 reordered_fc_id2index key换成原始矩阵位置
reordered_idx2fc_idx = {v : fc_id2index[k] for k, v in reordered_fc_id2index.items()}
df['reordered_index'] = df['index'].map(reordered_idx2fc_idx)

# 重新排序整个矩阵
reordered_idx = list(reordered_idx2fc_idx.values())
reordered_matrix = similarity_matrix.copy()
reordered_matrix = reordered_matrix[reordered_idx, :]
reordered_matrix = reordered_matrix[:, reordered_idx]

# 可视化原始相似度矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

cax1 = ax1.imshow(similarity_matrix, cmap='magma')
ax1.set_title('Original Similarity Matrix')
ax1.set_xticks(axis_sep)
ax1.set_yticks(axis_sep)
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(reordered_matrix, cmap='magma')
ax2.set_title('Reordered Similarity Matrix')
ax2.set_xticks(axis_sep)
ax2.set_yticks(axis_sep)
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_compare.png', dpi=300, bbox_inches='tight')
plt.show()

'''
1、对其顺序(FB)画重排序之前之后对比的矩阵
2、对NBLAST的矩阵也按照同样的顺序画出来
3、在这个排列顺序上呈现原始的相似度分数
4、同樣的順序用binary化的方式呈現
5、畫MAE隨threshold變化的分佈圖
假設groundtruth是四類，按照這個計算的MAE差不多，但是我們能夠分出更細的分類
6、找ALLN和PN個一個神經，在NBLAST中分數很高，但是在我們的模型中分數很低
7、加上FB分成不同小類的神經3D圖，找有代表性的NBLAST無法區別但是我們可以（我們這邊分數很低但是NBLAST分數很高）
'''

# 以下為使用自動找出色塊邊界的嘗試
# # 找出分裂出三小塊的ID
# # 挖出感興趣的部分
# fb_matrix = reordered_matrix[axis_sep[-2]:, axis_sep[-2]:]
# # 找到邊界位置
# from scipy import ndimage
# # 使用 Sobel 算子进行边缘检测
# sobel_x = ndimage.sobel(fb_matrix, axis=0)
# sobel_y = ndimage.sobel(fb_matrix, axis=1)
# edges = np.hypot(sobel_x, sobel_y)
# # 找到边界位置
# threshold = 0.5 * np.max(edges)  # 例如，设置为边缘强度最大值的一半
# boundary_positions = np.where(edges > threshold)

# # 可视化边缘检测结果
# plt.figure(figsize=(8, 8))
# plt.imshow(fb_matrix, cmap='gray')
# plt.scatter(boundary_positions[1], boundary_positions[0], color='red', s=1)
# plt.title('Detected Boundaries in fb_matrix')
# plt.show()

# %%    重排序之後 原本人類標註的四大類是否分散？是否有更多有意義的結構
# 分腦區資料
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

# 找出FB matrix中1的位置
fb_idx = np.where(fb_matrix == 1)
fb_idx = np.unique(fb_idx[0])
print('FB:', fb_idx)
# 将连在一起的fb_idx对应的fc_id分类到一个list
# 遍历fb_idx，如果fb_idx[i] - fb_idx[i-1] == 1，说明是连续的，否则添加新的一组
fb_idx_lst = []
new_lst = [fb_idx[0]]
for i in range(1, len(fb_idx)):
    if fb_idx[i] - fb_idx[i-1] == 1:
        new_lst.append(fb_idx[i])
    else:
        fb_idx_lst.append(new_lst)
        new_lst = [fb_idx[i]]
fb_idx_lst.append(new_lst)

# 找到对应的fc_id
fb_fc_id_lst = []
for lst in fb_idx_lst:
    # 从reordered_fc_id2index中找到对应的fc_id
    fc_id_sub_lst = [k for k, v in reordered_fc_id2index.items() if v in lst]
    fb_fc_id_lst.append(fc_id_sub_lst)

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
# plt.title(f'NBLAST Matrix (MSE: {mse_nblast:.2f}, MAE: {mae_nblast:.2f}, Frobenius Norm: {frobenius_nblast:.2f})')
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
plt.style.use('default')
def plot_distribution(matrix, title='Distribution', save=False):
    plt.hist(matrix.flatten(), bins=50, color='teal')
    plt.title(title)
    plt.minorticks_on()
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Similarity Score', fontsize=12)
    if save:
        plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/'+title+'.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_distribution(nblast_matrix, 'NBLAST Score Distribution', save=True)
plot_distribution(similarity_matrix, 'Model Score Distribution', save=True)

# %% binary化相似度矩陣
def binarize(matrix, threshold):
    return (matrix > threshold).astype(int)

threshold_nblast = 0.20
threshold_model = 0.55

nblast_binary = binarize(nblast_matrix, threshold_nblast)
# model_binary = binarize(similarity_matrix, threshold_model)
model_binary = binarize(reordered_matrix, threshold_model)

# mse_nblast_binary = np.mean((nblast_binary - ideal_matrix) ** 2)
# mse_model_binary = np.mean((model_binary - ideal_matrix)**2)
mae_nblast_binary = np.mean(np.abs(nblast_binary - ideal_matrix))
mae_model_binary = np.mean(np.abs(model_binary - ideal_matrix))


print('MAE NBLAST Binary:', mae_nblast_binary)
print('MAE Model Binary:', mae_model_binary)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
cax1 = ax1.imshow(nblast_binary, cmap='magma')
ax1.set_title(f'NBLAST (Binary, thr={threshold_nblast:.2f}, MAE={mae_nblast_binary:.2f})')
ax1.set_xticks([0,45,145,216,316])
ax1.set_yticks([0,45,145,216,316])
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(model_binary, cmap='magma')
ax2.set_title(f'Model (Binary, thr={threshold_model:.2f}, MAE={mae_model_binary:.2f})')
ax2.set_xticks([0,45,145,216,316])
ax2.set_yticks([0,45,145,216,316])
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/similarity_matrix_binary.png', dpi=300, bbox_inches='tight')
plt.show()
# %% 如果只考慮ideal matrix範圍中的MSE
# def located_mse(matrix, ideal_matrix):
#     mse_matrix = (matrix - ideal_matrix) **2
#     mse_lst = mse_matrix[ideal_matrix == 1].tolist()
#     return np.mean(mse_lst)

# nblast_mse = located_mse(nblast_matrix, ideal_matrix)
# model_mse = located_mse(similarity_matrix, ideal_matrix)

def plot_thr_mae_distribution(matrix, ideal_matrix, save=False):
    # 計算各threshold下的MAE
    mae_lst = []
    for i in range(0, 105, 5):
        threshold = i/100
        binary_matrix = (matrix > threshold).astype(int)
        mae = np.mean(np.abs(binary_matrix - ideal_matrix))
        mae_lst.append(mae)
    
    fig, ax1 = plt.subplots()
    # 畫分數的分佈圖
    ax1.hist(matrix.flatten(), bins=50, color='teal', label="Prediction's Distribution")
    ax1.set_ylabel('Count', color='teal', fontsize=12)
    ax1.tick_params(axis='y')

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制MAE曲线
    ax2.plot(np.linspace(0, 1, len(mae_lst)), mae_lst, '-d', color='#BB0F1E', label='MAE')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('MAE', color='#BB0F1E', fontsize=12)
    ax2.tick_params(axis='y')

    fig.tight_layout()  # 调整布局以防止标签重叠

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)


    if save:
        plt.savefig('/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/threshold_vs_mae.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_thr_mae_distribution(nblast_matrix, ideal_matrix, save=True)
plot_thr_mae_distribution(similarity_matrix, ideal_matrix, save=True)
# %%
