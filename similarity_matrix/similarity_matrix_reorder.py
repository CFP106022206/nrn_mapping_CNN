# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os


# 用絕對路徑: Interactive Window 的工作目錄是本檔所在資料夾, 從專案根目錄執行時則是根目錄,
# 相對路徑只有其中一種情況找得到
folder_path = '/cluster/home/ming/Project/nrn_mapping_CNN/similarity_matrix/'
out_dir = folder_path   # 圖與 csv 的輸出位置

MODEL_CSV = 'FineTune_miniLR_D1-D6_0_316.csv'   # model_predict_316.py 產生, 欄位 fc_id, em_id, model_predict
NBLAST_NPY = 'NBLAST_316_official.npy'          # nblast_official_316.py 產生 (官方 NBLAST, 與論文 D1/D2 同設定)

# 全部資料
similarity = pd.read_csv(os.path.join(folder_path, MODEL_CSV))

# 建构总数据表
df = pd.DataFrame()

# 构建字典，将fc_id映射到唯一的数字
# 讀取預設順序
lpu_lst = ['ALLN','PN','KC','FB']
fc_id_lst, lpu_num = [], []
for lpu in lpu_lst:
    fc_id_lst.append(pd.read_csv(os.path.join(folder_path, lpu+'.txt'), header=None, names=['fc_id']))
    lpu_num.append(len(fc_id_lst[-1]))
fc_ids = pd.concat(fc_id_lst, ignore_index=True)
fc_id2index = {fc_id: index for index, fc_id in enumerate(fc_ids['fc_id'])}

df['fc_id'] = fc_ids['fc_id']
df['index'] = df['fc_id'].map(fc_id2index)

axis_sep = [0]
for i in lpu_num:
    axis_sep.append(i+axis_sep[-1])
# axis_sep=[0,45,145,216,316]
# %%
# 過濾掉不在預設順序的
similarity = similarity[similarity['fc_id'].isin(fc_ids['fc_id'])]
similarity = similarity[similarity['em_id'].isin(fc_ids['fc_id'])]  #方便起見，此處的em_id實際上是fc_id

# 获取矩阵的大小
matrix_size = len(fc_ids)

# 创建一个空的矩阵
similarity_matrix = np.zeros((matrix_size, matrix_size))

# 填充矩阵 (row 神經進模型的 FC 輸入, col 神經進 EM 輸入)
similarity_matrix[similarity['fc_id'].map(fc_id2index).to_numpy(),
                  similarity['em_id'].map(fc_id2index).to_numpy()] = similarity['model_predict'].to_numpy()

# 模型的分類頭是串接兩側特徵, 交換輸入的結果不同, 使用平均讓矩陣對稱
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
# plt.xticks(axis_sep)
# plt.yticks(axis_sep)
# plt.savefig(os.path.join(out_dir, 'ideal_matrix.png'), dpi=300, bbox_inches='tight')
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
plt.xticks(axis_sep)
plt.yticks(axis_sep)
plt.savefig(os.path.join(out_dir, 'similarity_matrix_single.png'), dpi=300, bbox_inches='tight')
plt.show()

# 对行进行层次聚类
# 注意: 傳入的是方陣 1 - matrix, scipy 會把每一列當成一個觀測向量 (列與列之間取 euclidean),
# 也就是依「與同腦區所有神經的相似度輪廓」分群, 而不是直接把 1 - matrix 當成距離矩陣
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
    return reordered_matrix, reordered_fc_id2index, linkage_matrix


def reorder_by_lpu(matrix):
    """分别对每个脑区进行重排序, 腦區之間的相對位置不變。

    回傳 (重排後的整個矩陣, fc_id -> 重排後位置, 各腦區的 linkage)。
    """
    reordered_fc_id2index, linkage_lst = {}, []
    for i, num in enumerate(lpu_num):
        start = axis_sep[i]
        _, block_index, linkage_matrix = reorder_matrix(matrix[start:start+num, start:start+num],
                                                        fc_id_lst[i]['fc_id'].tolist())
        # 合并时需要加上脑区原本位置的偏移
        reordered_fc_id2index.update({fc_id: idx + start for fc_id, idx in block_index.items()})
        linkage_lst.append(linkage_matrix)
    # 依重排後位置列出原始位置, 重新排序整个矩阵
    reordered_idx = [fc_id2index[k] for k in sorted(reordered_fc_id2index, key=reordered_fc_id2index.get)]
    reordered = matrix[np.ix_(reordered_idx, reordered_idx)]
    return reordered, reordered_fc_id2index, linkage_lst


# 模型矩陣的重排序 (NBLAST 的結果另存為 nblast_*, 避免覆蓋)
reordered_matrix, reordered_fc_id2index, model_linkage_lst = reorder_by_lpu(similarity_matrix)
df['reordered_index'] = df['fc_id'].map(reordered_fc_id2index)

# 可视化原始相似度矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# 使用虛線分隔開各個種類
lines_positions = axis_sep[1:-1]
lines_width = 0.75   # 設定線寬

cax1 = ax1.imshow(similarity_matrix, cmap='magma')
# ax1.set_title('Original Similarity Matrix')
ax1.set_xticks(axis_sep)
ax1.set_yticks(axis_sep)
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 在原始相似度矩阵中添加白色虚线
for pos in lines_positions:
    ax1.axvline(x=pos, color='white', linestyle='--', linewidth=lines_width)
    ax1.axhline(y=pos, color='white', linestyle='--', linewidth=lines_width)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(reordered_matrix, cmap='magma')
# ax2.set_title('Reordered Similarity Matrix')
ax2.set_xticks(axis_sep)
ax2.set_yticks(axis_sep)
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)

# 在原始相似度矩阵中添加白色虚线
for pos in lines_positions:
    ax2.axvline(x=pos, color='white', linestyle='--', linewidth=lines_width)
    ax2.axhline(y=pos, color='white', linestyle='--', linewidth=lines_width)

plt.savefig(os.path.join(out_dir, 'similarity_matrix_compare.png'), dpi=300, bbox_inches='tight')
plt.show()

# # 高清排序後的相似度矩陣(找新分區位置用)
# plt.figure(figsize=(30, 28))
# plt.imshow(reordered_matrix, cmap='magma')
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.xticks(np.arange(0, 317, 2), rotation=90)
# plt.yticks(np.arange(0, 317, 2))
# plt.savefig(os.path.join(out_dir, 'similarity_matrix_reorder_MEGA.png'), dpi=400, bbox_inches='tight')
# plt.show()

# %%

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
plt.savefig(os.path.join(out_dir, 'location_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% FB 分區: 直接切 FB 重排序所用的同一棵樹, 不再手動讀圖上的索引
# 切樹得到的每一群在 dendrogram 葉序上必定連續, 即重排後圖上的一個色塊。
# part 依圖上由上到下編號為 1, 2, 3。
FB_N_PARTS = 3
fb_i = lpu_lst.index('FB')
fb_df = pd.DataFrame({'fc_id': fc_id_lst[fb_i]['fc_id'].tolist(),
                      'cluster': sch.fcluster(model_linkage_lst[fb_i], FB_N_PARTS, criterion='maxclust')})
fb_df['reordered_index'] = fb_df['fc_id'].map(reordered_fc_id2index)
fb_df = fb_df.sort_values('reordered_index').reset_index(drop=True)
fb_df['part'] = fb_df['cluster'].map({c: i + 1 for i, c in enumerate(fb_df['cluster'].unique())})
span = fb_df.groupby('part')['reordered_index'].agg(['min', 'max', 'size'])
assert (span['max'] - span['min'] + 1 == span['size']).all(), 'FB 各 part 在圖上應為連續區塊'
fb_edges = [axis_sep[fb_i]] + (span['max'] + 1).tolist()   # 各 part 在整個矩陣中的邊界
print('FB part 大小:', span['size'].to_dict(), ' 邊界:', fb_edges)

# 若已跑過 cluster_quantification.py, 順便列出各 part 的左右分布 (x<0 側纜長占比)
lat_path = os.path.join(folder_path, 'laterality_316.csv')
if os.path.exists(lat_path):
    lat = pd.read_csv(lat_path).set_index('fc_id')
    fb_df['frac_x_neg'] = fb_df['fc_id'].map(lat['frac_neg'])
    print(fb_df.groupby('part')['frac_x_neg'].describe()[['count', 'mean', 'min', 'max']].round(2))

fb_df[['fc_id', 'part']].to_csv(os.path.join(out_dir, 'FB_part.csv'), index=False)

# 畫出FB部分, 並標出分區邊界
with plt.style.context('dark_background'):
    plt.figure(figsize=(8, 8))
    plt.imshow(reordered_matrix, cmap='magma')
    for pos in fb_edges[1:-1]:
        plt.axvline(x=pos - 0.5, color='white', linestyle='--', linewidth=lines_width)
        plt.axhline(y=pos - 0.5, color='white', linestyle='--', linewidth=lines_width)
    plt.xlim([axis_sep[fb_i] - 0.5, axis_sep[fb_i + 1] - 0.5])
    plt.ylim([axis_sep[fb_i + 1] - 0.5, axis_sep[fb_i] - 0.5])
    plt.xticks(fb_edges)
    plt.yticks(fb_edges)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('FB Similarity Matrix')
    plt.savefig(os.path.join(out_dir, 'FB_part.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %% 讀取NBLAST那邊的計算分數
nblast_matrix = np.load(os.path.join(folder_path, NBLAST_NPY))
nblast_matrix = (nblast_matrix + nblast_matrix.T) / 2   # 官方 scores="mean" 本身已對稱

# 歸一化
nblast_matrix = (nblast_matrix - np.min(nblast_matrix)) / (np.max(nblast_matrix) - np.min(nblast_matrix))

# 对NBLAST结果做重排序
nblast_reordered_matrix, nblast_reordered_fc_id2index, nblast_linkage_lst = reorder_by_lpu(nblast_matrix)


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
plt.xticks(axis_sep[1:])
plt.yticks(axis_sep[1:])
plt.savefig(os.path.join(out_dir, 'similarity_matrix_NBLAST.png'), dpi=300, bbox_inches='tight')
plt.show()


# 畫NBLAST和模型的結果對比
def plot_compare(nblast, model, file_name, suffix=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    cax1 = ax1.imshow(nblast, cmap='magma')
    ax1.set_title('NBLAST Similarity Matrix' + suffix)
    ax1.set_xticks(axis_sep)
    ax1.set_yticks(axis_sep)
    fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

    cax2 = ax2.imshow(model, cmap='magma')
    ax2.set_title('Model Similarity Matrix' + suffix)
    ax2.set_xticks(axis_sep)
    ax2.set_yticks(axis_sep)
    fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(out_dir, file_name), dpi=300, bbox_inches='tight')
    plt.show()


plot_compare(nblast_matrix, similarity_matrix, 'similarity_matrix_NBLAST_Model_compare.png')
# 兩者各自依自己的層次聚類重排序 (論文 Fig. 10 的形式)
plot_compare(nblast_reordered_matrix, reordered_matrix,
             'similarity_matrix_NBLAST_Model_reordered_compare.png', ' (reordered)')

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
        plt.savefig(os.path.join(out_dir, title+'.png'), dpi=300, bbox_inches='tight')
    plt.show()


plot_distribution(nblast_matrix, 'NBLAST Score Distribution', save=True)
plot_distribution(similarity_matrix, 'Model Score Distribution', save=True)

# %% binary化相似度矩陣
def binarize(matrix, threshold):
    return (matrix > threshold).astype(int)

# 門檻是依舊版矩陣定的; 換矩陣後可參考下一個 cell 印出的 Lowest MAE threshold 重新設定
threshold_nblast = 0.20
threshold_model = 0.50

# 兩者都用各自重排序後的矩陣 (重排序只在腦區內進行, 不影響與 ideal_matrix 的 MAE)
nblast_binary = binarize(nblast_reordered_matrix, threshold_nblast)
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
ax1.set_xticks(axis_sep)
ax1.set_yticks(axis_sep)
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

# 重新排序后的相似度矩阵
cax2 = ax2.imshow(model_binary, cmap='magma')
ax2.set_title(f'Model (Binary, thr={threshold_model:.2f}, MAE={mae_model_binary:.2f})')
ax2.set_xticks(axis_sep)
ax2.set_yticks(axis_sep)
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
plt.savefig(os.path.join(out_dir, 'similarity_matrix_binary.png'), dpi=300, bbox_inches='tight')
plt.show()
# %% 如果只考慮ideal matrix範圍中的MSE
# def located_mse(matrix, ideal_matrix):
#     mse_matrix = (matrix - ideal_matrix) **2
#     mse_lst = mse_matrix[ideal_matrix == 1].tolist()
#     return np.mean(mse_lst)

# nblast_mse = located_mse(nblast_matrix, ideal_matrix)
# model_mse = located_mse(similarity_matrix, ideal_matrix)

def plot_thr_mae_distribution(matrix, ideal_matrix, save_name=None):
    # 計算各threshold下的MAE
    mae_lst = []
    for i in range(0, 105, 5):
        threshold = i/100
        binary_matrix = (matrix > threshold).astype(int)
        mae = np.mean(np.abs(binary_matrix - ideal_matrix))
        mae_lst.append(mae)
    
    print('Lowest MAE:', min(mae_lst))
    print('Threshold:', mae_lst.index(min(mae_lst))/20)

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


    if save_name:
        plt.savefig(os.path.join(out_dir, save_name), dpi=150, bbox_inches='tight')
    plt.show()

plot_thr_mae_distribution(nblast_matrix, ideal_matrix, save_name='threshold_vs_mae_NBLAST.png')
plot_thr_mae_distribution(similarity_matrix, ideal_matrix, save_name='threshold_vs_mae_Model.png')
# %%
