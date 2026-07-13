# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from swc_util import load_swc_fast


def load_swc(swc_path):
    if swc_path[-4:] != '.swc':
        swc_path = swc_path + '.swc'

    swc = load_swc_fast(swc_path)
    return swc.xyz


def find_CM(swc_np):
    swc_cm = np.mean(swc_np, axis=0)
    return swc_cm


def calculate_moi(swc_np):

    swc_cm = np.mean(swc_np, axis=0)

    # calculate moment of inertia
    moi_array = np.zeros((3,3))

    xx = np.sum((swc_np[:, 1] - swc_cm[1])**2 + (swc_np[:, 2] - swc_cm[2])**2)
    yy = np.sum((swc_np[:, 0] - swc_cm[0])**2 + (swc_np[:, 2] - swc_cm[2])**2)
    zz = np.sum((swc_np[:, 0] - swc_cm[0])**2 + (swc_np[:, 1] - swc_cm[1])**2)

    xy = np.sum(-(swc_np[:, 0] - swc_cm[0])*(swc_np[:, 1] - swc_cm[1]))
    xz = np.sum(-(swc_np[:, 0] - swc_cm[0])*(swc_np[:, 2] - swc_cm[2]))
    yz = np.sum(-(swc_np[:, 1] - swc_cm[1])*(swc_np[:, 2] - swc_cm[2]))

    moi_array[0,0] = xx
    moi_array[1,1] = yy
    moi_array[2,2] = zz
    moi_array[0,1], moi_array[1,0] = xy, xy
    moi_array[0,2], moi_array[2,0] = xz, xz
    moi_array[1,2], moi_array[2,1] = yz, yz

    # 對角化
    eig_val, eig_vec = np.linalg.eig(moi_array)
    norm_eig_val = eig_val/np.max(eig_val)
    return norm_eig_val




# %%
fc_swc_file = './data/selected_data/FC/'
fc_add_file = './data/selected_data/FC_add/'
em_swc_file = './data/selected_data/EM/'

# load labeled_dataframe
df_test = pd.read_csv('./train_test_split/test_split_0_D1-D6.csv')
df_train = pd.read_csv('./train_test_split/train_split_0_D1-D6.csv')

labeled_df = pd.concat([df_test, df_train], axis=0)

# Add 隨機亂配的pairs
fc_nrn_lst = os.listdir(fc_swc_file)
em_nrn_lst = os.listdir(em_swc_file)

# clear swc
fc_nrn_lst = [i[:-4] for i in fc_nrn_lst if i[-4:] == '.swc']
em_nrn_lst = [i[:-4] for i in em_nrn_lst if i[-4:] == '.swc']

# shuffle lst
np.random.shuffle(fc_nrn_lst)
np.random.shuffle(em_nrn_lst)

# limit length
min_len = min(len(fc_nrn_lst), len(em_nrn_lst), 3000)
fc_nrn_lst = fc_nrn_lst[:min_len]
em_nrn_lst = em_nrn_lst[:min_len]

score = [0] * min_len
label = [0] * min_len
shuffle_df = pd.DataFrame({'fc_id': fc_nrn_lst, 'em_id': em_nrn_lst, 'score':score, 'label':label})

labeled_df = pd.concat([labeled_df, shuffle_df], axis=0)

cm_dist, moi_diff = [], []

for i in range(len(labeled_df)):
    
    fc_path = os.path.join(fc_swc_file, str(labeled_df['fc_id'].iloc[i]) + '.swc')
    if os.path.exists(fc_path) == False:
        fc_path = os.path.join(fc_add_file, str(labeled_df['fc_id'].iloc[i]) + '.swc')

    em_path = os.path.join(em_swc_file, str(labeled_df['em_id'].iloc[i]) + '.swc')
    
    fc_np = load_swc(fc_path)
    em_np = load_swc(em_path)

    d_cm = np.linalg.norm(find_CM(fc_np) - find_CM(em_np),2)
    cm_dist.append(d_cm)

    d_moi = np.linalg.norm(calculate_moi(fc_np) - calculate_moi(em_np),2)
    moi_diff.append(d_moi)

labeled_df['cm_dist'] = cm_dist
labeled_df['moi_diff'] = moi_diff

labeled_df.to_csv('./swc_analysis/labeled_df.csv', index=False)
# %%
pos = labeled_df[labeled_df['label']>=0.5]
neg = labeled_df[labeled_df['label']<0.5]

plt.scatter(neg['cm_dist'], neg['moi_diff'], s=8, linewidth=0)
plt.scatter(pos['cm_dist'], pos['moi_diff'],c='r', s=8, linewidth=0)
plt.savefig('./swc_analysis/cm_moi.png', dpi=100, bbox_inches='tight')
plt.show()

# box plot
sns.boxplot(data=[np.array(pos['cm_dist']), np.array(neg['cm_dist'])])
plt.show()
sns.boxplot(data=[np.array(pos['moi_diff']), np.array(neg['moi_diff'])])
plt.show()
# %%
