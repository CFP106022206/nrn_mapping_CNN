# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_swc(swc_path):
    # load swc file
    swc = pd.read_csv(swc_path, sep='\s+', comment='#', header=None, names=['type', 'x', 'y', 'z', 'R', 'Parent'])
    return swc


def find_CM(swc_path):
    if swc_path[-4:] != '.swc':
        swc_path = swc_path + '.swc'

    swc = load_swc(swc_path)

    swc_cm = np.mean(swc[['x', 'y', 'z']].values, axis=0)

    return swc_cm


def calculate_moi(swc_path):
    if swc_path[-4:] != '.swc':
        swc_path = swc_path + '.swc'

    swc = load_swc(swc_path)
    swc_cm = find_CM(swc_path)
    # calculate moment of inertia
    moi_array = np.zeros((3,3))
    xx, yy, zz = 0,0,0
    xy, xz, yz = 0,0,0
    for i in range(len(swc)):
        xx += (swc['y'].iloc[i] - swc_cm[1])**2 + (swc['z'].iloc[i] - swc_cm[2])**2
        yy += (swc['x'].iloc[i] - swc_cm[0])**2 + (swc['z'].iloc[i] - swc_cm[2])**2
        zz += (swc['x'].iloc[i] - swc_cm[0])**2 + (swc['y'].iloc[i] - swc_cm[1])**2

        xy += -(swc['x'].iloc[i] - swc_cm[0])*(swc['y'].iloc[i] - swc_cm[1])
        xz += -(swc['x'].iloc[i] - swc_cm[0])*(swc['z'].iloc[i] - swc_cm[2])
        yz += -(swc['y'].iloc[i] - swc_cm[1])*(swc['z'].iloc[i] - swc_cm[2])

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

cm_dist, moi_diff = [], []
for i in range(len(labeled_df)):
    
    fc_path = os.path.join(fc_swc_file, str(labeled_df['fc_id'].iloc[i]) + '.swc')
    if os.path.exists(fc_path) == False:
        fc_path = os.path.join(fc_add_file, str(labeled_df['fc_id'].iloc[i]) + '.swc')

    em_path = os.path.join(em_swc_file, str(labeled_df['em_id'].iloc[i]) + '.swc')
    
    d_cm = np.linalg.norm(find_CM(fc_path) - find_CM(em_path),2)
    cm_dist.append(d_cm)

    d_moi = np.linalg.norm(calculate_moi(fc_path) - calculate_moi(em_path),2)
    moi_diff.append(d_moi)

labeled_df['cm_dist'] = cm_dist
labeled_df['moi_diff'] = moi_diff

# %%
pos = labeled_df[labeled_df['label']>=0.5]
neg = labeled_df[labeled_df['label']<0.5]

plt.scatter(neg['cm_dist'], neg['moi_diff'], s=8, linewidth=0)
plt.scatter(pos['cm_dist'], pos['moi_diff'],c='r', s=8, linewidth=0)
plt.show()
# %%
