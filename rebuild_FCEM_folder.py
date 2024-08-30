# %%
'''
注意 執行此段程序將清除指定路徑文件夾。前請注意備份相應文件夾

'''
from collections import defaultdict
from util import load_pkl
import pandas as pd
import shutil
import os
import numpy as np
import pickle

# %%clear folder list
clear_folder_lst = ['data/converted_data', './data/mapping_data1/', './data/mapping_data2/']


# Clear files in folder
for folder in clear_folder_lst:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
# %% 將有標註神經swc複製進EM、FC文件夾

# 讀取所有標註資料
train_nrn_path = './train_test_split/train_split_0_D1-D6.csv'
test_nrn_path = './train_test_split/test_split_0_D1-D6.csv'

train_df = pd.read_csv(train_nrn_path)
test_df = pd.read_csv(test_nrn_path)

total_df = pd.concat([train_df, test_df], axis=0)
fc_lst = list(set(total_df['fc_id']))
em_lst = list(set(total_df['em_id']))

# 將這些檔案複製進EM、FC
def copy2(target_folder, source_folder1, source_folder2, file_name_lst, ext='.swc'):
    not_found_lst = []
    for file_name in file_name_lst:
        source_path1 = os.path.join(source_folder1, str(file_name)+ext)
        source_path2 = os.path.join(source_folder2, str(file_name)+ext)
        if os.path.exists(source_path1):
            shutil.copy2(source_path1, target_folder)
        elif os.path.exists(source_path2):
            shutil.copy2(source_path2, target_folder)
        else:
            print('File not found:', file_name)
            not_found_lst.append(file_name)
    return not_found_lst

# FC
target_folder = './data/selected_data/FC/'
source_folder1 = './data/selected_data/FC_Original/'
source_folder2 = './data/selected_data/FC_add/'
not_found_fc = copy2(target_folder, source_folder1, source_folder2, fc_lst)
# EM
target_folder = './data/selected_data/EM/'
source_folder1 = './data/selected_data/EM_Original/'
source_folder2 = './'
not_found_em = copy2(target_folder, source_folder1, source_folder2, em_lst)
# %%
# 利用total_df, 從mapping_data2中讀取pkl並重建為初始格式

def collect_img_from(id_lst, map_folder, map_type='sn'):
    img_lst = []
    for i in id_lst:
        img = load_pkl(map_folder + str(i) + '.pkl')[map_type]
        # img_lst.append(np.transpose(img, (1, 2, 0)))
        img_lst.append(img)

    return np.array(img_lst)

def save_pkl(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


fc_img = collect_img_from(fc_lst, './data/mapping_data2/')
em_img = collect_img_from(em_lst, './data/mapping_data2/')

# 建立字典
mapping_dict = defaultdict(list)

for row in total_df.iterrows():
    fc_id = row[1].iloc[0]
    em_id = row[1].iloc[1]

    # rebuild old data structure
    # [fc_id, em_id, score(0), fc_img, em_img]
    data_lst = [str(fc_id), str(em_id), 0, fc_img[fc_lst.index(fc_id)], em_img[em_lst.index(em_id)]]

    # 以fc_id為key
    mapping_dict[fc_id].append(data_lst)

# 依據 keys作為文件名保存piclkle
save_name = 'mapping_data_sn_'
for key in mapping_dict.keys():
    save_path = './data/coor_orient_sn/' + save_name + str(key) + '.pkl'
    save_pkl(mapping_dict[key], save_path)

# %%檢查一致性
import matplotlib.pyplot as plt

original_path = './data/labeled_sn'
new_path = './data/coor_orient_sn'

new_fc_file_lst =  os.listdir(new_path)

def plot_compare(orignal_path, new_path, file_name):
    orignal_data = load_pkl(os.path.join(orignal_path, file_name))
    new_data = load_pkl(os.path.join(new_path, file_name))

    print('Original_FC:', orignal_data[0][0])
    print('New_FC:', new_data[0][0])

    for i in range(3):
        plt.subplot(2,3,i+1)
        plt.imshow(orignal_data[0][3][i,:,:],cmap='magma')
        plt.subplot(2,3,i+4)
        plt.imshow(new_data[0][3][i,:,:], cmap='magma')
    # plt.savefig('./Figure/CoorOrient_Map_'+orignal_data[0][0]+'.png', dpi=120, bbox_inches='tight')
    plt.show()

    print('Original_EM:', orignal_data[0][1])
    print('New_EM:', new_data[0][1])
    for i in range(3):
        plt.subplot(2,3,i+1)
        plt.imshow(orignal_data[0][4][i,:,:],cmap='magma')
        plt.subplot(2,3,i+4)
        plt.imshow(new_data[0][4][i,:,:], cmap='magma')
    # plt.savefig('./Figure/CoorOrient_Map_'+orignal_data[0][1]+'.png', dpi=120, bbox_inches='tight')
    plt.show()



# %%
