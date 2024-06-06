# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from util import load_pkl


# %%
pic_path = 'data/statistical_results/three_view_pic_paper_RSN/'
fc_id = 'Cha-F-000009'
em_id = '1078693835	'

output_path = './Figure/predict_3view/'

fc_path = pic_path + 'mapping_data_sn_' + fc_id + '.pkl'
pair_data = load_pkl(fc_path)   # list

if not os.path.exists(output_path):
    os.makedirs(output_path)

def plot_pair(pair_data):
    for pairs in pair_data:
        fc_id = pairs[0]
        em_id = pairs[1]

        fc_img = pairs[3]   # shape=(3, 50, 50)
        em_img = pairs[4]   # shape=(3, 50, 50)

        plt.figure(figsize=(11
                            ,6))
        for i in range(3):
            plt.subplot(2,3,i+1)
            plt.imshow(fc_img[i], cmap='magma')
            plt.colorbar()
            
            plt.subplot(2,3,i+4)
            plt.imshow(em_img[i], cmap='magma')
            plt.colorbar()

        plt.suptitle(f'{fc_id}_{em_id}')

        plt.savefig(output_path+f'{fc_id}_{em_id}.png', dpi=150, bbox_inches='tight')
        plt.show()

#生成論文使用之三種權重九宮格圖
pic_path_lst = ['data/statistical_results/three_view_pic_paper_UNIT/',
                'data/statistical_results/three_view_pic_paper_SN/',
                'data/statistical_results/three_view_pic_paper_RSN/']

fc_id = 'Cha-F-000009'      #生圖文件名以FC命名

map_data_path_lst = [pp + 'mapping_data_sn_' + fc_id + '.pkl' for pp in pic_path_lst]


fc_img_lst, em_img_lst = [], []
for path in map_data_path_lst:
    map_data = load_pkl(path)[0]    #裡面只有一項
    fc_id = map_data[0]     #每次循環都是一樣的
    em_id = map_data[1]
    fc_img_lst += [img for img in map_data[3]]
    em_img_lst += [img for img in map_data[4]]

fc_img = np.array(fc_img_lst)
em_img = np.array(em_img_lst)


def plot_all_weight_pics(img_id, img_np):
    row_labels = ['unit', 'SN', 'RSN']
    axis_labels = ['Y-Z', 'Z-X', 'X-Y']

    fig, axes = plt.subplots(3, 3, figsize=(6, 6), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(img_np[i], cmap='magma')
        ax.set_xticks([0, 50])
        ax.set_yticks([0, 50])
        ax.invert_yaxis()  # 反转y轴

    for ax, row_label in zip(axes[:,0], row_labels):
        ax.set_ylabel(row_label, rotation=90, size='large')

    for i, ax in enumerate(axes[-1]):
        ax.set_xlabel(axis_labels[i], size='large')

    # 加入color bar
    cbar_ax = fig.add_axes([0.97, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.suptitle("neuron ID : " + str(img_id))
    plt.savefig(output_path+f'{img_id}.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_all_weight_pics(fc_id, fc_img)
plot_all_weight_pics(em_id, em_img)

# plot_pair(pair_data)


# %%
