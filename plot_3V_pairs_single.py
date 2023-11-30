# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from util import load_pkl


# %%
pic_path = './data/statistical_results/three_view_pic/'
fc_id = '5-HT1B-F-500015'
em_id = '5813034531'

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

plot_pair(pair_data)


# %%
