# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# 读取 CSV 文件
df = pd.read_csv("data/neuron1x1Coding.csv")

# D2
df_select = pd.read_csv('labeled_info/D2_conf.csv')
df_select2 = pd.read_csv('labeled_info/D6_conf.csv')
df_select = pd.concat([df_select, df_select2], axis=0)

#D5
# df_select = pd.read_csv('labeled_info/D5_conf.csv')

# 选择特定的列
fc_id = set(df_select['fc_id'].astype(str))
df_filtered = df[df['neuron'].astype(str).isin(fc_id)]
# 结果应该有150条FC神经

# 使用图表示各脑区占比
# 筛选出已知脑区数据区域
selected_columns = df_filtered.columns[1:-2]#[5,9,41,45]]
selected_array = df_filtered[selected_columns].values
# 对每一行取百分比
row_sums = selected_array.sum(axis=1, keepdims=True)
selected_array = selected_array / row_sums * 100

# heat map
plt.figure(figsize=(20, 50))

sns.heatmap(selected_array, cmap='magma', cbar=True)
plt.title('Neuropils Heatmap')

# x坐标标签为selected_columns
plt.xticks(ticks=np.arange(len(selected_columns)) + 0.5, labels=selected_columns, rotation=45, ha='right')
plt.ylabel('Neurons')
plt.savefig('./Figure/neuropils_heatmap.png', bbox_inches='tight', dpi=300)
plt.show()

target = ['mb_4_l', 'mb_4_r']

# %%

from __future__ import annotations

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm as mcm
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings('ignore')
# %%
