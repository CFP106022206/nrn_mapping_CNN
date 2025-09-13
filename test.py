# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from util import load_pkl

# %%
pic = load_pkl('720575940614026193.pkl')

# %%
df = pd.read_csv('data/selected_data/EMxFC_all_0_rk20.csv')
df = df.iloc[:10,:]
df.to_csv('data/selected_data/EMxFC_test.csv', index=False)
# %%
