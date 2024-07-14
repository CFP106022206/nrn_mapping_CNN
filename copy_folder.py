# %%
import shutil
from tqdm import tqdm
import os

source = '/cluster/home/ming/Project_N/nrn_mapping_CNN/data/selected_data/FC'

target = '/cluster/home/ming/Project_N/Kuan_Ting/nrn_mapping_package/data/selected_data/FC'

source_lst = os.listdir(source)

for path in tqdm(source_lst):
    source_path = os.path.join(source,path)
    target_path = os.path.join(target, path)
    shutil.copy(source_path, target_path)