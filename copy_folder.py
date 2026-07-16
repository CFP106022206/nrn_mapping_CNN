# %%
import shutil
from tqdm import tqdm
import os
import pandas as pd


# %%
source = '/cluster/home/ming/Project_N/Kuan_Ting/nrn_mapping_package/data/selected_data/EM'

target = '/cluster/home/ming/Project_N/nrn_mapping_CNN/data/selected_data/EM'

source_lst = os.listdir(source)

for path in tqdm(source_lst):
    source_path = os.path.join(source,path)
    target_path = os.path.join(target, path)
    shutil.copy(source_path, target_path)




    
# %%    Copy by list
copy_list = pd.read_csv('similarity_matrix/FTmodel_predict.csv')
copy_list = copy_list[['fc_id','em_id']]
fc_lst = copy_list['fc_id'].unique()

source = '/cluster/home/ming/Project_N/nrn_mapping_CNN/data/selected_data/FC_Original/'
source2 = '/cluster/home/ming/Project_N/nrn_mapping_CNN/data/seleted_data/FC_add/'
target = '/cluster/home/ming/Project_N/nrn_mapping_CNN/similarity_matrix/to_jinzhe_FC_swc/'

for fc in tqdm(fc_lst):
    source_path = os.path.join(source,fc+'.swc')
    if not os.path.exists(source_path):
        source_path = os.path.join(source2,fc+'.swc')
    target_path = os.path.join(target,fc+'.swc')
    shutil.copy(source_path, target_path)
# %%
