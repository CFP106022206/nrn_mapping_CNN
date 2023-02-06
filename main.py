# Purpose:
# Map neuronal skeletons into two dimensional data format and group them

# %% #######################################################################################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ranking_method as rk
import shutil
from util import *
from config import *
from class_mapping import NrnMapping
from class_ranking import NrnRanking
from class_CNN import CNN

# %% Data Cleaning
clean_path = ['/home/yining_juan/Documents/nrn_mapping_CNN/data/converted_data',
              '/home/yining_juan/Documents/nrn_mapping_CNN/data/mapping_data1/',
              '/home/yining_juan/Documents/nrn_mapping_CNN/data/mapping_data2/',
              '/home/yining_juan/Documents/nrn_mapping_CNN/data/statistical_results/',
              '/home/yining_juan/Documents/nrn_mapping_CNN/data/selected_data/EM/',
              '/home/yining_juan/Documents/nrn_mapping_CNN/data/selected_data/FC/'
              ]

for path in clean_path:
    try:
        shutil.rmtree(path)
        os.mkdir(path)
    except:
        print("Path Not Found:", path)

# %% Pick swc data we need

def put_file_in_folder(folder, file):
    if not os.path.isdir(folder):
        os.makedirs(folder, mode=0o777) # absolute makedirs

    try:
        shutil.copy(file, folder)
    except:
        print('File not found:', file)

EM_filepath = '/home/yining_juan/Documents/nrn_mapping_CNN/data/DataBase/EM/'
FC_filepath = '/home/yining_juan/Documents/nrn_mapping_CNN/data/DataBase/FC/'

labeled_lst_path = '/home/yining_juan/Documents/nrn_mapping_CNN/data/D5_20221230.csv'
labeled_lst = pd.read_csv(labeled_lst_path)

EM_selected_path = '/home/yining_juan/Documents/nrn_mapping_CNN/data/selected_data/EM/'
FC_selected_path = '/home/yining_juan/Documents/nrn_mapping_CNN/data/selected_data/FC/'

EM_nrn_arr = np.unique(labeled_lst['em_id'].tolist())
FC_nrn_arr = np.unique(labeled_lst['fc_id'].tolist())

for em_nrn in EM_nrn_arr:
    put_file_in_folder(EM_selected_path, EM_filepath+str(em_nrn)+'.swc')

for fc_nrn in FC_nrn_arr:
    put_file_in_folder(FC_selected_path, FC_filepath+str(fc_nrn)+'.swc')



########################################################################################################################
# %% Parameters
########################################################################################################################

# Step 0 Augmentation
aug_num = 0
vibration_amplitude = {"FC": 20, "EM": 20}  # vibrate : amplitude --> 20 for fc ; 2000 for em

# Step 1 Linear interpolation
interpolate_length = {"FC": 2.5, "EM": 2.5}  # 0.5 for FC data, 50 for original EM data

# Step 2 Define coordinate system
# todo: key-independently overwrite
weighting_keys_c = ["unit", "sn", "rsn"]  # unit, sn, rsn
max_sn = np.inf  # the maximum acceptable value of Strahler number will appear in mapping
grid_num = 75  # the number of grids on each side of map
ignore_soma = False  # ignore the soma branch
normalization_of_sn = True  # normalizing the Strahler number in 2D-maps to 1
normalization_of_moi = True  # normalizing the eigenvalues of moment of inertia with its maximum value

# Step 3 Match pairs of neurons
weighting_keys_m = ["sn"]  # unit, sn, rsn --> "unit"
coordinate_selection = "target-orientation"  # "coordinate-orientation", "MOI-orientation", "target-orientation"
target_list = ["FC"]
candidate_list = ["EM"]
threshold_of_exchange = 0.0  # threshold of considering the exchange of principal axes
threshold_of_nI = 0.4  # threshold of choosing pairs of neurons by normalized inertia of moment
threshold_in = np.cos(np.pi*50/180)  # threshold of inner product
threshold_of_distance = 100  # threshold of distance between wrapping EM data and FC data

# Step 4 Score and rank the selected pairs
cluster = False  # simplify the Strahler number
cluster_num = 3  # the number of clusters which we group neuron nodes into
ranking_method = rk.mask_test_gpu  # customized design

########################################################################################################################
# Main Code
########################################################################################################################
# Additional: make vibration data
swc_vibration(config_path,
              num=aug_num, vibrate_amplitude=vibration_amplitude)

# STEP 1. convert the swc file into the specific data format (with linear interpolation)
clear = False
overwrite = False
plot = False
file_lst = load_swc(config_path,
                    clear, overwrite, interpolate_length, plot)

# %% STEP 2. A. define the coordinates by diagonalizing the matrix of moment of inertia
overwrite = False
Map = NrnMapping(config_path, file_lst, weighting_keys_c, grid_num)
Map.batch_coordinate_process(overwrite, max_sn,
                             normalization_moi=normalization_of_moi, normalization_sn=normalization_of_sn,
                             ignore_soma=ignore_soma)

# STEP 2. B. enumerate possible combinations of coordinates and create its mapping data
overwrite = False
Map.batch_mapping_process(overwrite)

# %% STEP 3. A. set thresholds and match possible pairs of neurons
overwrite = True
Match = NrnRanking(config_path, grid_num, weighting_keys_m, ranking_method, coordinate_selection)
Match.batch_matching_process(overwrite, target_list, candidate_list,
                             threshold_of_nI, threshold_of_distance, threshold_in)

# STEP 3. B. rank each pair of neurons and output the figure of the best combination of coordinates in each pair
overwrite = True
map_data_saving = True
plot = False
Match.batch_ranking_process(overwrite, map_data_saving, plot)

# The Result save at data/statistical_results/mapping_data_unit.pkl
# 格式為 [nrn_ID1, nrnID2, score, mapping1, mapping2]
# %%
