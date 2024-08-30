# Purpose:
# %% Map neuronal skeletons into two dimensional data format and group them

########################################################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ranking_method as rk
from util import *
from config import *
from class_mapping import NrnMapping
from class_ranking import NrnRanking
########################################################################################################################
# Parameters
########################################################################################################################
# Error Message
error_message = True

# target & candidate
target_list = ["FC"]
candidate_list = ["EM"]
for i in target_list:
    config_path["name"] += i
config_path["name"] += "_"
for i in candidate_list:
    config_path["name"] += i

# Step 1 Linear interpolation
interpolate_length = {"FC": 2.5, "EM": 2.5}  # minimum length of grid

# Step 2 Define coordinate system
# todo: key-independently overwrite
weighting_keys_c = ["sn"]  # unit, sn, rsn
max_sn = np.inf  # the maximum acceptable value of Strahler number will appear in mapping
grid_num = 50  # the number of grids on each side of map
ignore_soma = False  # ignore the soma branch
normalization_of_sn = True  # normalizing the Strahler number in 2D-maps to 1
normalization_of_moi = True  # normalizing the eigenvalues of moment of inertia with its maximum value

# Step 3 Match pairs of neurons
weighting_keys_m = ["sn"]  # unit, sn, rsn --> "unit"
coordinate_selection = "coordinate-orientation"  # "coordinate-orientation", "MOI-orientation", "target-orientation"
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

# STEP 1. convert the swc file into the specific data format (with linear interpolation)
clear = False
overwrite = False
plot = False
file_lst = load_swc(config_path,
                    clear, overwrite, interpolate_length, plot)

# STEP 2. A. define the coordinates by diagonalizing the matrix of moment of inertia
overwrite = False
Map = NrnMapping(config_path, file_lst, weighting_keys_c, grid_num)
Map.batch_coordinate_process(overwrite, max_sn,
                             normalization_moi=normalization_of_moi, normalization_sn=normalization_of_sn,
                             ignore_soma=ignore_soma)

# STEP 2. B. enumerate possible combinations of coordinates and create its mapping data
overwrite = False
Map.batch_mapping_process(overwrite)

# STEP 3. A. set thresholds and match possible pairs of neurons
overwrite = True
Match = NrnRanking(config_path, grid_num, weighting_keys_m, ranking_method, coordinate_selection)
Match.batch_matching_process(overwrite, target_list, candidate_list,
                             threshold_of_nI, threshold_of_distance, threshold_in)

# %% example
with open(config_path["stats"] + "match_dict" + config_path["name"] + ".pkl", "rb") as file:
    match_dict = pickle.load(file)["sn"]
# match_dict[target_id] --> [candidate1_id, candidate2_id, candidate3_id, ...]

# 建立pair的dataframe
source_lst, target_lst = [], []
for target_id, candidate_list in match_dict.items():
    if candidate_list:  # 保留candidate 列表不為空的鍵值
        for candidate_id in candidate_list:
            source_lst.append(candidate_id)
            target_lst.append(target_id)

match_df = pd.DataFrame({"source_id": source_lst, "target_id": target_lst})
match_df.to_csv('./data/statistical_results/match_list.csv', index=False)

# target_id_example = list(match_dict.keys())[0]
# print("target_id: ", target_id_example)
# candidate_list_example = match_dict[target_id_example]
# print("corresponding candidate_list: ", candidate_list_example)

# # load map
# with open(config_path["map2"] + target_id_example + ".pkl", "rb") as file:
#     target_map = pickle.load(file)["sn"]
# %%
