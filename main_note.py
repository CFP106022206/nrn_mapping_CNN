# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ranking_method as rk
from util import *
from config import *
from class_mapping import NrnMapping
from class_ranking import NrnRanking
from class_CNN import CNN

# Step 0 Augmentation
aug_num = 0  # 定义扩充的次数
vibration_amplitude = {"FC": 20, "EM": 20}  # 定义振动的幅度

# Step 1 Linear interpolation
interpolate_length = {"FC": 2.5, "EM": 2.5}  # 定义插值的长度

# Step 2 Define coordinate system
weighting_keys_c = ["unit", "sn", "rsn"]  # 定义坐标系的权重关键词
max_sn = np.inf  # 定义在映射过程中可以接受的Strahler number的最大值
grid_num = 50  # 定义地图的网格数量
ignore_soma = False  # 是否忽略soma分支
normalization_of_sn = True  # 是否将2D地图中的Strahler number归一化到1
normalization_of_moi = True  # 是否将惯性矩的特征值归一化到其最大值

# Step 3 Match pairs of neurons
weighting_keys_m = ["unit", "sn", "rsn"]  # 定义神经元配对的权重关键词
coordinate_selection = "target-orientation"  # 定义坐标选择方法
target_list = ["FC"]
candidate_list = ["EM"]
threshold_of_exchange = 0.0  # 定义考虑交换主轴的阈值
threshold_of_nI = 400  # 定义通过惯性矩的阈值来选择神经元配对
threshold_in = np.cos(np.pi*90/180)  # 定义内积的阈值
threshold_of_distance = 10000  # 定义EM数据和FC数据之间的距离阈值

# Step 4 Score and rank the selected pairs
cluster = False  # 是否简化Strahler number
cluster_num = 3  # 定义神经元节点分组的组数
ranking_method = rk.mask_test_gpu  # 定义评分和排名的方法

# 创造振动数据
swc_vibration(config_path,
              num=aug_num, vibrate_amplitude=vibration_amplitude)

# 将swc文件转换为特定的数据格式（使用线性插值）
clear = False
overwrite = False
plot = False
file_lst = load_swc(config_path,
                    clear, overwrite, interpolate_length, plot)

# 定义坐标系
overwrite = False
Map = NrnMapping(config_path, file_lst, weighting_keys_c, grid_num)
Map.batch_coordinate_process(overwrite, max_sn,
                             normalization_moi=normalization_of_moi, normalization_sn=normalization_of_sn,
                             ignore_soma=ignore_soma)

# 枚举可能的坐标组合并创建其映射数据
overwrite = False
Map.batch_mapping_process(overwrite)

# 设置阈值并匹配可能的神经元对
overwrite = True
Match = NrnRanking(config_path, grid_num, weighting_keys_m, ranking_method, coordinate_selection)
Match.batch_matching_process(overwrite, target_list, candidate_list,
                             threshold_of_nI, threshold_of_distance, threshold_in)

# 对每对神经元进行排名，并输出每对中最佳坐标组合的图像
overwrite = True
map_data_saving = False
plot = False
Match.batch_ranking_process(overwrite, map_data_saving, plot)
