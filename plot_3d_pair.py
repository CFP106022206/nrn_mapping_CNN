'''
用來畫出指定pair的神經骨架三視圖
# 通常用來快速人工檢查
'''

# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import load_pkl
from tqdm import trange

def normalize(matrix):
    # 将矩阵中的数据等距压缩到 0 到 1 之间(若有超過範圍的數據)
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    normalized_matrix = (matrix - min_value) / (max_value - min_value)

    return normalized_matrix

# 判斷 TP, FP, FN, TN
def classification_result(ground_truth, predicted_label):
    return (ground_truth << 1) | predicted_label

'''
在这个高效的解决方案中，我们使用了位操作来组合 ground truth 标签和预测标签。这是解释这个表达式 (ground_truth << 1) | predicted_label 的详细步骤:

ground_truth << 1: 这是一个位左移操作。我们将 ground_truth 的二进制表示向左移动一位。
对于二进制数，左移一位相当于乘以 2。在我们的例子中, ground_truth 只能是 0 或 1, 
所以左移一位后的结果将是 0 或 2。

(ground_truth << 1) | predicted_label: 这是一个按位或操作。我们将前面步骤中得到的结果与预测标签进行按位或操作。



以下是所有可能组合的情况：

True Negative: ground_truth = 0, predicted_label = 0
(0 << 1) = 0
(0 | 0) = 0

False Positive: ground_truth = 0, predicted_label = 1
(0 << 1) = 0
(0 | 1) = 1

False Negative: ground_truth = 1, predicted_label = 0
(1 << 1) = 2
(2 | 0) = 2

True Positive: ground_truth = 1, predicted_label = 1
(1 << 1) = 2
(2 | 1) = 3

通过这种方法，我们可以将四种可能的情况表示为 0(True negative)、1(False positive)、2(False negative)和 3(True positive)
'''

# 与分类结果相关的字典，用于解释整数值
classification_dictionary = {
    0: "True negative",
    1: "False positive",
    2: "False negative",
    3: "True positive"
}



# %%

test_set_num = 0       # 指定test_set 的特殊編號, 只有在 test_mode == 'single'中才要特別設置

data_range = 'D5'


true_pos_path = './Figure/3_View/TruePos/'
false_pos_path = './Figure/3_View/FalsePos/'
true_neg_path = './Figure/3_View/TrueNeg/'
false_neg_path = './Figure/3_View/FalseNeg/'

# 检查文件夹路径是否存在
path_check_lst = [true_pos_path, false_pos_path, true_neg_path, false_neg_path]
for path in path_check_lst:
    if not os.path.exists(path):
        os.makedirs(path)

classification_path = {    
    0: true_neg_path,
    1: false_pos_path,
    2: false_neg_path,
    3: true_pos_path}

# load model predict test nrn set
nrn_pair = pd.read_csv('./result/final_label_model_D1-'+data_range+'_'+str(test_set_num)+'.csv')

# load map_data (3-view pkl file)
map_data = load_pkl('./data/mapping_data_sn.pkl')

# 從 map_data 建立 nrn-map_data 字典
data_dict = {}
for data in map_data:
    key = f'{data[0]}_{data[1]}'    # key = 'FC_nrn'_'EM_nrn'
    data_dict[key] = data           # 儲存該 pair 下的所有內容

# %%選擇要看的pair num
# plot_num = 0

def plot_3View(plot_num, data_dict, nrn_pair):
    key = f"{nrn_pair['fc_id'][plot_num]}_{nrn_pair['em_id'][plot_num]}"

    if key in data_dict:
        plt.figure(figsize=(10,5))

        fc_img = data_dict[key][3]
        em_img = data_dict[key][4]

        # normalized(到0～1之間)
        fc_img = normalize(fc_img)
        em_img = normalize(em_img)

        # 将维度顺序从 (3, 50, 50) 转换为 (50, 50, 3)
        fc_img = np.transpose(fc_img, (1, 2, 0))
        em_img = np.transpose(em_img, (1, 2, 0))
        
        plt.subplot(1,2,1)
        plt.imshow(fc_img)
        plt.axis('equal')
        plt.title(data_dict[key][0])    # fc_id

        plt.subplot(1,2,2)
        plt.imshow(em_img)
        plt.axis('equal')
        plt.title(data_dict[key][1])    # em_id
        
        # 设置整个 figure 的标题
        KT_score = data_dict[key][2]
        csv_score = nrn_pair['score'][plot_num]
        label = nrn_pair['label'][plot_num]
        pred_score = nrn_pair['model_pred'][plot_num]
        pred_binary = nrn_pair['model_pred_binary'][plot_num]
        plt.suptitle('Label: '+str(label)+'       pred_bin: '+str(pred_binary)+'       model_pred: '+str(np.round(pred_score,2))+'       KT_score: '+str(np.round(KT_score,2))+'       csv_score: '+str(np.round(csv_score,2)))

        # 判斷模型決策情形, 放入相應文件夾
        result = classification_result(label, pred_binary)
        save_path = classification_path[result]

        plt.savefig(os.path.join(save_path, key+'.png'), dpi=200, bbox_inches="tight")
        # plt.show()
    else:
        print('\n ! Not found nrn pair in map_data: ', plot_num)

for i in trange(len(nrn_pair)):
    plot_3View(i, data_dict, nrn_pair)

# %%
