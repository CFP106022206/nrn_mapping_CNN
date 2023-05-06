
# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from util import load_pkl



# %% load model

test_mode = 'cross'    # Single: 指定單一 test data, Cross: 使用cross validation 覆蓋完整 test data

test_set_num = 98       # 指定test_set 的特殊編號, 只有在 test_mode == 'single'中才要特別設置

cross_fold_num = 5      # cross validation 的 fold 數量, 只有在test_mode=='cross' 中才需要特別設置

data_range = 'D5'

use_final = False      # 如果True，使用最後階段的預測結果，如果False，使用第一階段的預測結果

if use_final:
    label_csv_name = './result/final_label_model_D1-'+data_range+'_'
else:
    label_csv_name = './result/test_label_Annotator_D1-'+data_range+'_'


if test_mode == 'single':
    # load model predict test nrn set
    nrn_pair = pd.read_csv(label_csv_name+str(test_set_num)+'.csv')

    y_pred = nrn_pair['model_pred']
    y_true = nrn_pair['label']

elif test_mode == 'cross':
    y_pred, y_true = [], []
    for i in range(cross_fold_num):
        nrn_pair = pd.read_csv(label_csv_name+str(i)+'.csv')
        y_pred += nrn_pair['model_pred'].tolist()
        y_true += nrn_pair['label'].tolist()
    
    # 將列表轉換為numpy 數組以完成後續條件篩選操作
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

# 畫密度圖
# 提取预测值中属于每个类别的部分
y_pred_label0 = y_pred[y_true == 0]
y_pred_label1 = y_pred[y_true == 1]

# 设置Seaborn样式
plt.style.use('default')
sns.set(style="whitegrid")

# 绘制密度曲线
# sns.kdeplot 用于绘制核密度估计（Kernel Density Estimation, KDE）图
sns.kdeplot(y_pred_label0, label="Label 0", color="blue", lw=2)
sns.kdeplot(y_pred_label1, label="Label 1", color="red", lw=2)

# 设置图标题和坐标轴标签
plt.title("Classification with CNN")
plt.xlabel("Score")
plt.ylabel("Density")

# 显示图例
plt.legend()

# 保存
plt.savefig('./Figure/Density_Curve', dpi=150, bbox_inches="tight")
# 显示图
plt.show()


# %% 混淆矩陣分析
if test_mode == 'single':
    annotator_result = load_pkl('./result/Test_Result_Annotator_D1-'+data_range+'_'+str(test_set_num)+'.pkl')
    final_result = load_pkl('./result/Final_stage_Result_model_D1-'+data_range+'_'+str(test_set_num)+'.pkl')
    
    print('Annotator Confusion Matrix\n', annotator_result['conf_matrix'])
    print('Annotator Precision, Recall, F1-Score\n', np.round(annotator_result['Precision'],2), np.round(annotator_result['Recall'],2), np.round(annotator_result['F1_pos'],2))

    print('Final Confusion Matrix\n', final_result['conf_matrix'])
    print('Final Precision, Recall, F1-Score\n', np.round(final_result['Precision'],2), np.round(final_result['Recall'],2), np.round(final_result['F1_pos'],2))

# %%
