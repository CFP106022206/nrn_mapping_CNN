
# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from util import load_pkl
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc



# %% load model

test_mode = 'cross'    #single: 指定單一 test data, cross: 使用cross validation 覆蓋完整 test data, 'nblast': 讀取nblast分數

test_set_num = 98       # 指定test_set 的特殊編號, 只有在 test_mode == 'single'中才要特別設置

cross_fold_num = 5      # cross validation 的 fold 數量, 只有在test_mode=='cross' 中才需要特別設置

data_range = 'D6'

use_final = False      # 如果True，使用最後階段的預測結果，如果False，使用第一階段的預測結果

if use_final:
    label_csv_name = './result/final_label_model_D1-'+data_range+'_'
else:
    label_csv_name = './result/test_label_Annotator_D1-'+data_range+'_'

nblast_correct_path = './data/nblast_D2_D5_correct.csv'

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

    roc_color = 'lightseagreen'
    plot_title = 'CNN Score'

elif test_mode == 'nblast':
    nblast_score_correct = pd.read_csv(nblast_correct_path)

    y_pred = nblast_score_correct['score'].to_numpy()
    y_true = nblast_score_correct['label'].to_numpy()

    roc_color='darkorange'
    plot_title = 'NBlast Score'

# Normalized
pred_min = np.min(y_pred)  
pred_max = np.max(y_pred)
y_pred = (y_pred - pred_min)/(pred_max - pred_min)


# 提取预测值中属于每个类别的部分
y_pred_label0 = y_pred[y_true == 0]
y_pred_label1 = y_pred[y_true == 1]

# 设置Seaborn样式
plt.style.use('default')
sns.set(style="whitegrid")



# 繪製 violinplot


fig, ax = plt.subplots(figsize=(7, 7))

sns.violinplot(data=[y_pred_label0, y_pred_label1], inner="box", palette=['#001BC2', '#E90132']) # 箱線圖
        
# 设置透明度
for violin in ax.collections[::2]:
    violin.set_alpha(0.6)


# # 獲取自動設置的繪圖邊界
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()


# 畫原始數據點(有抖動)
sns.stripplot(data=[y_pred_label0, y_pred_label1], jitter=0.04, size=4, zorder=1, palette=['#001BC2', '#E90132'])
plt.xticks([0, 1], ['Label = 0', 'Label = 1'])

# 计算平均数
averages = [np.mean(p) for p in [y_pred_label0, y_pred_label1]]

# 在小提琴图上标注平均数
for i, avg in enumerate(averages):
    # ax.scatter(i, avg, marker='o', color='yellow', s=15, zorder=3)
    ax.text(i, y_lim[0]+0.02, f"Avg = {avg:.2f}", horizontalalignment='center', fontsize=12, color='black')

# 重新設置繪圖邊界 (默認設置會被stripplot帶偏)
ax.set_ylim(y_lim)
ax.set_xlim(x_lim)

# 添加标题和轴标签
plt.title('Result of D1-'+data_range+' models')
plt.ylabel('Score (Normalized)')

plt.savefig('./Figure/Violin.png', dpi=150, bbox_inches="tight")
# 显示图像
plt.show()


# 绘制密度曲线
# sns.kdeplot 用于绘制核密度估计（Kernel Density Estimation, KDE）图
sns.kdeplot(y_pred_label0, label="Label 0", color="blue", lw=2)
sns.kdeplot(y_pred_label1, label="Label 1", color="red", lw=2)

# 设置图标题和坐标轴标签
plt.title(plot_title)
plt.xlabel("Score (Normalized)")
plt.ylabel("Density")

# 显示图例
plt.legend()

# 保存
plt.savefig('./Figure/Density_Curve', dpi=150, bbox_inches="tight")
# 显示图
plt.show()


# ROC Cruve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color=roc_color, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(plot_title)
plt.legend(loc="lower right")
plt.savefig('./Figure/ROC_Curve', dpi=150, bbox_inches="tight")

plt.show()


threshold_lst = np.arange(0,1,0.05)
# threshold = 0.30
def gen_conf_matrix(y_true, y_pred, threshold):

    y_pred_binary = []
    for score in y_pred:
        if score > threshold:
            y_pred_binary.append(1)
        else:
            y_pred_binary.append(0)

    conf_matrix = confusion_matrix(y_true.tolist(), y_pred_binary, labels=[1,0])
    return y_pred_binary, conf_matrix

precision_lst, recall_lst, f1_lst = [],[],[]
for threshold in threshold_lst:
    y_pred_binary, conf_matrix = gen_conf_matrix(y_true, y_pred, threshold)
    # Precision and recall
    precision = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0])
    recall = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[0,1])
    # print("Precision:", precision)
    # print("Recall:", recall)

    # F1 Score
    result_f1_score = f1_score(y_true, y_pred_binary, average=None)
    # print('F1 Score for Neg:', result_f1_score[0])
    # print('F1 Score for Pos:', result_f1_score[1])
    precision_lst.append(precision)
    recall_lst.append(recall)
    f1_lst.append(result_f1_score[1])

plt.plot(threshold_lst,precision_lst,'o-',label='Precision')
plt.plot(threshold_lst,recall_lst,'d-',label='Recall')
plt.plot(threshold_lst,f1_lst,'*-', label='F1')
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.savefig('./Figure/Threshold_Curve', dpi=150, bbox_inches='tight')
plt.show()


# Find Best F1 score

best_result_idx = f1_lst.index(max(f1_lst))
threshold = threshold_lst[best_result_idx]
print('Best F1 at threshold = ', threshold)

print('Precision: ', precision_lst[best_result_idx])
print('Recall: ', recall_lst[best_result_idx])
print('F1: ', f1_lst[best_result_idx])
print(gen_conf_matrix(y_true, y_pred, threshold=threshold)[1])

# %% 混淆矩陣分析
if test_mode == 'single':
    annotator_result = load_pkl('./result/Test_Result_Annotator_D1-'+data_range+'_'+str(test_set_num)+'.pkl')
    final_result = load_pkl('./result/Final_stage_Result_model_D1-'+data_range+'_'+str(test_set_num)+'.pkl')
    
    print('Annotator Confusion Matrix\n', annotator_result['conf_matrix'])
    print('Annotator Precision, Recall, F1-Score\n', np.round(annotator_result['Precision'],2), np.round(annotator_result['Recall'],2), np.round(annotator_result['F1_pos'],2))

    print('Final Confusion Matrix\n', final_result['conf_matrix'])
    print('Final Precision, Recall, F1-Score\n', np.round(final_result['Precision'],2), np.round(final_result['Recall'],2), np.round(final_result['F1_pos'],2))

# %%
