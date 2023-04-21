'''
1, Make Train/Test Set from D1~D4 or D1~D5
2, Load Each Set and train model
3, Transfer Big Model
4, Result Analysis
5, Iterative self-labeling
6, Transfer Big Model...
'''

# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from util import load_pkl


# %% 读取三个阶段结果档案：annotator, first stage, second stage

cross_fold_num = 3
data_range = 'D5'

annotator_model_name = 'Annotator_D1-' + data_range + '_'
model_name = 'model_D1-'+data_range+'_'

annotator_result = './result/Test_Result_' + annotator_model_name
second_stage_result = './result/Second_stage_Result_' + model_name
final_stage_result = './result/Final_stage_Result_' + model_name


def result_collector(file_path, cross_fold_num):

    Precisions, Recalls, F1_pos = [], [], []
    conf_matrixs = np.zeros((cross_fold_num, 2, 2))

    for i in range(cross_fold_num):
        result_path = file_path + str(i) + '.pkl'
        result = load_pkl(result_path)
        conf_matrixs[i] = result['conf_matrix']
        Precisions.append(result['Precision'])
        Recalls.append(result['Recall'])
        F1_pos.append(result['F1_pos'])
    
    return conf_matrixs, Precisions, Recalls, F1_pos

conf_matrixs_annotator, Precisions_annotator, Recalls_annotator, F1_pos_annotator = result_collector(annotator_result, cross_fold_num)
conf_matrixs_second_stage, Precisions_second_stage, Recalls_second_stage, F1_pos_second_stage = result_collector(second_stage_result, cross_fold_num)
conf_matrixs_final_stage, Precisions_final_stage, Recalls_final_stage, F1_pos_final_stage = result_collector(final_stage_result, cross_fold_num)

# %% 将数据组合成一个列表
Precision_set = [Precisions_annotator, Precisions_second_stage, Precisions_final_stage]
Recall_set = [Recalls_annotator, Recalls_second_stage, Recalls_final_stage]
F1_pos_set = [F1_pos_annotator, F1_pos_second_stage, F1_pos_final_stage]


# 创建标签
labels = ['Annotator', 'Second Stage', 'Final Stage']

# 设置 Seaborn 风格
plt.style.use('default')
sns.set(style="whitegrid")


# 小提琴图
def generate_violinplot(data_set, y_label):

    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(data=data_set, inner="box", palette='Set2') # 箱線圖
            
    # 设置透明度
    for violin in ax.collections[::2]:
        violin.set_alpha(0.6)
    
    
    # # 獲取自動設置的繪圖邊界
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()


    # 畫原始數據點(有抖動)
    sns.stripplot(data=data_set, jitter=0.09, color='black', size=4, zorder=1)


    plt.xticks([0, 1, 2], labels)

    # 计算平均数
    averages = [np.mean(p) for p in data_set]

    # 在小提琴图上标注平均数
    for i, avg in enumerate(averages):
        # ax.scatter(i, avg, marker='o', color='yellow', s=15, zorder=3)
        ax.text(i, y_lim[0]+0.02, f"Avg = {avg:.2f}", horizontalalignment='center', fontsize=12, color='black')

    # 重新設置繪圖邊界 (默認設置會被stripplot帶偏)
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)

    # 添加标题和轴标签
    plt.title('Result of D1-'+data_range+' models')
    plt.ylabel(y_label)

    plt.savefig('./Figure/3_stage_'+y_label+'_violin.png', dpi=300, bbox_inches="tight")
    # 显示图像
    plt.show()




# Precision 
generate_violinplot(Precision_set, 'Precision')


# Recall
generate_violinplot(Recall_set, 'Recall')



# F1 Positive
generate_violinplot(F1_pos_set, 'F1_Score')



# %% 箱线图



# %% 计算各指标平均值

# average_precision = []
# for i in Precision_set:
#     average_precision.append(np.average(i))
# print('\nAverage Precision', np.round(average_precision,2))


# average_recall = []
# for i in Recall_set:
#     average_recall.append(np.average(i))
# print('\nAverage Recall', np.round(average_recall,2))


# average_f1 = []
# for i in F1_pos_set:
#     average_f1.append(np.average(i))
# print('\nAverage F1', np.round(average_f1,2))

def average_conf_matrix(conf_matrix_set):
    conf_average = np.zeros((2,2))
    for i in conf_matrix_set:
        conf_average += i
    conf_average = conf_average/conf_matrix_set.shape[0]
    return conf_average

conf_average_annotator = average_conf_matrix(conf_matrixs_annotator)
conf_average_first_stage = average_conf_matrix(conf_matrixs_second_stage)
conf_average_second_stage = average_conf_matrix(conf_matrixs_final_stage)

print('\nAverage conf_matrix: Annotator\n', np.round(conf_average_annotator))
print('Average Precision, Recall, F1: Annotator\n', np.round(np.mean(Precisions_annotator),2), np.round(np.mean(Recalls_annotator), 2), np.round(np.mean(F1_pos_annotator), 2))

print('\nAverage conf_matrix: Second_stage\n', np.round(conf_average_first_stage))
print('Average Precision, Recall, F1: Second_stage\n', np.round(np.mean(Precisions_second_stage), 2), np.round(np.mean(Recalls_second_stage), 2), np.round(np.mean(F1_pos_second_stage), 2))

print('\nAverage conf_matrix: Final_stage\n', np.round(conf_average_second_stage))
print('Average Precision, Recall, F1: Final_stage\n', np.round(np.mean(Precisions_final_stage), 2), np.round(np.mean(Recalls_final_stage), 2), np.round(np.mean(F1_pos_final_stage), 2))

# calculate F1 score
def calculate_f1(conf_matrix):
    precision = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[1,0])
    recall = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    return 2*precision*recall/(precision+recall)

target = conf_matrixs_annotator

f1_score_lst = []
for conf_matrix in target:
    f1_score_lst.append(calculate_f1(conf_matrix))
print(f1_score_lst.index(np.max(f1_score_lst)))
print(target[f1_score_lst.index(np.max(f1_score_lst))])
# %%


# 绘制训练损失和验证损失范围
def generate_cross_loss_curve(losses_df, curve_color, name):


    # 计算均值曲线和范围
    loss_mean = losses_df.mean(axis=0)
    loss_min = losses_df.min(axis=0)
    loss_max = losses_df.max(axis=0)


    fig, ax = plt.subplots(figsize=(6,4))

    ax.fill_between(losses_df.columns, loss_min, loss_max, color=curve_color, alpha=0.3, label=name+' Loss Range')

    # 绘制均值曲线
    ax.plot(loss_mean, color=curve_color, linewidth=2, label=name +' Loss Mean')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.savefig('./Figure/Loss_Curve_'+name+'.png', dpi=300, bbox_inches="tight")
    plt.show()


# Annotator
annotator_history_path = './result/Train_History_' + annotator_model_name

# 加载训练和验证历史记录
train_losses = []
val_losses = []

for i in range(cross_fold_num):
    history = load_pkl(annotator_history_path+str(i)+'.pkl')
    train_losses.append(history['loss'])
    val_losses.append(history['val_loss'])


# 将历史记录转换为DataFrames
train_losses_df = pd.DataFrame(train_losses)
val_losses_df = pd.DataFrame(val_losses)


generate_cross_loss_curve(train_losses_df, '#008367', 'Annotator_Train')
generate_cross_loss_curve(val_losses_df, '#467F7E', 'Annotator_Val')



# Second Stage
second_history_path = './result/Train_History_Second_stage_' + model_name

# 加载训练和验证历史记录
train_losses = []
val_losses = []

for i in range(cross_fold_num):
    history = load_pkl(second_history_path+str(i)+'.pkl')
    train_losses.append(history['loss'])
    val_losses.append(history['val_loss'])


# 将历史记录转换为DataFrames
train_losses_df = pd.DataFrame(train_losses)
val_losses_df = pd.DataFrame(val_losses)


generate_cross_loss_curve(train_losses_df, '#8C3A0D', 'Second_Stage_Train')
generate_cross_loss_curve(val_losses_df, '#844632', 'Second_Stage_Val')




# Final Stage
final_history_path = './result/Train_History_Final_' + model_name

# 加载训练和验证历史记录
train_losses = []
val_losses = []

for i in range(cross_fold_num):
    history = load_pkl(final_history_path+str(i)+'.pkl')
    train_losses.append(history['loss'])
    val_losses.append(history['val_loss'])


# 将历史记录转换为DataFrames
train_losses_df = pd.DataFrame(train_losses)
val_losses_df = pd.DataFrame(val_losses)


generate_cross_loss_curve(train_losses_df, '#103C7F', 'Final_Stage_Train')
generate_cross_loss_curve(val_losses_df, '#3C5283', 'Final_Stage_Val')




# %%
