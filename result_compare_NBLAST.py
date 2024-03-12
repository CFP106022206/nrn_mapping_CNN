# %%
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import seaborn as sns
import pandas as pd
from util import load_pkl
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

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

    plt.savefig('./Figure/Loss_Curve_'+name+'.png', dpi=150, bbox_inches="tight")
    plt.show()


# %% load model
    
# 设置Seaborn样式
plt.style.use('default')

test_mode = 'cross'    #single: 指定單一 test data, cross: 使用cross validation 覆蓋完整 test data, 'nblast': 讀取nblast分數

test_set_num = 0       # 指定test_set 的特殊編號, 只有在 test_mode == 'single'中才要特別設置

cross_fold_num = 10      # cross validation 的 fold 數量, 只有在test_mode=='cross' 中才需要特別設置

# 如果為False, 則使用完整的test set, 如需要分析指定的test set(需在模型原本的Testing資料內), 輸入指定文件路徑, 此文件為包含指定fc_id, em_id的csv
selected_test_set = 0 #'./labeled_info/nblast_D2+D6_50as1.csv'

label_csv_name = './result/test_label_Annotator_D1-D6_'

nblast_correct_path = './labeled_info/nblast_D2+D5+D6_50as1.csv'

if test_mode == 'single':
    # load model predict test nrn set
    nrn_pair = pd.read_csv(label_csv_name+str(test_set_num)+'.csv')

    y_pred = np.array(nrn_pair['model_pred'])
    y_true = np.array(nrn_pair['label'])

    roc_color='darkorange'
    plot_title = 'Model Prediction'

elif test_mode == 'cross':
    train_losses, val_losses = [], []
    predict_result_lst = []
    for i in range(9,10):#cross_fold_num):
        predict_result = pd.read_csv(label_csv_name+str(i)+'.csv')
        predict_result_lst.append(predict_result)

        # 加载训练和验证历史记录
        history = load_pkl('./result/Train_History_Annotator_D1-D6_'+str(i)+'.pkl')
        train_losses.append(history['loss'])
        val_losses.append(history['val_loss'])

    # 組合所有結果
    predict_df = pd.concat(predict_result_lst, ignore_index=True)
    fc_lst = predict_df['fc_id'].tolist()
    em_lst = predict_df['em_id'].tolist()

    y_pred = predict_df['model_pred'].tolist()
    y_true = predict_df['label'].tolist()

    # 将历史记录转换为DataFrames
    train_losses_df = pd.DataFrame(train_losses)
    val_losses_df = pd.DataFrame(val_losses)


    generate_cross_loss_curve(train_losses_df, '#008367', 'Training')
    generate_cross_loss_curve(val_losses_df, '#467F7E', 'Validation')


    if selected_test_set:    # 若開啟 selected_test_set, 需要使用指定的test set, 因此需要有一份對應名單

        selected_test_set_df = pd.read_csv(selected_test_set)[['fc_id', 'em_id']]
        selected_test_set_df.drop_duplicates(subset=['fc_id','em_id'], inplace=True)

        # 只保留selected_test_set_df中的nrn pair
        predict_df = predict_df.merge(selected_test_set_df, on=['fc_id', 'em_id'], how='inner')

        # 更新篩選後的y_pred, y_true
        y_pred = predict_df['model_pred'].tolist()
        y_true = predict_df['label'].tolist()

    # 將列表轉換為numpy 數組以完成後續條件篩選操作
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    roc_color = 'lightseagreen'
    plot_title = 'Model Prediction'


elif test_mode == 'nblast':
    nblast_score_correct = pd.read_csv(nblast_correct_path)

    nblast_score_correct.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

    y_pred = nblast_score_correct['score'].to_numpy()
    y_true = nblast_score_correct['label'].to_numpy()

    roc_color='darkorange'
    plot_title = 'NBlast Score'

# binary label in y_true(for soft label)
y_true = np.array([1 if y > 0.5 else 0 for y in y_true])

# Normalized
pred_min = np.min(y_pred)  
pred_max = np.max(y_pred)
y_pred = (y_pred - pred_min)/(pred_max - pred_min)


# 提取预测值中属于每个类别的部分
y_pred_label0 = y_pred[y_true == 0]
y_pred_label1 = y_pred[y_true == 1]


# 繪製 violinplot

fig, ax = plt.subplots(figsize=(6, 5))

sns.violinplot(data=[y_pred_label0, y_pred_label1], inner="box", palette=['#001BC2', '#E90132']) # 箱線圖
        
# 设置透明度
for violin in ax.collections:
    violin.set_alpha(0.5)


# # 獲取自動設置的繪圖邊界
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()


# 畫原始數據點(有抖動)
# sns.stripplot(data=[y_pred_label0, y_pred_label1], jitter=0.06, size=2, zorder=1, palette=['#001BC2', '#E90132'])
sns.swarmplot(data=[y_pred_label0, y_pred_label1], size=2.5, zorder=1, palette=['#001BC2', '#E90132'])
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
# plt.title('Violin Plot')
plt.ylabel('Score (Normalized)')

plt.savefig('./Figure/Violin.png', dpi=150, bbox_inches="tight")
# 显示图像
plt.show()


# --------- 绘制核函數密度曲线 ---------

# 曲線版

# # sns.kdeplot 用于绘制核密度估计（Kernel Density Estimation, KDE）图
# sns.kdeplot(y_pred_label0, label="Label 0", color="blue", lw=2)
# sns.kdeplot(y_pred_label1, label="Label 1", color="red", lw=2)

# # 设置图标题和坐标轴标签
# plt.title(plot_title)
# plt.xlabel("Score (Normalized)")
# plt.ylabel("Density")

# # 显示图例
# plt.legend()

# # 保存
# plt.savefig('./Figure/Density_Curve', dpi=150, bbox_inches="tight")
# # 显示图
# plt.show()

# histogram 版本 (因為核密度曲線平滑處理，0～1範圍以外的部分)
sns.histplot(y_pred_label0, label="Label 0", color="blue", lw=0.5, alpha=0.6, bins=50)   
sns.histplot(y_pred_label1, label="Label 1", color="red", lw=0.5, alpha=0.6, bins=50)

# 设置图标题和坐标轴标签
plt.tick_params(axis='both', which='major', labelsize=12)

plt.title(plot_title)
plt.xlabel("Score (Normalized)")
plt.ylabel("Number")

# 显示图例
plt.legend()

# 保存
plt.savefig('./Figure/Density_Curve', dpi=150, bbox_inches="tight")
# 显示图
plt.show()


# ROC Cruve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color=roc_color, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(plot_title+' ROC Curve')
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


sns.set_theme(style="whitegrid")    # 背景灰色格線
plt.figure(figsize=(4,4))
plt.plot(threshold_lst,precision_lst,'*-',label='Precision',color='b')
plt.plot(threshold_lst,recall_lst,'d-',label='Recall',color='y')
plt.plot(threshold_lst,f1_lst,'o-', label='F1',color='r')
plt.legend()
plt.xlabel('Threshold')
# plt.ylabel('Score')
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






# %% Ranking analysis

top_k = 1

predict_df_clear = predict_df[['fc_id', 'em_id', 'label', 'model_pred']].copy()
# 二元化label(for soft label)
predict_df_clear['bi_label'] = [1 if x > 0.5 else 0 for x in predict_df['label']]

grouped = predict_df_clear.groupby('fc_id')

# 创建一个空字典来保存每个分组的新 DataFrame
dfs = {}

for name, group in grouped:
    sorted_group = group.sort_values(by='model_pred', ascending=False)
    dfs[name] = sorted_group

# 挑出dfs中值長度大於5
filtered_dfs = {k:v for k,v in dfs.items() if len(v) > top_k and 1 in v['bi_label'].values}

# top k accuracy
top_k_accuracy = []
for k in range(top_k,0,-1):
    correct = 0
    for key in filtered_dfs:
        # 只要前k個裡面有一個positive就算正確
        if filtered_dfs[key].iloc[0:k]['label'].sum() > 0:
            correct += 1
    print('Top', k, 'Accuracy:', correct/len(filtered_dfs))
    
    top_k_accuracy.append(correct/len(filtered_dfs))
# bar plot top k accuracy
plt.figure(figsize=(int(np.round(2.1+0.5*top_k)),4))
plt.ylim([0,1.1])
plt.grid(axis='y')
x_axis_name = ['Top 5', 'Top 4', 'Top 3', 'Top 2', 'Top 1']
x_axis_name_filt = x_axis_name[-top_k:]
plt.bar(x_axis_name_filt, top_k_accuracy, color='lightseagreen',linewidth=0)

# 加上數字標籤，以百分比形式
for x,y in enumerate(top_k_accuracy):
    plt.text(x, y+0.01, '{:.1%}'.format(y), ha='center', color='black', fontsize=12)

plt.ylabel('Accuracy')
# plt.title('Top k Accuracy')
plt.savefig('./Figure/Top_'+str(top_k)+'_Accuracy', dpi=150, bbox_inches='tight')
plt.show()

print('\nTotal:', len(filtered_dfs))

# %%
# Training Label 分佈分析
fig, ax1 = plt.subplots()

# kde plot for human label
sns.kdeplot(predict_df_clear['label'], label="KDE plot (Left y-axis)", lw=2, c='#28428A', ax=ax1)
ax1.set_ylabel('Density (KDE)')
ax1.set_ylim(0,2)
# Turn off vertical grid
ax1.grid(False)


# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the histogram on the second y-axis
# sns.histplot(predict_df_clear['label'], bins=25, color='#75fbd2', label='histogram (Right y-axis)', ax=ax2)
ax2.hist(predict_df_clear['label'], bins=50, color='lightseagreen', label='histogram (Right y-axis)')
ax2.set_ylabel('Number')
ax2.set_ylim(0,800)

# Get the lines and labels for legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Create the legend
plt.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Soft Label Distribution')
plt.savefig('./Figure/Soft_Label_Distribution', dpi=150, bbox_inches='tight')






# %% 添加人工標註神經腦區分佈
fc_brain_info1 = pd.read_csv('./data/neuropil_Ver1.csv')
fc_brain_info2 = pd.read_csv('./data/neuron1x1Coding_Ver2.csv')
em_brain_info = pd.read_csv('./data/neuron1x1Coding_FlyEM.csv')

def region_merge(region_name):
    # 刪除數字
    region_name = re.sub(r'\d+', '', region_name)
    # 合併左右腦（只保留腦區名稱）
    region_name = region_name.split('__')[0]
    return region_name

def rebuild_df(info_df):
    region_lst = []
    for region in info_df.columns:
        region_name = region_merge(region)
        if region_name not in region_lst:
            region_lst.append(region_name)
    
    edit_df = pd.DataFrame(columns=region_lst)
    for region in region_lst:
        # 找到所有同名腦區並相加
        region_value = info_df[info_df.columns[info_df.columns.str.contains(region)]].sum(axis=1)
        edit_df[region] = region_value

    return edit_df

fc_brain_info2 = rebuild_df(fc_brain_info2)


# Distribution of brain region in labeled fc
distribution_df = pd.DataFrame(columns=['brain_region', 'count'])
all_region_lst = fc_brain_info1.columns.tolist()
# Add fc_brain_info2
add_item = [x for x in fc_brain_info2.columns.tolist() if x not in all_region_lst]
all_region_lst += add_item

remove_item = ['nrn', 'neuron', 'volume', 'other']
all_region_lst = [x for x in all_region_lst if x not in remove_item]

distribution_df['brain_region'] = all_region_lst
distribution_df['count'] = [0]*len(all_region_lst)

fc_lst = predict_df['fc_id'].tolist()
for fc in fc_lst:
    fc_pass_region = fc_brain_info2[fc_brain_info2['neuron'] == fc]
    
    if len(fc_pass_region) == 0:
        fc_pass_region = fc_brain_info1[fc_brain_info1['nrn'] == fc]
        if len(fc_pass_region) == 0:
            print(fc)

    else:
        # 找到大於0的腦區
        for region in fc_pass_region.columns:
            region_part = fc_pass_region[region].values[0]
            # 排除非數字（nrn）和0
            if type(region_part) != str and region_part > 0:
                distribution_df.loc[distribution_df['brain_region'] == region, 'count'] += 1

# # 完整dataset分佈
# all_distribution = distribution_df.copy()
# fc_lst = 

# predict 準確率
predict_df['label_binary'] = [1 if x >= 0.5 else 0 for x in predict_df['label']]
predict_df['correct'] = predict_df['label_binary'] == predict_df['model_pred_binary']

# 畫histogram
plt.figure(figsize=(12,9))
sns.barplot(y='brain_region', x='count', data=distribution_df, hue='count', palette='dark:#5A9_r', legend=False)
# 加上數字標籤
for x,y in enumerate(distribution_df['count']):
    plt.text(5, x+0.2, '{:.0f}'.format(y), ha='left', color='#CFD2D2', fontsize=10)
# x軸刻度調整
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(25))

plt.savefig('./Figure/Labeled_Region_Distribution', dpi=300, bbox_inches='tight')
plt.show()

# %%  分析兩代資料差異
# 整理fc_brain_info
v1_region_lst = fc_brain_info1.columns.tolist()
v2_region_lst = fc_brain_info2.columns.tolist()

# 去除數字
v2_region_lst = [re.sub(r'\d+', '', x) for x in v2_region_lst]
# 合併左右腦
v2_region_merge = []
for region in v2_region_lst:
    if '__' in region:
        region_name = region.split('__')[0]  # 下劃線前的部分是腦區名稱
        v2_region_merge.append(region_name)
    else:
        v2_region_merge.append(region)
    
# 去除重複元素
v2_region_merge = [x for i, x in enumerate(v2_region_merge) if v2_region_merge.index(x) == i]
# remove 'volumn' 'other'
remove_item = ['neuron', 'volume', 'other']
v2_region_merge = [x for x in v2_region_merge if x not in remove_item]
v2_region_merge.insert(0,'nrn')

# 檢查v1 v2是否一致
for i in v2_region_merge:
    if i not in v1_region_lst:
        print(i)
for i in v1_region_lst:
    if i not in v2_region_merge:
        print(i)


# %%
