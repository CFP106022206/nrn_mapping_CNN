# %%
# import sys
# sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
import seaborn as sns
import pandas as pd
from util import load_pkl
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from util import load_pkl
from keras.models import *
from tqdm import tqdm

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



# %%
model_name = 'FineTune_miniLR'#'FineTune_miniLR_e7' #'Annotator' #'Fine_Tune_Model_150KnF_CoorOrient_'# 網頁版本模型結果

# 设置Seaborn样式
plt.style.use('default')

test_mode = 'cross'    #single: 使用單一模型產生的 test result csv, cross: 使用cross validation 覆蓋完整 data, 'nblast': 讀取nblast分數

test_set_num = 0       # 指定test_set 的特殊編號, 只有在 test_mode == 'single'中才要特別設置

cross_num = 10      # cross validation 的 fold 數量, 只有在test_mode=='cross' 中才需要特別設置

# 如果為False, 則使用完整的test set, 如需要分析指定的test set(需在模型原本的Testing資料內), 輸入指定文件路徑, 此文件為包含指定fc_id, em_id的csv
selected_test_set ='./labeled_info/D5_conf.csv' # './labeled_info/D2+D6_ID.csv'#'./labeled_info/D5_conf.csv'  #False

label_csv_name = f'./result/test_label_{model_name}_D1-D6_'
# label_csv_name = './result/predict_result/model_predict_'

nblast_path = './labeled_info/nblast_official_mean.csv'

if test_mode == 'single':
    # load model predict test nrn set
    nrn_pair = pd.read_csv(label_csv_name+str(test_set_num)+'.csv')

    y_pred = np.array(nrn_pair['model_pred'])
    y_true = np.array(nrn_pair['label'])

    roc_color='darkorange'
    plot_title = 'Model Predict score'

elif test_mode == 'cross':
    train_losses, val_losses = [], []
    predict_result_lst = []
    for i in range(cross_num):
        predict_result = pd.read_csv(label_csv_name+str(i)+'.csv')
        predict_result_lst.append(predict_result)

        # 加载训练和验证历史记录
        history = load_pkl(f'./result/Train_History_{model_name}_D1-D6_{i}.pkl')
        train_losses.append(history['loss'])
        val_losses.append(history['val_loss'])

    predict_df = pd.concat(predict_result_lst, ignore_index=True)

    # 将历史记录转换为DataFrames
    train_losses_df = pd.DataFrame(train_losses)
    val_losses_df = pd.DataFrame(val_losses)

    # 畫train loss 曲線
    generate_cross_loss_curve(train_losses_df, '#008367', 'Training')
    generate_cross_loss_curve(val_losses_df, '#467F7E', 'Validation')


    if selected_test_set:    # 若開啟 selected_test_set, 需要使用指定的test set, 因此需要有一份對應名單
        selected_test_set_df = pd.read_csv(selected_test_set)[['fc_id', 'em_id']]
        selected_test_set_df.drop_duplicates(subset=['fc_id','em_id'], inplace=True)

        # 只保留selected_test_set_df中的nrn pair
        predict_df = predict_df.merge(selected_test_set_df, on=['fc_id', 'em_id'], how='inner')

    y_pred = predict_df['model_pred'].to_numpy()
    y_true = predict_df['label'].to_numpy()

    roc_color = 'lightseagreen'
    plot_title = 'Model Predict score'


elif test_mode == 'nblast':
    nblast_score = pd.read_csv(nblast_path)

    nblast_score.drop_duplicates(subset=['fc_id','em_id'], inplace=True) # 删除重复

    if selected_test_set:    # 若開啟 selected_test_set, 需要使用指定的test set, 因此需要有一份對應名單
        selected_test_set_df = pd.read_csv(selected_test_set)[['fc_id', 'em_id']]
        selected_test_set_df.drop_duplicates(subset=['fc_id','em_id'], inplace=True)

        # 只保留selected_test_set_df中的nrn pair
        nblast_score = nblast_score.merge(selected_test_set_df, on=['fc_id', 'em_id'], how='inner')

    # 更新篩選後的y_pred, y_true
    y_pred = nblast_score['similarity score'].to_numpy()
    y_true = nblast_score['label'].to_numpy()

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


# ROC Cruve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Find the ROC point closest to the top-left corner (0, 1)
# (Common heuristic: minimum Euclidean distance to (0,1))
finite_mask = np.isfinite(thresholds)
fpr_f = fpr[finite_mask]
tpr_f = tpr[finite_mask]
thr_f = thresholds[finite_mask]
dist_to_topleft = np.sqrt((fpr_f - 0.0) ** 2 + (tpr_f - 1.0) ** 2)
best_roc_idx = int(np.argmin(dist_to_topleft))
roc_best_threshold = float(thr_f[best_roc_idx])
roc_best_fpr = float(fpr_f[best_roc_idx])
roc_best_tpr = float(tpr_f[best_roc_idx])
print(f'ROC closest-top-left threshold = {roc_best_threshold:.4f} (FPR={roc_best_fpr:.4f}, TPR={roc_best_tpr:.4f})')

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color=roc_color, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.scatter([roc_best_fpr], [roc_best_tpr], s=90, c='#A62C3A', edgecolors='white', linewidths=1.2,
            zorder=5, label=f'Closest to (0,1) thr={roc_best_threshold:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(plot_title+' ROC Curve')
plt.legend(loc="lower right")
plt.savefig('./Figure/ROC_Curve', dpi=150, bbox_inches="tight")

plt.show()

def gen_conf_matrix(y_true, y_pred, threshold):

    y_pred_binary = []
    for score in y_pred:
        if score > threshold:
            y_pred_binary.append(1)
        else:
            y_pred_binary.append(0)

    conf_matrix = confusion_matrix(y_true.tolist(), y_pred_binary, labels=[1,0])
    return y_pred_binary, conf_matrix


threshold_lst = np.arange(0,1,0.01)
# threshold = 0.30

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


# Find Best F1 score

print('Use closest point on ROC at threshold = ', roc_best_threshold)
y_pred_binary, conf_matrix = gen_conf_matrix(y_true, y_pred, threshold=roc_best_threshold)
precision = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0])
recall = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[0,1])
pos_f1_score = f1_score(y_true, y_pred_binary, average=None)[1]
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', pos_f1_score)
print('Confusion Matrix:\n', conf_matrix)


# sns.set_theme(style="whitegrid")    # 背景灰色格線
plt.figure(figsize=(5,4))
# 五個留一個作圖
threshold_sample = threshold_lst[::5]
precision_sample = precision_lst[::5]
recall_sample = recall_lst[::5]
f1_sample = f1_lst[::5]

plt.plot(threshold_sample,precision_sample,'*-',label='Precision',color='navy', alpha=0.6)
plt.plot(threshold_sample,recall_sample,'d-',label='Recall',color='#008367', alpha=0.6)
plt.plot(threshold_sample,f1_sample,'o--', label='F1',color='#A62C3A')

plt.legend()
plt.xlabel('Threshold')
# plt.ylabel('Score')
plt.minorticks_on()
plt.savefig('./Figure/Threshold_Curve', dpi=150, bbox_inches='tight')
plt.show()



# --------- 绘制核函數密度曲线 ---------
plt.figure(figsize=(5,4))
# histogram 版本 (因為核密度曲線平滑處理，0～1範圍以外的部分)
# 计算两组数据的最小值和最大值
min_val = min(y_pred_label0.min(), y_pred_label1.min())
max_val = max(y_pred_label0.max(), y_pred_label1.max())

# 计算bin的边界
bins = np.linspace(min_val, max_val, 50)

sns.histplot(y_pred_label0, label="Label 0", color="blue", lw=0.5, alpha=0.6, bins=bins)   
sns.histplot(y_pred_label1, label="Label 1", color="red", lw=0.5, alpha=0.6, bins=bins)

# 標出最佳threshold
plt.axvline(x=roc_best_threshold, color='#A62C3A', linestyle='--', label='Threshold')

# 设置图标题和坐标轴标签
plt.tick_params(axis='both', which='major', labelsize=12)

# plt.title(plot_title+' Distribution')
plt.xlabel("Score (Normalized)")
plt.ylabel('Count')
plt.minorticks_on()
# 显示图例
plt.legend()

# 保存
plt.savefig('./Figure/predict_distribution.png', dpi=150, bbox_inches="tight")
# 显示图
plt.show()

# %%   -------- 和NBLAST比較找出NBLAST表現不佳的案例(該區域不能在mode==nblast下運行) --------
nblast_label = pd.read_csv(nblast_path)
# normalize nblast similarity score
nblast_label['Norm NBLAST score'] = (nblast_label['similarity score'] - np.min(nblast_label['similarity score']))/(np.max(nblast_label['similarity score']) - np.min(nblast_label['similarity score']))
# 計算predict_df 中每一個pair和nblast的分數差距
predict_df = predict_df.merge(nblast_label[['fc_id', 'em_id', 'Norm NBLAST score']], on=['fc_id', 'em_id'], how='left')
predict_df['NBLAST_diff'] = np.abs(predict_df['model_pred'] - predict_df['Norm NBLAST score'])
# 按照NBLAST_diff排序，找出差距最大的前30個案例
worst_nblast = predict_df.sort_values(by='NBLAST_diff', ascending=False).head(30)
# %% Ranking analysis
plt.style.use('default')

top_k = 5
if test_mode == 'nblast':
    predict_df_clear = nblast_score[['fc_id', 'em_id', 'label', 'similarity score']].copy()
    predict_df_clear.rename(columns={'similarity score':'model_pred'}, inplace=True)
else:
    predict_df_clear = predict_df[['fc_id', 'em_id', 'label', 'model_pred']].copy()
# 二元化label(for soft label)
predict_df_clear['bi_label'] = [1 if x > 0.5 else 0 for x in predict_df_clear['label']]

grouped = predict_df_clear.groupby('fc_id')

# 创建一个空字典来保存每个分组的新 DataFrame
dfs = {}

for name, group in grouped:
    sorted_group = group.sort_values(by='model_pred', ascending=False)
    dfs[name] = sorted_group

# 挑出dfs中值長度大於top_k且包含模型預測為正的部分
filtered_dfs = {k:v for k,v in dfs.items() if len(v) >= top_k and 1 in v['bi_label'].values}

# top k accuracy
top_k_accuracy = []
for k in range(top_k,0,-1):
    correct = 0
    for key in filtered_dfs:
        # 只要前k個裡面有一個positive就算正確
        if filtered_dfs[key].iloc[0:k]['bi_label'].sum() > 0:
            correct += 1
    print('Top', k, 'Accuracy:', correct/len(filtered_dfs))
    
    top_k_accuracy.append(correct/len(filtered_dfs))
# bar plot top k accuracy
plt.figure(figsize=(int(np.round(2.1+0.5*top_k)),4))
plt.ylim([0,1.1])
plt.grid(axis='y', alpha=0.5)
x_axis_name = ['Top 5', 'Top 4', 'Top 3', 'Top 2', 'Top 1']
x_axis_name_filt = x_axis_name[-top_k:]
plt.bar(x_axis_name_filt, top_k_accuracy, color='#BB0F1E',linewidth=0)
# 設置y軸刻度線為虛線
plt.grid(axis='y', linestyle='--')

# 加上數字標籤，以百分比形式
for x,y in enumerate(top_k_accuracy):
    plt.text(x, y+0.01, '{:.1%}'.format(y), ha='center', color='black', fontsize=12)

plt.ylabel('Recall at K')
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


# %% 分析交換輸入的結果
cross_num = 1       #這裏為了快速分析，只調用一個模型來預測。
test_path = './data/statistical_results/three_view_pic_rk10/'
test_num = 1000     # 取1000個未標注資料測試
model_path = './Annotator_Model/'   #'Annotator_Model_Uninverse'

def annotator(model,fc_img, em_img):
    # 使用transpose()将数组形状从(3, 50, 50)更改为(50, 50, 3)
    fc_img = np.transpose(fc_img, (1, 2, 0))
    em_img = np.transpose(em_img, (1, 2, 0))

    # 将数据维度扩展至4维 (1,50,50,3)（符合CNN输入）
    fc_img = np.expand_dims(fc_img, axis=0)
    em_img = np.expand_dims(em_img, axis=0)
    label = model.predict({'FC':fc_img, 'EM':em_img}, verbose=0)

    label = label.flatten()[0]  #因為模型輸出是一個 numpy array

    return label

model_lst = []
for i in range(cross_num):
    model_name = 'Annotator_D1-D6_' +str(i) +'.h5'
    model = load_model(os.path.join(model_path, model_name))
    model_lst.append(model)

def gen_predict_df(file_path, model_lst, inverse=False):
    new_data_lst = []

    # 遍历母文件夹下的所有条目
    for pkl_file in tqdm(file_path, total=len(file_path)):
        # 读取pkl文件
        data_lst = load_pkl(pkl_file)
        for data in data_lst:
            # 计算结果
            # 計算各模型結果
            result_lst = []
            for model in model_lst:
                if inverse:         # 測試顛倒輸入，對比結果
                    result = annotator(model, data[4], data[3])
                else:
                    result = annotator(model, data[3], data[4])
            
                result_lst.append(result)

            # 計算平均值
            result_avg = np.mean(result_lst)

            # 計算二元標籤
            result_bin = 1 if result_avg > 0.5 else 0

            # 計算標準差
            result_std = np.std(result_lst)

            # 将文件名和计算结果添加到DataFrame
            new_data = {'fc_id': data[0], 'em_id': data[1], 'KT_score': data[2], 'model_predict': result_avg, 'binary_label': result_bin, 'pred_std': result_std}
            new_data_lst.append(new_data)

    label_df = pd.DataFrame(new_data_lst)

    return label_df

file_lst = [file_name for file_name in os.listdir(test_path) if file_name.endswith('.pkl')]
file_lst = [os.path.join(test_path, file_name) for file_name in file_lst]
file_lst = file_lst[:test_num]

label_df = gen_predict_df(file_lst, model_lst)
inv_label_df = gen_predict_df(file_lst, model_lst, inverse=True)

pred_lst = label_df['model_predict'].to_numpy()
inv_pred = inv_label_df['model_predict'].to_numpy()

d_inv = np.abs(pred_lst - inv_pred)

# 舊模型結果
model_path = './Annotator_Model_Uninverse/'
model_lst = []
for i in range(cross_num):
    model_name = 'Annotator_D1-D6_' +str(i) +'.h5'
    model = load_model(os.path.join(model_path, model_name))
    model_lst.append(model)


old_label_df = gen_predict_df(file_lst, model_lst)
old_inv_label_df = gen_predict_df(file_lst, model_lst, inverse=True)

old_pred_lst = old_label_df['model_predict'].to_numpy()
old_inv_pred = old_inv_label_df['model_predict'].to_numpy()

old_d_inv = np.abs(old_pred_lst - old_inv_pred)

# %% violin plot
fig, ax = plt.subplots(figsize=(6, 5))

sns.violinplot(data=[d_inv, old_d_inv], inner="box", palette=['#001BC2', '#E90132']) # 箱線圖
        
# 设置透明度
for violin in ax.collections:
    violin.set_alpha(0.8)

# 计算平均数
averages = [np.mean(p) for p in [d_inv, old_d_inv]]



# 在小提琴图上标注平均数
for i, avg in enumerate(averages):
    ax.text(i, y_lim[0]+0.02, f"Avg = {avg:.2f}", horizontalalignment='center', fontsize=12, color='black')

# 添加标题和轴标签
plt.xticks([0, 1], ['New', 'Old'])
plt.ylabel('Deviation')

plt.savefig('./Figure/exchange_diviation.png', dpi=150, bbox_inches="tight")
# 显示图像
plt.show()


# 畫box plot
plt.figure(figsize=(6,5))
sns.boxplot(data=[d_inv, old_d_inv], palette=['#001BC2', '#E90132'])

y_lim = ax.get_ylim()

# 标注平均数
for i, avg in enumerate(averages):
    ax.text(i, y_lim[0]+0.02, f"Avg = {avg:.2f}", horizontalalignment='center', fontsize=12, color='black')

plt.xticks([0, 1], ['Input permutation invariance', 'Before'])
plt.ylabel('Deviation')
plt.savefig('./Figure/exchange_diviation_box.png', dpi=150, bbox_inches="tight")
plt.show()
# %%
# save d_inv to npy
np.save('./result/d_inv.npy', d_inv)
np.save('./result/old_d_inv.npy', old_d_inv)





# %% 分析 nblast asymmetry（交換輸入的差異）
nblast_inv = pd.read_csv('./labeled_info/nblast_all_list_D2_D5_include_inverse_label.csv')
# normalize
score = nblast_inv['similarity score'].to_numpy()
score = (score - np.min(score))/(np.max(score) - np.min(score))

score_inv = nblast_inv['inverse score'].to_numpy()
score_inv = (score_inv - np.min(score_inv))/(np.max(score_inv) - np.min(score_inv))

asymmetry = np.abs(score - score_inv)

# violin plot
fig, ax = plt.subplots(figsize=(6, 5))

sns.violinplot(data=[asymmetry], inner="box", palette=['#E90132']) # 箱線圖
        
# 设置透明度
for violin in ax.collections:
    violin.set_alpha(0.8)

# 计算平均数
averages = [np.mean(p) for p in [asymmetry]]

y_lim = ax.get_ylim()

# 在小提琴图上标注平均数
for i, avg in enumerate(averages):
    ax.text(i, y_lim[0]+0.01, f"Avg = {avg:.2f}", horizontalalignment='center', fontsize=12, color='black')

# 添加标题和轴标签
plt.xticks([0], ['NBLAST'])
plt.ylabel('Deviation')

# %% 找出NBLAST表現不佳且我們有顯著改善的典型案例
D2 = pd.read_csv('./labeled_info/D2_conf.csv').drop_duplicates(subset=['fc_id', 'em_id'])
D6 = pd.read_csv('./labeled_info/D6_conf.csv').drop_duplicates(subset=['fc_id', 'em_id'])
D5 = pd.read_csv('./labeled_info/D5_conf.csv').drop_duplicates(subset=['fc_id', 'em_id'])

D2 = pd.concat([D2[['fc_id', 'em_id']].copy(), D6[['fc_id', 'em_id']].copy()], ignore_index=True)
D5 = D5[['fc_id', 'em_id']].copy()

# 需要執行test_mode == 'cross' 且 selected_test_set = False
model_predict = predict_df[['fc_id', 'em_id', 'model_pred', 'label']]
D2_model = model_predict.merge(D2, on=['fc_id', 'em_id'], how='inner')
D5_model = model_predict.merge(D5, on=['fc_id', 'em_id'], how='inner')

D2_D5_nblast = pd.read_csv('./labeled_info/nblast_official_mean.csv').drop(columns=['label'])
#Normalized score
D2_D5_nblast['Norm score'] = (D2_D5_nblast['similarity score'] - np.min(D2_D5_nblast['similarity score']))/(np.max(D2_D5_nblast['similarity score']) - np.min(D2_D5_nblast['similarity score']))

D2_nblast = D2_D5_nblast.merge(D2, on=['fc_id', 'em_id'], how='inner')
D5_nblast = D2_D5_nblast.merge(D5, on=['fc_id', 'em_id'], how='inner')

D2_result = D2_nblast.merge(D2_model, on=['fc_id', 'em_id'], how='outer')
D5_result = D5_nblast.merge(D5_model, on=['fc_id', 'em_id'], how='outer')

# 這裡嘗試將NBLAST分數和模型分數畫在同一scatter plot 中比較，可以看出靠近座標軸的數據是兩種方法差異較大的
D2_pos = D2_result[D2_result['label'] >= 0.5]
D2_neg = D2_result[D2_result['label'] < 0.5]
D5_pos = D5_result[D5_result['label'] >= 0.5]
D5_neg = D5_result[D5_result['label'] < 0.5]

def plot_score_compare(pos_result, neg_result, title):
    plt.figure(figsize=(6,6))
    plt.plot(pos_result['Norm score'], pos_result['model_pred'], '.', c='r', label=title+'_True')
    plt.plot(neg_result['Norm score'], neg_result['model_pred'], '.', c='b', label=title+'_False')
    plt.plot([0, 1], [0, 1], color='lightseagreen', linestyle='--', label='y=x')
    plt.xlabel('NBLAST Score')
    plt.ylabel('Model Score')
    plt.legend()
    plt.savefig('./Figure/score_compare_'+title+'.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_score_compare(D2_pos, D2_neg, 'D2')
plot_score_compare(D5_pos, D5_neg, 'D5')

# 排序後更容易找代表性案例
D2_pos = D2_pos.sort_values(by=['Norm score', 'model_pred'], ascending=[True, False])
D5_pos = D5_pos.sort_values(by=['Norm score', 'model_pred'], ascending=[True, False])
D2_neg = D2_neg.sort_values(by=['Norm score', 'model_pred'], ascending=[False, True])
D5_neg = D5_neg.sort_values(by=['Norm score', 'model_pred'], ascending=[False, True])

# 計算score和label的差異
D2_result['nblast diff'] = D2_result['Norm score'] - D2_result['label']
D2_result['model diff'] = D2_result['model_pred'] - D2_result['label']

D5_result['nblast diff'] = D5_result['Norm score'] - D5_result['label']
D5_result['model diff'] = D5_result['model_pred'] - D5_result['label']

# %% plot model_diff-nblast_diff 2D scatter
plt.style.use('dark_background')

plt.plot(D2_result['nblast diff'], D2_result['model diff'], '.', label='D2')
plt.plot(D5_result['nblast diff'], D5_result['model diff'], '.', label='D5')
# 畫對角線
plt.plot([0, 1], [0, 1], color='lightgreen', linestyle='--', label='y=x')
# 畫回歸線
z2 = np.polyfit(D2_result['nblast diff'], D2_result['model diff'], 1)   #斜率和截距
p2 = np.poly1d(z2)
# R 值
r_D2 = np.corrcoef(D2_result['nblast diff'], D2_result['model diff'])[0,1]

plt.plot(D2_result['nblast diff'], p2(D2_result['nblast diff']), "--", label='D2 Regression')

z5 = np.polyfit(D5_result['nblast diff'], D5_result['model diff'], 1)   #斜率和截距
p5 = np.poly1d(z5)
r_D5 = np.corrcoef(D5_result['nblast diff'], D5_result['model diff'])[0,1]

plt.plot(D5_result['nblast diff'], p5(D5_result['nblast diff']), "--", label='D5 Regression')
# 印上R值
x_lim = plt.gca().get_xlim()
plt.text(x_lim[0]+0.1, 0.4, f'R_D2 = {r_D2:.2f}', fontsize=10, color='white')
plt.text(x_lim[0]+0.1, 0.3, f'R_D5 = {r_D5:.2f}', fontsize=10, color='white')

plt.xlabel('NBLAST Diff')
plt.ylabel('Model Diff')
plt.legend()
plt.savefig('./Figure/Diff_Scatter.png', dpi=150, bbox_inches='tight')
plt.show()
# %% 找出模型、NBLAST表現差異大的
def find_diff_examplt(df, rank_num=10):
    # 找出差異最大的
    df['diff'] = np.abs(df['model diff']) - np.abs(df['nblast diff'])
    df = df.sort_values(by='diff', ascending=False)
    model_fail = df.iloc[:rank_num]
    nblast_fail = df.iloc[-rank_num:]
    nblast_fail = nblast_fail[::-1]


    return model_fail, nblast_fail

model_fail_D2, nblast_fail_D2 = find_diff_examplt(D2_result, rank_num=20)
model_fail_D5, nblast_fail_D5 = find_diff_examplt(D5_result, rank_num=20)


# %% 