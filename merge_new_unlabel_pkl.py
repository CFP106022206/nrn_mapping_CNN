# %%    確認yifan後續提供的新map_data和舊的是否重複 並且合併不重複的神經對
import sys
sys.path.insert(0, '/opt/tensorflow/2.9.0/local/lib/python3.10/dist-packages')

import os
import shutil
import pickle
from util import load_pkl

# %%
unlabel_path = './data/mapping_data_0.6-0.7'
new_unlabel = './data/mapping_data_0.7+'


merge_path = unlabel_path + '_' + new_unlabel[7:]
os.makedirs(merge_path, exist_ok=True)



# %%

# 筛选出指定文件夹下以 .pkl 结尾的文件並存入列表
file_list_old = [file_name for file_name in os.listdir(unlabel_path) if file_name.endswith('.pkl')]
file_list_add = [file_name for file_name in os.listdir(new_unlabel) if file_name.endswith('.pkl')]


# 比較文件名稱, 文件以FC nrn命名編組。查找新增文件中是否有舊文件不包含的神經

exist_nrn_file, new_nrn_file = [], []
for add_nrn in file_list_add:
    if add_nrn in file_list_old:
        exist_nrn_file.append(add_nrn)
    else:
        new_nrn_file.append(add_nrn)

print('新增文件: ')
print(new_nrn_file)
print('\n新增數量: ', len(new_nrn_file))
print('\n已存在同名文件(.pkl)數量: ', len(exist_nrn_file) )


# 將新增文件複製到merge path
for file_name in new_nrn_file:
    new_file_path = os.path.join(new_unlabel, file_name)
    merge_file_path = os.path.join(merge_path, file_name)
    shutil.copy2(new_file_path, merge_file_path)

if len(new_nrn_file):
    print('新增文件已複製到目標文件夾')



# 對比已存在nrn內容, 保留內容較多者

repeat_num = 0         # 紀錄覆蓋了多少相同 nrn pair
add_nrn_num = 0        # 紀錄新增多少 nrn pair




for file_name in exist_nrn_file:
    # 创建完整文件路径
    old_path = os.path.join(unlabel_path, file_name)
    new_path = os.path.join(new_unlabel, file_name)

    # 读取pkl文件
    old_data_lst = load_pkl(old_path)
    new_data_lst = load_pkl(new_path)

    # 比較 data_lst 內容, 注意: 存在相同nrn時默認新的文件覆蓋舊的
    index_to_remove = set()    # 使用集合搜索時時間複雜度較低

    data_dict = dict()         # 將新 nrn list 中的神經名稱儲存成字典鍵值以快速查找
    for data in new_data_lst:
        nrn = f'{data[0]}_{data[1]}'
        data_dict[nrn] = []
        # fc_nrn = str(data[0])
        # em_nrn = str(data[1])
    
    # 查找舊文件是否有同名元素
    for j, data in enumerate(old_data_lst):
        nrn = f'{data[0]}_{data[1]}'

        if nrn in data_dict:
            index_to_remove.add(j)
            repeat_num += 1
            break
    
    # 將 old_data_lst 加入 merge list, 刪除 new_data_lst包含的同名nrn pair 並加入 new_data_lst
    merge_lst = [item for i, item in enumerate(old_data_lst) if i not in index_to_remove] + new_data_lst
    add_nrn_num += len(new_data_lst) - len(index_to_remove)

    # 將merge_lst 打包成pkl 保存到 merge_path
    with open(os.path.join(merge_path, file_name), 'wb') as f:
        pickle.dump(merge_lst, f)


print('\n已發現擁有同 nrn pair 數量(新文件覆蓋): ', repeat_num)
print('新增 nrn pair 數量: ', add_nrn_num)
print('\n\n完整任務進程結束.')

        








# %%
