'''
清除以下文件夾:
./Figure
./result
./Annotator_Model
./Second_Stage_Model
./Final_Model
./Iterative_Figure
./Iterative_result
./Iterative_Second_Stage_Model
./Iterative_Final_Model
./DNN_Classifier
'''
# %%
import os
import shutil

def clear_directory(directory_path):
    if os.path.exists(directory_path):
        # 删除整个文件夹
        shutil.rmtree(directory_path)

    # 创建一个同名文件夹
    os.mkdir(directory_path)

delete_folder_lst = ['./Figure', './result', './Annotator_Model', './Second_Stage_Model', './Final_Model',
                     './Iterative_Figure', './Iterative_result', './Iterative_Second_Stage_Model',
                     './Iterative_Final_Model', './DNN_Classifier'] #, './CAE_EM', './CAE_FC']

for folder in delete_folder_lst:
    clear_directory(folder)

# %%
