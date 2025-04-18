# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib import animation
from scipy.interpolate import interp1d

def interpolate_points(df, num_points):
    # 創建原始data frame數據的索引
    old_indices = np.arange(0, df.shape[0])

    # 創建內插後的新索引，以便在每個點之間進行內插
    new_indices = np.linspace(0, df.shape[0]-1, df.shape[0] + (df.shape[0]-1) * num_points)

    # 創建新的data frame，索引採用新索引
    new_df = pd.DataFrame(index=new_indices)

    # Interpolate each column 注意 必須保證每列數據為連續值
    for column in df.columns:
        interpolator = interp1d(old_indices, df[column])
        new_df[column] = interpolator(new_indices)


    return new_df
def plot_pairs_neuron(df_neuron_lst, color_map, output_folder, file_name='skeleton.mp4', dot_size=0.2, interpolate=[1], show_axis=True):
    if type(file_name) != str:
        file_name = str(file_name)
    if file_name[-4:] != '.mp4':
        file_name += '.mp4'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # interpolate
    if interpolate:
        for i in interpolate:
            df_neuron_lst[i] = interpolate_points(df_neuron_lst[i].copy(), 10)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, df_neuron in enumerate(df_neuron_lst):
        ax.scatter(df_neuron['x'], df_neuron['y'], df_neuron['z'], s = dot_size, linewidths = 0, c=color_map[i])
        # plot soma
        if -1 in df_neuron['Parent'].values.astype(int):
            soma = df_neuron[df_neuron['Parent'].astype(int) == -1]
            ax.scatter(soma['x'], soma['y'], soma['z'], s = dot_size*90, c=color_map[i], alpha=0.7, label='Soma')

    # Set axis equal 避免神經變形失真
    max_length = np.max([np.abs(ax.get_xlim()[0]-ax.get_xlim()[1]), np.abs(ax.get_ylim()[0]-ax.get_ylim()[1]), np.abs(ax.get_zlim()[0]-ax.get_zlim()[1])])
    ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[0]+max_length])
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+max_length])
    ax.set_zlim([ax.get_zlim()[0], ax.get_zlim()[0]+max_length])

    if show_axis == False:
        ax.axis('off')

    ax.set_title(file_name[:-4])

    def rotate(angle): 
        ax.view_init(azim=angle)

    print('Saving...')
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,361,1),interval=100) 
    writer = animation.FFMpegWriter(fps=24, bitrate=1536)
    rot_animation.save(output_folder+file_name, dpi=400, writer=writer)
    print('Complete.')

plt.style.use('default')

em_id = 'G0239-F-000012'
fc_id = 'G0239-F-000001'

em_path = './data/selected_data/EM/'+ em_id +'.swc'
fc_path = './data/selected_data/FC/'+ fc_id +'.swc'

mp4_output_path = './Figure/plot_skeletons/'
if not os.path.exists(mp4_output_path):
    os.makedirs(mp4_output_path)

skeleton_dot_size = 0.5
show_axis = True

if os.path.exists(em_path) and os.path.exists(fc_path):
    # 读取 swc 文件
    em_df = pd.read_csv(em_path, sep='\s+', comment='#', header=None, names=['type', 'x', 'y', 'z', 'R', 'Parent'])
    fc_df = pd.read_csv(fc_path, sep='\s+', comment='#', header=None, names=['type', 'x', 'y', 'z', 'R', 'Parent'])

    plot_pairs_neuron([em_df, fc_df], ['#5641D5','#E22146'], mp4_output_path, file_name=em_id+'_'+fc_id+'.mp4', dot_size=skeleton_dot_size, interpolate=[1], show_axis=show_axis)

else:
    print('File not exists or wrong path')
# %%
