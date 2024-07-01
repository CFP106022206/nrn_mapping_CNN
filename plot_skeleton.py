# %%
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import numpy as np
from matplotlib import animation
from scipy.interpolate import interp1d
from tqdm import tqdm

def load_pkl(path):
    if path[-4:] != '.pkl':
        # print('Check the file type')
        path += '.pkl'
    with open(path,'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data


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


def plot_neuron(df_neuron, output_folder, file_name='skeleton.mp4', plot_mode='normal', dot_size=0.2, interpolate=True, show_axis=True):
    if type(file_name) != str:
        file_name = str(file_name)
    if file_name[-4:] != '.mp4':
        file_name += '.mp4'

    # interpolate
    if interpolate:
        interpolate_df = interpolate_points(df_neuron[['x', 'y', 'z']].copy(), 8)
        expand_columns = df_neuron[['type', 'R', 'Parent']].copy()
        new_index = interpolate_df.index
        expand_columns = expand_columns.reindex(new_index)
        expand_columns.ffill(inplace=True)

        df_neuron = pd.concat([expand_columns, interpolate_df], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if plot_mode not in ['normal', 'polar', 'strahler']:
        plot_mode = 'normal'
        print("\nWrong mode input! Auto choose mode 'normal'.")

    if plot_mode == 'normal':
        ax.scatter(df_neuron['x'], df_neuron['y'], df_neuron['z'], s = dot_size, linewidths = 0)

    elif plot_mode == 'polar':
        x_unlabel = df_neuron.loc[df_neuron['type'] == 0]['x']
        x_soma = df_neuron.loc[df_neuron['type'] == 1]['x']
        x_axon = df_neuron.loc[df_neuron['type'] == 2]['x']
        x_dend = df_neuron.loc[df_neuron['type'] == 3]['x']

        y_unlabel = df_neuron.loc[df_neuron['type'] == 0]['y']
        y_soma = df_neuron.loc[df_neuron['type'] == 1]['y']
        y_axon = df_neuron.loc[df_neuron['type'] == 2]['y']
        y_dend = df_neuron.loc[df_neuron['type'] == 3]['y']

        z_unlabel = df_neuron.loc[df_neuron['type'] == 0]['z']
        z_soma = df_neuron.loc[df_neuron['type'] == 1]['z']
        z_axon = df_neuron.loc[df_neuron['type'] == 2]['z']
        z_dend = df_neuron.loc[df_neuron['type'] == 3]['z']

        ax.scatter(x_unlabel, y_unlabel, z_unlabel, s = dot_size, linewidths=0, label='unlabel')
        ax.scatter(x_soma, y_soma, z_soma, c = 'k', s = dot_size*12, linewidths=0, label='soma')
        ax.scatter(x_axon, y_axon, z_axon, c = 'r', s = dot_size, linewidths=0, label='axon')
        ax.scatter(x_dend, y_dend, z_dend, c = 'b', s = dot_size, linewidths=0, label='dendrite')
        # ax.legend()   # Animation of rotate doesn't shows this well

    elif plot_mode == 'strahler':
        x = df_neuron['x']
        y = df_neuron['y']
        z = df_neuron['z']

        im = ax.scatter(x, y, z, c=df_neuron.iloc[:,-1], s = dot_size, linewidths=0, cmap='jet')
        fig.colorbar(im)

    # Set axis equal 避免神經變形失真
    max_length = np.max([np.abs(ax.get_xlim()[0]-ax.get_xlim()[1]), np.abs(ax.get_ylim()[0]-ax.get_ylim()[1]), np.abs(ax.get_zlim()[0]-ax.get_zlim()[1])])
    ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[0]+max_length])
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+max_length])
    ax.set_zlim([ax.get_zlim()[0], ax.get_zlim()[0]+max_length])
    # axis_min, axis_max = np.min([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]]), np.max([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])
    # ax.set_xlim([axis_min, axis_max])
    # ax.set_ylim([axis_min, axis_max])
    # ax.set_zlim([axis_min, axis_max])

    if show_axis == False:
        ax.axis('off')

    ax.set_title(file_name[:-4])

    def rotate(angle): 
        ax.view_init(azim=angle)

    print('Saving...')
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,361,1),interval=100) 
    writer = animation.FFMpegWriter(fps=24, bitrate=1536)
    rot_animation.save(output_folder + file_name, dpi=400, writer=writer)
    print('Complete.')


def plot_pairs_neuron(df_neuron_lst, label_lst, color_map, output_folder, file_name='skeleton.mp4', dot_size=0.2, interpolate=[0], show_axis=True):
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
            if label_lst:
                ax.scatter(soma['x'], soma['y'], soma['z'], s = dot_size*90, c=color_map[i], alpha=0.7, label=label_lst[i])
            else:
                ax.scatter(soma['x'], soma['y'], soma['z'], s = dot_size*90, c=color_map[i], alpha=0.7)
    ax.legend()
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

    print('Drawing 3D Plot...')
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,361,1),interval=100) 
    writer = animation.FFMpegWriter(fps=24, bitrate=1536)
    rot_animation.save(output_folder+file_name, dpi=400, writer=writer)
    print('Complete.')




# %%
plt.style.use('default')
plot_id = 'TH-F-200026'
plot_path = './data/selected_data/FC/'+ plot_id +'.swc'

if os.path.exists(plot_path):
    # 读取 swc 文件
    df = pd.read_csv(plot_path, sep=' ', header=None, skiprows=1, names=['type', 'x', 'y', 'z', 'R', 'Parent'])
    #re-index
    df.reset_index(drop=True, inplace=True)
    plot_neuron(df, './Figure/plot_skeletons/', file_name=plot_id+'.mp4', plot_mode='normal', dot_size=1, interpolate=True, show_axis=True)




else:
    print('File not exists or wrong path')

# %% pairs mode
plt.style.use('default')

em_id = '859265651'
fc_id = 'TH-F-100075'

em_path = './data/selected_data/EM/'+ em_id +'.swc'
fc_path = './data/selected_data/FC/'+ fc_id +'.swc'

if os.path.exists(em_path) and os.path.exists(fc_path):
    # 读取 swc 文件
    em_df = pd.read_csv(em_path, sep='\s+', comment='#', header=None, names=['type', 'x', 'y', 'z', 'R', 'Parent'])
    fc_df = pd.read_csv(fc_path, sep='\s+', comment='#', header=None, names=['type', 'x', 'y', 'z', 'R', 'Parent'])

    plot_pairs_neuron([fc_df, em_df], ['FC', 'EM'], ['#5641D5','#E22146'], './Figure/plot_skeletons/', file_name=fc_id+'_'+em_id+'.mp4', dot_size=0.2, show_axis=True)

else:
    if not os.path.exists(em_path):
        print('EM File not exists')
    if not os.path.exists(fc_path):
        print('FC File not exists')
# %%
