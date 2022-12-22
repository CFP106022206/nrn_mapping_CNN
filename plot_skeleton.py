# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib import animation
from tqdm import tqdm

def load_pkl(path):
    if path[-4:] != '.pkl':
        # print('Check the file type')
        path += '.pkl'
    with open(path,'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data

def plot_neuron(df_neuron, output_folder, file_name='skeleton.mp4', plot_mode='normal', dot_size=0.2, show_axis=True):
    if type(file_name) != str:
        print('Wrong input type: file_name.\nAuto convert to string.')
        file_name = str(file_name)
    if file_name[-4:] != '.mp4':
        file_name += '.mp4'

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if plot_mode not in ['normal', 'polar', 'strahler']:
        plot_mode = 'normal'
        print("\nWrong mode input! Auto change to mode 'normal'.")

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
    rot_animation.save(output_folder+file_name, dpi=400, writer=writer)
    print('Complete.')

skeleton_path = '/home/ming/Project/nrn_mapping_package-master/data/converted_data/'

pkl_file = os.walk(skeleton_path)

skeleton_path_lst = []
for path, dir_list, file_list in pkl_file:
    for file_name in file_list:
        if file_name[-4:] == '.pkl':
            skeleton_path_lst.append(os.path.join(path, file_name))

# %%
plt.style.use('default')
output_folder = '/home/ming/Project/nrn_mapping_package-master/3D_Plot/'
for file_name in tqdm(skeleton_path_lst):
    data_df = load_pkl(file_name)
    plot_neuron(data_df, output_folder, file_name[66:-4], plot_mode='normal', dot_size=0.25, show_axis=True)

