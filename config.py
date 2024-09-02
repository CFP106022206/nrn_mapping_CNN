import platform
import os

config_path = {}
if platform.system() == "Windows":
    config_path["aug"] = os.getcwd()+"\\data\\fake_data\\"
    config_path["import_swc"] = os.getcwd()+"\\data\\selected_data\\"
    config_path["convert"] = os.getcwd()+"\\data\\converted_data\\"
    config_path["stats"] = os.getcwd()+"\\data\\statistical_results\\"
    config_path["forbidden"] = os.getcwd()+"\\data\\forbidden_data\\"
    config_path["map"] = os.getcwd()+"\\data\\mapping_data1\\"
    config_path["map2"] = os.getcwd()+"\\data\\mapping_data2\\"
    config_path["plot_single_neuron"] = os.getcwd()+"\\plot\\single_neuron\\"
    config_path["plot_pair_neuron"] = os.getcwd()+"\\plot\\pair_neurons\\"
    config_path["sep"] = "\\"
else:
    config_path["aug"] = os.getcwd()+"/data/fake_data/"
    config_path["import_swc"] = os.getcwd()+"/data/selected_data/"
    config_path["convert"] = os.getcwd()+"/data/converted_data/"
    config_path["stats"] = os.getcwd()+"/data/statistical_results/"
    config_path["forbidden"] = os.getcwd()+"/data/forbidden_data/"
    config_path["map"] = os.getcwd()+"/data/mapping_data1/"
    config_path["map2"] = os.getcwd()+"/data/mapping_data2/"
    config_path["plot_single_neuron"] = os.getcwd()+"/plot/single_neuron/"
    config_path["plot_pair_neuron"] = os.getcwd()+"/plot/pair_neurons/"
    config_path["sep"] = "/"
