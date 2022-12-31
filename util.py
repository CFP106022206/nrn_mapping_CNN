# todo: check unused package
import os
import csv
import sys
import copy
import math
import time
import random
import shutil
import pickle
import numpy as np
import neurom as nm
import pandas as pd
import itertools as it
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from itertools import chain
from matplotlib import figure
from matplotlib import animation
from collections import OrderedDict


def recoTxt(text_name):
    """
    reconstruct the text to the form we'd like to use

    :param text_name:
    :return:
    """
    f = open(text_name + ".swc", "r")
    lis = []
    start = True
    for line in f:
        if start:
            start = False
            if line == '#n T x y z R P\n':
                return None
        if '#' in line:
            continue
        elif line[0] == "\n":
            continue
        else:
            line = line.strip()
            line = line.replace("\t", " ")
            line = line.replace("   ", " ")
            line = line.replace("  ", " ")
            line = line + "\n"
            lis.append(line)
    lis.insert(0, '#n T x y z R P\n')
    f.close()
    f = open(text_name + ".swc", "w")
    for i in lis:
        f.write(i)
    f.close()


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def listInnerFind(lis, value, index=-1):
    if index == -1:
        index = len(lis) - 1
    while index >= 0:
        if value in lis[index]:
            return index
        index -= 1
    # print('The value is not in this list !')
    return -1


def NN_dis(lst_1, lst_2):
    temp = 0
    for i in range(2, 5):
        temp += (lst_1[i] - lst_2[i]) ** 2
    temp = temp ** 0.5
    return temp


def children_search(lst, value):
    index = []
    for i in range(len(lst)):
        if lst[i][6] == value:
            index.append(i)
    return index


def deep_children_search(lst, value):
    # level 12
    # deep level 13

    # filter
    index = -1
    for i in range(len(lst)):
        if lst[i][0] == value:
            index = i
    if index == -1:
        print("ID not in lst")
        return -1

    level = lst[index][12]
    dp_level = lst[index][13]

    effective_range = []

    for i in range(len(lst)):
        if lst[i][13] != dp_level:
            continue
        elif lst[i][12] < level:
            continue
        else:
            effective_range.append(i)

    index = [index]
    temp_index1 = [index]
    temp_index2 = []
    while True:
        for i in temp_index1:
            for j in effective_range:
                if lst[j][7] == lst[i][0]:
                    temp_index2.append(j)
        index += temp_index2
        temp_index1 = temp_index2
        temp_index2 = []

    return index


@jit(nopython=True)
def diving(x1, y1, r1, N1):
    mp = []

    # odd N
    if N1 % 2 == 1:
        dr = 2 * r1 / N1
        cn = int(N1 // 2)
        for i in range(x1.shape[0]):
            temp = []
            # x direction
            xs = int(np.sign(x1[i]))
            xq = np.abs(x1[i] / dr)
            xp = int(((xq//0.5)+1)//2)
            if xp == int(N1-N1//2):
                xp -= 1
            xp = cn + xs*xp
            temp.append(xp)

            # y direction
            ys = int(np.sign(y1[i]))
            yq = np.abs(y1[i] / dr)
            yp = int(((yq // 0.5) + 1) // 2)
            if yp == int(N1 - N1 // 2):
                yp -= 1
            yp = cn + ys * yp
            temp.append(yp)

            mp.append(temp)

    # even N
    if N1 % 2 == 0:
        dr = 2 * r1 / N1
        cnp = N1 // 2
        cnn = N1 // 2 - 1
        for i in range(x1.shape[0]):
            temp = []
            # x direction
            xs = int(np.sign(x1[i]))
            xq = np.abs(x1[i] / dr)
            xp1 = int(xq//1)
            if xp1 == int(N1 - N1 // 2):
                xp1 -= 1
            if xs == 1:
                xp = cnp + xp1
            elif xs == -1:
                xp = cnn - xp1
            else:
                print(x1[i])
                print("even N wrong!")
            temp.append(xp)

            # y direction
            ys = int(np.sign(y1[i]))
            yq = np.abs(y1[i] / dr)
            yp1 = int(yq // 1)
            if yp1 == int(N1 - N1 // 2):
                yp1 -= 1
            if ys == 1:
                yp = cnp + yp1
            elif ys == -1:
                yp = cnn - yp1
            else:
                print("even N wrong!")
            temp.append(yp)

            mp.append(temp)

    return mp


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def matching_score(map1, map2, mode="normal", optional=5, shift=3):
    if mode == "normal":
        score_x = np.average(1 - np.abs(np.subtract(map1[0], map2[0])))
        score_y = np.average(1 - np.abs(np.subtract(map1[1], map2[1])))
        score_z = np.average(1 - np.abs(np.subtract(map1[2], map2[2])))

        return score_x, score_y, score_z

    elif mode == "shift_mask":
        for i in range(len(map1[0].shape[0]-optional+1)):
            a = 1 + 1

    elif mode == "mask1":
        score_x = np.zeros((map1[0].shape[0], map1[0].shape[0]))
        score_y = np.zeros((map1[0].shape[0], map1[0].shape[0]))
        score_z = np.zeros((map1[0].shape[0], map1[0].shape[0]))

        for i in range(score_x.shape[0]):
            for j in range(score_x.shape[0]):
                score_x[i][j] = np.average(1 - np.abs(np.subtract(map1[0][i][j],
                                                                  np.max(map2[i:i + optional, j:j + optional]))))
        for i in range(score_x.shape[0]):
            for j in range(score_x.shape[0]):
                score_y[i][j] = np.average(1 - np.abs(np.subtract(map1[0][i][j],
                                                                  np.max(map2[i:i + optional, j:j + optional]))))
        for i in range(score_x.shape[0]):
            for j in range(score_x.shape[0]):
                score_x[i][j] = np.average(1 - np.abs(np.subtract(map1[0][i][j],
                                                                  np.max(map2[i:i + optional, j:j + optional]))))
        return score_x, score_y, score_z

    elif mode == "mask2":
        length = map1[0].shape[0]-optional+1
        score_x = np.zeros((length, length))
        score_y = np.zeros((length, length))
        score_z = np.zeros((length, length))

        for i in range(length):
            for j in range(length):
                score_x[i][j] = np.average(1-np.abs(np.subtract(np.max(map1[0][i:i+optional, j:j+optional]),
                                                                np.max(map2[0][i:i+optional, j:j+optional]))))
        for i in range(length):
            for j in range(length):
                score_y[i][j] = np.average(1 - np.abs(np.subtract(np.max(map1[1][i:i + optional, j:j + optional]),
                                                                  np.max(map2[1][i:i + optional, j:j + optional]))))
        for i in range(length):
            for j in range(length):
                score_z[i][j] = np.average(1 - np.abs(np.subtract(np.max(map1[2][i:i + optional, j:j + optional]),
                                                                  np.max(map2[2][i:i + optional, j:j + optional]))))
        return score_x, score_y, score_z


def rotation_3D(cv, theta):
    x, y, z = cv[0], cv[1], cv[2]
    return np.array([[np.cos(theta)+(1-np.cos(theta))*np.power(x, 2), (1-np.cos(theta))*x*y-np.sin(theta)*z, (1-np.cos(theta))*x*z+np.sin(theta)*y],
                     [(1-np.cos(theta))*y*x+np.sin(theta)*z, np.cos(theta)+(1-np.cos(theta))*np.power(y, 2), (1-np.cos(theta))*y*z-np.sin(theta)*x],
                     [(1-np.cos(theta))*z*x-np.sin(theta)*y, (1-np.cos(theta))*z*y+np.sin(theta)*x, np.cos(theta)+(1-np.cos(theta))*np.power(z, 2)]])


def reduce_mapping(lst, num):
    """
    reduce the Strahler number into a given number
    :param lst:
    :param num:
    :return:
    """
    max_num = np.max(lst[:, 3])
    if max_num <= num:
        return lst
    s, q = int(max_num // num), max_num % num
    s_dict = {}
    for i in range(num):
        if i == 0:
            s_dict["s" + str(i + 1)] = [j + 1 for j in range(s)]
        else:
            s_dict["s" + str(i + 1)] = [j + 1 + s_dict["s" + str(i)][-1] for j in range(s)]

        if q >= i + 1:
            s_dict["s" + str(i + 1)].append(s_dict["s" + str(i + 1)][-1] + 1)

    for i in range(len(s_dict)):
        mask = np.isin(lst[:, 3], s_dict["s" + str(i+1)])
        lst[:, 3][mask] = i + 1

    return lst


def baseline_rule(lst_temp, max_num, norm):
    """
    set the upper limit of the Strahler number
    :param lst_temp:
    :param max_num:
    :return:
    """
    lst_temp = np.hstack((lst_temp, np.zeros((lst_temp.shape[0], 1))))
    if np.max(lst_temp[:, 3]) > max_num:
        baseline = max_num - np.max(lst_temp[:, 3])
        lst_temp[:, 5] = lst_temp[:, 3] + baseline
        mask = lst_temp[:, 5] < 0
        lst_temp[:, 5][mask] = 0
    else:
        lst_temp[:, 5] = lst_temp[:, 3]

    if norm:
        lst_temp[:, 5] = lst_temp[:, 5]/np.max(lst_temp[:, 5])

    return lst_temp


def coordinate_core(lst, norm, weight_method):
    # evaluate the centor of mass with specific weight
    weight = weight_method(lst[:, 5])
    total_weight = np.sum(weight)

    cx, cy, cz = [0.0, 0.0, 0.0]
    cx += np.dot(lst[:, 0], weight)/total_weight
    cy += np.dot(lst[:, 1], weight)/total_weight
    cz += np.dot(lst[:, 2], weight)/total_weight

    lst[:, 0] -= cx
    lst[:, 1] -= cy
    lst[:, 2] -= cz

    # Moment of Inertia
    Ixx = 0.0
    Iyy = 0.0
    Izz = 0.0
    Ixy = 0.0
    Iyz = 0.0
    Izx = 0.0

    Ixx += np.sum(np.multiply(np.power(lst[:, 1], 2) + np.power(lst[:, 2], 2), weight))
    Iyy += np.sum(np.multiply(np.power(lst[:, 2], 2) + np.power(lst[:, 0], 2), weight))
    Izz += np.sum(np.multiply(np.power(lst[:, 0], 2) + np.power(lst[:, 1], 2), weight))
    Ixy -= np.sum(np.multiply(np.multiply(lst[:, 0], lst[:, 1]), weight))
    Iyz -= np.sum(np.multiply(np.multiply(lst[:, 1], lst[:, 2]), weight))
    Izx -= np.sum(np.multiply(np.multiply(lst[:, 2], lst[:, 0]), weight))

    I = np.array([[Ixx, Ixy, Izx],
                  [Ixy, Iyy, Iyz],
                  [Izx, Iyz, Izz]])
    eigval, eigvec = np.linalg.eig(I)
    if norm:
        eigval1 = eigval / eigval.max()
    else:
        eigval1 = eigval
    eigval_ave = eigval / lst.shape[0]

    # principal axis
    principal_vec = np.zeros((3, 3))
    index = eigval1.argsort()
    eigval1.sort()
    eigval_ave.sort()
    principal_vec[:, 0] += eigvec[:, index[0]]
    principal_vec[:, 1] += eigvec[:, index[1]]
    principal_vec[:, 2] += eigvec[:, index[2]]

    # right-hand rule
    if np.dot(np.cross(principal_vec[:, 0], principal_vec[:, 1]), principal_vec[:, 2]) < 0:
        principal_vec[:, 2] = -principal_vec[:, 2]

    return [eigval1, eigval_ave, principal_vec, np.array([cx, cy, cz])]


def coordinate_rule(nrn_lst, norm_v, key_lst):
    _dict = {}
    # unit weight
    key = "unit"
    if key in key_lst:
        _weight = lambda item: item/item
        _dict[key] = coordinate_core(nrn_lst, norm_v, _weight)

    # sn weight
    key = "sn"
    if key in key_lst:
        _weight = lambda item: item
        _dict[key] = coordinate_core(nrn_lst, norm_v, _weight)

    # reverse sn unit
    key = "rsn"
    if key in key_lst:
        _weight = lambda item: 1/item
        _dict[key] = coordinate_core(nrn_lst, norm_v, _weight)

    return _dict


def coord_enumerator(cor, key1="R", key2="L"):
    """
    give the possible combinations of three axis
    :param cor:
    shape = (3, 3) row- -> corresponding components
                   column --> x, y, z new coordinate
    :param key1:
    the name of the right hand orientation coordinate

    :param key2:
    the name of the left hand orientation coordinate

    :return:

    """
    cors = [cor[0], -cor[0], cor[1], -cor[1], cor[2], -cor[2]]
    # all possible permutations of coordinates
    #poss_comb = list(it.permutations((0, 2, 4), 3))

    # set the permutation by the eigenvalues of moment of inertia
    poss_comb = [(0, 2, 4)]

    add_comb = list(it.combinations_with_replacement((0, 1), 3))
    temp = []
    for i in range(len(add_comb)):
        temp += list(it.permutations(add_comb[i], 3))
    add_comb = list(set(temp))
    del temp

    cor_dict = {}
    rr = 0
    ll = 0
    for i in range(len(poss_comb)):
        for j in range(len(add_comb)):
            if np.dot(np.cross(cors[poss_comb[i][0]+add_comb[j][0]],
                               cors[poss_comb[i][1]+add_comb[j][1]]),
                      cors[poss_comb[i][2]+add_comb[j][2]]) > 0:
                rr += 1
                cor_dict[key1 + str(rr)] = [cors[poss_comb[i][0]+add_comb[j][0]],
                                            cors[poss_comb[i][1]+add_comb[j][1]],
                                            cors[poss_comb[i][2]+add_comb[j][2]]]
            else:
                ll += 1
                cor_dict[key2 + str(ll)] = [cors[poss_comb[i][0] + add_comb[j][0]],
                                            cors[poss_comb[i][1] + add_comb[j][1]],
                                            cors[poss_comb[i][2] + add_comb[j][2]]]
    return cor_dict


def mapping_rule(lst, coordinate, cm, key):
    if key == "unit":

        def func(x):
            return 0*x+1
        return components_core(lst, coordinate, cm, func)

    elif key == "sn":

        def func(x):
            return x
        return components_core(lst, coordinate, cm, func)

    elif key == "rsn":
        max_value = max(list(map(list, zip(*lst)))[3])

        def func(x):
            return 2-x/max_value
        return components_core(lst, coordinate, cm, func)


def mapping_rule_standard(lst, cm, key, grid_num):
    if key == "unit":
        def func(x):
            return 0*x+1
        return direct_map_core(lst, cm, func, grid_num)

    elif key == "sn":
        def func(x):
            return x
        return direct_map_core(lst, cm, func, grid_num)

    elif key == "rsn":
        max_value = max(list(map(list, zip(*lst)))[3])
        def func(x):
            return 2-x/max_value
        return direct_map_core(lst, cm, func, grid_num)


def direct_map_core(lst, cm, map_value, grid_num):
    main_vector = lst[:, 0:3]
    value_vector = map_value(lst[:, 3])[:, np.newaxis]
    standard_vector = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])

    main_vector[:, 0] = main_vector[:, 0] - cm[0]
    main_vector[:, 1] = main_vector[:, 1] - cm[1]
    main_vector[:, 2] = main_vector[:, 2] - cm[2]
    inner_product = np.dot(main_vector, standard_vector)
    r_max = np.sqrt(np.power(inner_product[:, 0], 2)+np.power(inner_product[:, 1], 2)+np.power(inner_product[:, 2], 2))
    r_max = np.max(r_max)

    components = np.hstack((inner_product, value_vector))
    maps = mapping_core(components, grid_num, r_max)

    return r_max, maps


def components_core(lst, coordinate, cm, map_value):
    main_vector = lst[:, 0:3]
    value_vector = map_value(lst[:, 3])[:, np.newaxis]

    main_vector[:, 0] = main_vector[:, 0] - cm[0]
    main_vector[:, 1] = main_vector[:, 1] - cm[1]
    main_vector[:, 2] = main_vector[:, 2] - cm[2]
    inner_product = np.dot(main_vector, coordinate)
    r_max = np.sqrt(np.power(inner_product[:, 0], 2)+np.power(inner_product[:, 1], 2)+np.power(inner_product[:, 2], 2))
    r_max = np.max(r_max)

    components = np.hstack((inner_product, value_vector))

    return r_max, components


@jit(nopython=True)
def mapping_flow(x1, y1, z1, map_value, total_grid_number, length):
    # mapping rule
    mp = diving(x1, y1, 1.0, total_grid_number)
    xy_map = np.zeros((total_grid_number, total_grid_number), dtype=float)
    for i in range(length):
        if map_value[i] >= 0:
            if (mp[i][0] >= total_grid_number) or (mp[i][0] < 0) or (mp[i][1] >= total_grid_number) or (mp[i][1] < 0):
                continue
            if xy_map[mp[i][0]][mp[i][1]] < map_value[i]:
                xy_map[mp[i][0]][mp[i][1]] = map_value[i]

    mp = diving(y1, z1, 1.0, total_grid_number)
    yz_map = np.zeros((total_grid_number, total_grid_number), dtype=float)
    for i in range(length):
        if map_value[i] >= 0:
            if (mp[i][0] >= total_grid_number) or (mp[i][0] < 0) or (mp[i][1] >= total_grid_number) or (mp[i][1] < 0):
                continue
            if yz_map[mp[i][0]][mp[i][1]] < map_value[i]:
                yz_map[mp[i][0]][mp[i][1]] = map_value[i]

    mp = diving(z1, x1, 1.0, total_grid_number)
    zx_map = np.zeros((total_grid_number, total_grid_number), dtype=float)
    for i in range(length):
        if map_value[i] >= 0:
            if (mp[i][0] >= total_grid_number) or (mp[i][0] < 0) or (mp[i][1] >= total_grid_number) or (mp[i][1] < 0):
                continue
            if zx_map[mp[i][0]][mp[i][1]] < map_value[i]:
                zx_map[mp[i][0]][mp[i][1]] = map_value[i]

    return yz_map, zx_map, xy_map


def mapping_core(lst, total_grid_number, r_max):
    x1 = lst[:, 0]/r_max
    y1 = lst[:, 1]/r_max
    z1 = lst[:, 2]/r_max
    map_value = lst[:, 3]

    yz_map, zx_map, xy_map = mapping_flow(x1, y1, z1, map_value, total_grid_number, lst.shape[0])

    total_map = np.array([yz_map, zx_map, xy_map])

    return total_map


def figure_output(nrn_ID, maps, res, N, save_path, key_w, sub_name, I_dict):
    x = np.arange(0, N, 1)
    y = np.arange(0, N, 1)
    X, Y = np.meshgrid(x, y)
    fig = figure.Figure()
    ax = fig.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            ax[i][j].pcolormesh(X, Y, maps[i][j], vmin=np.min(maps[i][j]), vmax=np.max(maps[i][j]), shading='auto')
            ax[i][j].set_title(r"$I_%d: %.3f$" % (j+1, I_dict[key_w][nrn_ID[i]][j]))
            if i == 1:
                ax[i][j].set_xlabel(sub_name[j])
            if j == 0:
                ax[i][j].set_ylabel(nrn_ID[i])

    fig.suptitle("score: %.3f, orientation: %s" % (res[1], res[0]), y=0.96, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(save_path + nrn_ID[0] + "_" + nrn_ID[1] + "_" + key_w + "_" + res[0] + ".png", dpi=100)
    fig.clf()
    plt.close()

    return None


def find_file(s1, path):
    filenames = []
    for i in os.listdir(path):
        try:
            if os.path.splitext(i)[-1] == s1:
                filenames.append(i)
        except:
            pass
    return filenames


def change_name(s1, s2, path):
    for i in find_file(s1, path):
        os.renames(path + i, path + os.path.splitext(i)[0] + s2)


def index_search(lst, value):
    if value == -1:
        return -1
    index = -1
    for i in range(len(lst)):
        if lst[i][0] == value:
            index = i
            break
    #    if index == -1:
    #        print('index_search error')
    return index


def tree_builder_original(path, name, length_th,
                          compress_list=True):
    # todo: KTC perf
    # load the file and convert it into useful format
    recoTxt(path+name)

    # load the converted txt
    neuron_info = pd.read_csv(path + name + ".swc", sep=" ")
    if 'fk' in name:  # for vibration fake data
        neuron_info = neuron_info.drop(index=0)
        neuron_info = neuron_info.reset_index(drop=True)

    # create the list of neurons --> 0~6
    input_text = open(path + name + ".swc", 'r')
    nrn_list = [line.split(' ') for line in input_text.readlines()]
    nrn_list.pop(0)

    # for vibration fake data
    if 'fk' in name:
        nrn_list.pop(0)

    # type categorization
    for i in range(len(nrn_list)):
        for j in range(len(nrn_list[i])):
            if '.' in nrn_list[i][j]:
                nrn_list[i][j] = float(nrn_list[i][j])
            else:
                nrn_list[i][j] = int(nrn_list[i][j])

    # ID checker
    if nrn_list[-1][0] != len(nrn_list):
        return "error"

    # label --> 0 : unlabel
    #           1 : soma
    #           2 : axon
    #           3 : dendrite
    for i in range(len(nrn_list)):
        if nrn_list[i][1] == 0:
            nrn_list[i][1] = 0
            continue
        elif nrn_list[i][1] == 1:
            nrn_list[i][1] = 1
            continue
        elif nrn_list[i][1] == 2 or nrn_list[i][1] // 10 == 2 or nrn_list[i][1] == -2:
            nrn_list[i][1] = 2
        elif nrn_list[i][1] == 3 or nrn_list[i][1] // 10 == 3 or nrn_list[i][1] == -3:
            nrn_list[i][1] = 3
        else:
            nrn_list[i][1] = 0

    # delete the column 5 (the diameter is empty)
    nrn_list = list(map(list, zip(*nrn_list)))
    nrn_list.pop(5)
    #nrn_list.insert(1, [name for i in range(len(nrn_list[0]))])

    # create the CN column --> 6, 7
    nrn_list.append([0 for i in range(len(nrn_list[0]))])
    nrn_list.append([0.0 for i in range(len(nrn_list[0]))])
    nrn_list = list(map(list, zip(*nrn_list)))
    for i in range(1, len(nrn_list)):
        index = index_search(nrn_list, nrn_list[i][5])
        nrn_list[index][6] += 1
        nrn_list[i][7] += NN_dis(nrn_list[index], nrn_list[i])

    # create the layer column _ Strahler number --> 8
    index_lst = []
    for i in range(len(nrn_list)):
        if nrn_list[i][6] == 0:
            nrn_list[i].append(1)
            index_lst.append(i)
    not_fullfilled = True
    while not_fullfilled:
        p_index = []
        temp_lst = copy.copy(index_lst)
        for i in index_lst:
            p_index.append(nrn_list[i][5]-1)
        p_index_set = list(set(p_index))
        for i in p_index_set:
            if nrn_list[i][6] == p_index.count(i):
                index_lst.append(i)
                c_index = [temp_lst[index] for index, v in enumerate(p_index) if v == i]
                temp = []
                for j in c_index:
                    temp.append(nrn_list[j][8])
                if temp.count(max(temp)) >= 2:
                    nrn_list[i].append(max(temp)+1)
                else:
                    nrn_list[i].append(max(temp))
                for j in c_index:
                    index_lst.remove(j)
        # check
        if len(nrn_list[0]) == 9:
            not_fullfilled = False

    del temp, temp_lst

    # interpolation
    while True:
        check = False
        for i in range(len(nrn_list)-1, -1, -1):
            if len(nrn_list[i]) != 9:
                check = True
                nrn_list[i].append(nrn_list[i+1][-1])
        if not check:
            break
    append_lst = []
    for i in range(1, len(nrn_list)-1):
        if nrn_list[i][7] > length_th:
            p_index = index_search(nrn_list, nrn_list[i][5])
            length = nrn_list[i][7]
            num = int(length // length_th + 1)
            delta_x = (nrn_list[i][2]-nrn_list[p_index][2])/num
            delta_y = (nrn_list[i][3]-nrn_list[p_index][3])/num
            delta_z = (nrn_list[i][4]-nrn_list[p_index][4])/num
            for j in range(num):
                temp = copy.deepcopy(nrn_list[p_index])
                temp[8] = nrn_list[i][8]
                temp[2] += delta_x*(j+1)
                temp[3] += delta_y*(j+1)
                temp[4] += delta_z*(j+1)
                append_lst.append(temp)

    nrn_list = pd.DataFrame(nrn_list, columns=["ID", "type", "x", "y", "z", "parent_ID", "CN", "l",
                                               "sn"]).sort_values(by="ID")
    # compact form
    if compress_list:
        nrn_list = nrn_list.loc[:, ["x", "y", "z", "CN", "sn"]].astype({"x": "float32", "y": "float32", "z": "float32",
                                                                        "CN": "int32", "sn": "int32"})

    return nrn_list


def tree_builder(path, name, length_th,
                 compress_list=True):
    # load the file and convert it into useful format
    recoTxt(path+name)

    # create the list of neurons --> 0~6
    nrn_list = pd.read_csv(path + name + ".swc", sep=" ").sort_values(by="#n").values.tolist()

    # todo KTC: general case
    # ID checker
    if nrn_list[-1][0] != len(nrn_list):
        return "error"

    # delete the column 1, 5
    nrn_list = list(map(list, zip(*nrn_list)))
    nrn_list.pop(5)
    nrn_list.pop(1)

    # add parameters
    nrn_list[0] = [int(i) for i in nrn_list[0]]
    nrn_list[-1] = [int(i) for i in nrn_list[-1]]
    nrn_list.append([0 for i in range(len(nrn_list[0]))])
    nrn_list.append([0.0 for i in range(len(nrn_list[0]))])
    nrn_list.append([0 for i in range(len(nrn_list[0]))])
    nrn_list = list(map(list, zip(*nrn_list)))
    nrn_dict = {lst[0]: lst[1:] for lst in nrn_list}
    del nrn_list

    # create the CN column --> 4
    for i in nrn_dict.keys():
        if nrn_dict[i][3] != -1:
            nrn_dict[nrn_dict[i][3]][4] += 1

    # create distance column --> 5
    for i in nrn_dict.keys():
        if nrn_dict[i][3] != -1:
            nrn_dict[i][5] += np.sqrt(np.power(nrn_dict[i][0]-nrn_dict[nrn_dict[i][3]][0], 2) +
                                      np.power(nrn_dict[i][1]-nrn_dict[nrn_dict[i][3]][1], 2) +
                                      np.power(nrn_dict[i][2]-nrn_dict[nrn_dict[i][3]][2], 2))

    # create Strahler order --> 6
    index_lst = []
    for i in nrn_dict.keys():
        if nrn_dict[i][4] == 0:
            nrn_dict[i][6] = 1
            index_lst.append(i)
    not_fullfilled = True
    while not_fullfilled:
        p_index = []
        temp_lst = copy.copy(index_lst)
        for i in index_lst:
            p_index.append(nrn_dict[i][3])
        p_index_set = list(set(p_index))
        for k in p_index_set:
            i = int(k)
            if nrn_dict[i][4] == p_index.count(i):
                index_lst.append(i)
                c_index = [temp_lst[index] for index, v in enumerate(p_index) if v == i]
                temp = []
                for j in c_index:
                    temp.append(nrn_dict[j][6])
                if temp.count(max(temp)) >= 2:
                    nrn_dict[i][6] = max(temp)+1
                else:
                    nrn_dict[i][6] = max(temp)
                for j in c_index:
                    index_lst.remove(j)
        if nrn_dict[1][6] != 0:
            not_fullfilled = False
    del temp, temp_lst

    # interpolation
    nrn_list = []
    for i in nrn_dict.keys():
        if nrn_dict[i][3] != -1:
            if nrn_dict[i][5] > length_th:
                num = int(nrn_dict[i][5] // length_th + 1)
                delta_x = (nrn_dict[i][0]-nrn_dict[nrn_dict[i][3]][0])/num
                delta_y = (nrn_dict[i][1]-nrn_dict[nrn_dict[i][3]][1])/num
                delta_z = (nrn_dict[i][2]-nrn_dict[nrn_dict[i][3]][2])/num
                for j in range(num-1):
                    temp = [i, nrn_dict[nrn_dict[i][3]][0]+delta_x*(j+1), nrn_dict[nrn_dict[i][3]][1]+delta_y*(j+1), nrn_dict[nrn_dict[i][3]][2]+delta_z*(j+1),
                            nrn_dict[i][3], nrn_dict[i][4], nrn_dict[i][5], nrn_dict[i][6]]
                    nrn_list.append(temp)
    for item in nrn_dict.items():
        nrn_list.append([item[0]]+item[1])

    nrn_list = pd.DataFrame(nrn_list, columns=["ID", "x", "y", "z", "parent_ID", "CN", "l", "sn"]).sort_values(by="ID").reset_index(drop=True)

    # compact form & type categorization
    if compress_list:
        nrn_list = nrn_list.loc[:, ["x", "y", "z", "CN", "sn"]].astype({"x": "float32", "y": "float32", "z": "float32",
                                                                        "CN": "int32", "sn": "int32"})

    return nrn_list


def swc_plot(src_path, swc_name, plot_path, plot_name):

    neu = pd.read_csv(src_path + swc_name + ".swc", delim_whitespace=True, header=None, comment='#')
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for ind, row in neu.iterrows():
        if row[1] == 1:
            ax.scatter(row[2], row[3], row[4], color='black', s=20)
        else:
            ax.scatter(row[2], row[3], row[4], color='black', alpha=0.2, s=1)

    ax.set_title(swc_name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(azim=-50, elev=20)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path + plot_name + ".png", dpi=600)
    plt.close()


def swc_plot2(src_path, swc_name, plot_path, plot_name):
    with open(src_path + swc_name + ".pkl", "rb") as file:
        _df = pickle.load(file)
    lst = _df.loc[:, ["x", "y", "z"]].values.tolist()
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(len(lst)):
        if i == 0:
            ax.scatter(lst[i][0], lst[i][1], lst[i][2], color='black', s=20)
        else:
            ax.scatter(lst[i][0], lst[i][1], lst[i][2], color='black', alpha=0.2, s=1)

    ax.set_title(swc_name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(azim=-50, elev=20)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path + plot_name + ".png", dpi=600)
    plt.close()


def load_swc(path_dict, clear, overwrite, length_dict, plot=False):
    keys_g = list(length_dict)

    # wipe out
    if clear:
        try:
            paths = os.getcwd()+path_dict["sep"]+"data"+path_dict["sep"]
            files = os.listdir(paths)

            # Exception List
            files.remove("raw_data")
            files.remove("selected_data")
            files.remove("statistical_results")

            for i in files:
                if os.path.splitext(i)[0] == ".gitkeep":
                    continue
                elif os.path.splitext(i)[1] == "":
                    shutil.rmtree(os.getcwd()+path_dict["sep"]+"data"+path_dict["sep"]+i)
                    os.makedirs(os.getcwd()+path_dict["sep"]+"data"+path_dict["sep"]+i)
                    shutil.copyfile(os.getcwd()+path_dict["sep"]+"data"+path_dict["sep"]+".gitkeep",
                                    os.getcwd()+path_dict["sep"]+"data"+path_dict["sep"]+i+path_dict["sep"]+".gitkeep")
                else:
                    os.remove(os.getcwd()+path_dict["sep"]+"data"+path_dict["sep"]+i)

            if not os.path.isdir(path_dict["convert"]):
                os.makedirs(path_dict["convert"])

        except OSError as _:
            print(_)

        else:
            print("clear !")

    # process begin
    time.sleep(0.5)
    print("Tree Builder: ")
    time.sleep(0.5)

    # load data
    group_dict = {}  # direct to data matching
    for key in keys_g:
        time.sleep(0.5)
        print("  Region : ", key)
        time.sleep(0.5)

        _files = os.listdir(path_dict["import_swc"] + key + path_dict["sep"])

        # file extension: swc
        for _file in _files:
            if os.path.splitext(_file)[-1] != ".swc":
                _files.remove(_file)

        for i in range(len(_files)):
            _files[i] = os.path.splitext(_files[i])[0]

        group_dict[key] = _files

        if overwrite:
            error_nrn = ""
            for name in tqdm(_files):
                try:
                    _df = tree_builder(path_dict["import_swc"] + key + path_dict["sep"], name, length_dict[key])
                    if plot:
                        swc_plot(path_dict["import_swc"] + key + path_dict["sep"], name,
                                 path_dict["plot_single_neuron"], name+"_original")
                        swc_plot2(path_dict["convert"], name, path_dict["plot_single_neuron"], name+"_convert")
                    if type(_df) == str:
                        error_nrn += name+", "
                        group_dict[key].remove(name)
                        shutil.move(path_dict["import_swc"] + key + path_dict["sep"] + name + ".swc",
                                    path_dict["forbidden"] + name + ".swc")

                    else:
                        with open(path_dict["convert"] + name + ".pkl", "wb") as file:
                            pickle.dump(_df, file)

                except:
                    print("error: ", name)
                    group_dict[key].remove(name)

            if len(error_nrn) != 0:
                print("disconnected neurons: ", error_nrn[:-2])

        else:
            # check existing files
            check_list = [os.path.splitext(i)[0] for i in os.listdir(path_dict["convert"])]

            # ignore existing files
            _files = list(set(_files)-set(check_list))
            if len(_files) == 0:
                print("    Done!")
                continue

            error_nrn = ""
            for name in tqdm(_files):
                try:
                    _df = tree_builder(path_dict["import_swc"] + key + path_dict["sep"], name, length_dict[key])
                    if type(_df) == str:
                        error_nrn += name+", "
                        group_dict[key].remove(name)
                        shutil.move(path_dict["import_swc"] + key + path_dict["sep"] + name + ".swc",
                                    path_dict["forbidden"] + name + ".swc")
                    else:
                        with open(path_dict["convert"] + name + ".pkl", "wb") as file:
                            pickle.dump(_df, file)

                except:
                    print("error: ", name)
                    group_dict[key].remove(name)

            if len(error_nrn) != 0:
                print("disconnected neurons: ", error_nrn[:-2])

    with open(path_dict["stats"] + "group_dict.pkl", "wb") as file:
        pickle.dump(group_dict, file)
    file_list = list(chain.from_iterable(group_dict.values()))

    return file_list


def I_distribution_plot(file_region, file_dict, region_split=False):
    with open(file_region, "rb") as file:
        region = pickle.load(file)
    with open(file_dict, "rb") as file:
        nrn = pickle.load(file)

    # plot
    fig, ax = plt.subplots()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    if region_split:
        count = 0
        for key1 in region.keys():
            _lst = [[], [], []]
            for key2 in region[key1]:
                _x, _y, _z = nrn[key2]
                _lst[0].append(_x)
                _lst[1].append(_y)
                _lst[2].append(_z)
            x, y = _lst[0], _lst[1]
            ax.scatter(x, y, c=colors[count], s=30.0, label=key1,
                       alpha=0.9, edgecolors='none')
            count += 1
    else:
        _lst = [[], [], []]
        for key in nrn.keys():
            _x, _y, _z = nrn[key]
            _lst[0].append(_x)
            _lst[1].append(_y)
            _lst[2].append(_z)
        x, y = _lst[1], _lst[2]
        ax.scatter(x, y, c=colors[0], s=10.0,
                   alpha=0.9, edgecolors='none')
    temp = max(max(x), max(y))
    xx = np.arange(0, temp+temp/200, temp/100)
    yy = np.arange(0, temp+temp/200, temp/100)
    plt.plot(xx, yy, color="tab:red")
    ax.set_xlabel(r"$I_2$", fontsize=20)
    ax.set_ylabel(r"$I_3$", fontsize=20)
    ax.set_xlim(0, temp)
    ax.set_ylim(0, temp)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def I_distribution_plot_3D(file_region, file_dict, region_split=False):
    with open(file_region, "rb") as file:
        region = pickle.load(file)
    with open(file_dict, "rb") as file:
        nrn = pickle.load(file)

    # plot
    ax = plt.subplot(projection='3d')
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    if region_split:
        count = 0
        for key1 in region.keys():
            _lst = [[], [], []]
            for key2 in region[key1]:
                _x, _y, _z = nrn[key2]
                _lst[0].append(_x)
                _lst[1].append(_y)
                _lst[2].append(_z)
            x, y, z = _lst[0], _lst[1], _lst[2]
            ax.scatter(x, y, z, c=colors[count], s=30.0, label=key1,
                       alpha=0.9, edgecolors='none')
            count += 1
        ax.legend()

    else:
        _lst = [[], [], []]
        for key in nrn.keys():
            _x, _y, _z = nrn[key]
            _lst[0].append(_x)
            _lst[1].append(_y)
            _lst[2].append(_z)
        x, y, z = _lst[0], _lst[1], _lst[2]
        ax.scatter(x, y, z, c=colors[0], s=10.0,
                   alpha=0.9, edgecolors='none')
    ax.grid(True)
    #plt.savefig(os.getcwd()+"\\plot\\idict.png")
    plt.show()
    plt.close()


def numpy_col_inner_many_to_one_join(ar1, ar2):
    # Select connectable rows of ar1 and ar2 (ie. ar1 last_col = ar2 first col)
    ar1 = ar1[np.in1d(ar1[:, -1], ar2[:, 0])]   # error occurred if ar1 is empty.
    ar2 = ar2[np.in1d(ar2[:, 0], ar1[:, -1])]

    # if int >= 0, else otherwise
    if 'int' in ar1.dtype.name and ar1[:, -1].min() >= 0:
        bins = np.bincount(ar1[:, -1])
        counts = bins[bins.nonzero()[0]]
    else:
        # order of np is "-int -> 0 -> +int -> other type"
        counts = np.unique(ar1[:, -1], False, False, True)[1]

    # Reorder array with np's order rule
    left = ar1[ar1[:, -1].argsort()]
    right = ar2[ar2[:, 0].argsort()]

    # Connect the rows of ar1 & ar2
    return np.concatenate([left[:, :-1],right[np.repeat(np.arange(right.shape[0]),counts)]], 1)


def trace_nodes(edges):
    # Top layer
    mask = np.in1d(edges[:, 1], edges[:, 0])    # parent with/without further ancestor
    gen_branches = edges[~mask]   # branch of parent without further ancestor
    edges = edges[mask]   # branch of otherwise
    yield gen_branches    # generate result

    # Successor layers
    while edges.size != 0:
        mask = np.in1d(edges[:, 1], edges[:, 0])    # parent with/without further ancestor
        next_gen = edges[~mask]   # branch of parent without further ancestor
        gen_branches = numpy_col_inner_many_to_one_join(next_gen, gen_branches)  # connect with further ancestors
        edges = edges[mask]   # branch of otherwise
        yield gen_branches    # generate result


def neuron_first_fork(root_lst, fork_lst, branch_lst):
    soma = root_lst[0]
    if soma in fork_lst:
        first_fork = soma
    else:
        result = [t for t in branch_lst if all([t[1] == soma, t[0] in fork_lst])]
        if len(result) == 1:
            first_fork = result[0][0]
        elif len(result) == 0:
            sys.exit("\n No fork in the neuron! Check 'neuron_first_fork()'.")
        else:
            sys.exit("\n Multiple first_fork in the neuron! Check 'neuron_first_fork()'.")

    return first_fork


def zero_list_maker(n):
    list_of_zeros = [0] * n
    return list_of_zeros


def neuron_tree_node_dict(df, child_col, parent_col, childNum_col='NC'):
    ### Create list of leaf/fork/root
    leaf_lst = df.loc[df[childNum_col] == 0, child_col].tolist()
    fork_lst = df.loc[df[childNum_col] > 1, child_col].tolist()
    root_lst = df.loc[df[parent_col] == -1, child_col].tolist()
    if len(root_lst) == 1:
        pass
    else:
        sys.exit("\n Multiple roots(somas) in a neuron. Check 'neuron_num_of_child()'.")

    tree_node_dict = {'root': root_lst, 'fork': fork_lst, 'leaf': leaf_lst}

    return tree_node_dict


def neuron_childNumCol(df, child_col, parent_col, output_col='NC'):
    ### Create child number col
    df_freq = pd.value_counts(df[parent_col]).to_frame().reset_index()
    df_freq.columns = [child_col, output_col]
    df_freq = df_freq.sort_values([child_col]).reset_index(drop=True)

    df = pd.merge(df, df_freq, how='left', on=child_col)
    df[output_col] = np.where(np.isnan(df[output_col]), 0, df[output_col])
    df[output_col] = df[output_col].astype(int)

    ### Create list of leaf/fork/root
    tree_node_dict = neuron_tree_node_dict(df, child_col, parent_col, childNum_col=output_col)

    return df, tree_node_dict


def neuron_ancestors_and_path(df, child_col, parent_col):
    df_anc = df[[child_col, parent_col]]

    ### Need to drop row that "PARENT_ID = -1" first
    df_anc = df_anc[df_anc[parent_col] != -1]

    edges = df_anc.values

    ancestors = []
    path = []
    for ar in trace_nodes(edges):
        ancestors.append(np.c_[np.repeat(ar[:, 0], ar.shape[1] - 1),
                               ar[:, 1:].flatten()])
        path.append(np.c_[np.repeat(ar[:, 0], ar.shape[1]),
                          ar[:, :].flatten()])

    return pd.DataFrame(np.concatenate(ancestors),columns=['descendant', 'ancestor']), \
           pd.DataFrame(np.concatenate(path),columns=['descendant', 'path'])

def neuron_level_branch(df_path, tree_node_dict):
    leaf_lst = tree_node_dict['leaf']
    fork_lst = tree_node_dict['fork']
    root_lst = tree_node_dict['root']
    ### Create branches (level tuple list)
    level_points = list(set().union(leaf_lst, fork_lst, root_lst))
    df_level = df_path.loc[df_path['path'].isin(level_points)]
    df_level = df_level.loc[df_level['descendant'].isin(leaf_lst)]
    df_level = df_level.reset_index(drop=True)
    # df_level = df_level.sort_values(['descendant', 'path']).reset_index(drop=True)

    branches = []
    for i in df_level.descendant.unique().tolist():
        lst = df_level.loc[df_level['descendant'] == i, 'path'].tolist()
        branches = list(set().union(branches, zip(lst, lst[1:])))

    return branches

def neuron_levelCol(df, df_anc, df_path, tree_node_dict, branch_lst, child_col, parent_col):
    ### cihld_col is from "df"
    leaf_lst = tree_node_dict['leaf']
    fork_lst = tree_node_dict['fork']
    root_lst = tree_node_dict['root']

    ### 1. Points btw root and first_fork
    first_fork = neuron_first_fork(root_lst, fork_lst, branch_lst)
    temp_lst = df_anc.loc[df_anc['descendant'] == first_fork, 'ancestor'].tolist()
    temp_lst.append(first_fork)
    zero_lst = zero_list_maker(len(temp_lst))
    df_temp_1 = pd.DataFrame({child_col: temp_lst, 'level': zero_lst})

    ### 2. Points after first_fork
    df_temp_2 = df_anc.loc[df_anc['ancestor'].isin(fork_lst)]
    df_temp_2 = pd.value_counts(df_temp_2.descendant).to_frame().reset_index()
    df_temp_2.columns = [child_col, 'level']

    ### Merge 1. & 2. to df
    df_temp_1 = pd.concat([df_temp_1, df_temp_2])
    df_temp_1 = df_temp_1.sort_values([child_col]).reset_index(drop=True)

    df = pd.merge(df, df_temp_1, how='left', on=child_col)

    max_level = max(df['level'])


    ### 3. Find deepest level of each point
    df['dp_level'] = 0
    # df_path = df_path.loc[df_path['path'] != 1]

    df_temp = df.loc[df[child_col].isin(leaf_lst), [child_col, 'level']].sort_values(['level'], ascending=False).reset_index(drop=True)
    for l in df_temp[child_col]:
        path = df_path.loc[df_path['descendant'] == l, 'path'].tolist()
        level = df_temp.loc[df_temp[child_col] == l, 'level'].values[0]
        df['dp_level'] = np.where((df[child_col].isin(path)) & (df['dp_level'] < level),
                                  level, df['dp_level'])



    return df, max_level, first_fork

def calculate_distance(positions, decimal=None, type='euclidean'):
    results = []

    # Detect dimension of tuples in the positions
    try:
        if all(len(tup) == 2 for tup in positions):
            dim = 2
        elif all(len(tup) == 3 for tup in positions):
            dim = 3
    except:
        print('Dimension of positions must be same in calculate_distance()!')


    # Calculate distance
    try:
        if all([dim == 2, type == 'haversine']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                lat1 = loc1[0]
                lng1 = loc1[1]

                lat2 = loc2[0]
                lng2 = loc2[1]

                degreesToRadians = (math.pi / 180)
                latrad1 = lat1 * degreesToRadians
                latrad2 = lat2 * degreesToRadians
                dlat = (lat2 - lat1) * degreesToRadians
                dlng = (lng2 - lng1) * degreesToRadians

                a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(latrad1) * \
                    math.cos(latrad2) * math.sin(dlng / 2) * math.sin(dlng / 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                r = 6371000

                results.append(r * c)

        elif all([dim == 2, type == 'euclidean']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                x1 = loc1[0]
                y1 = loc1[1]

                x2 = loc2[0]
                y2 = loc2[1]

                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                results.append(d)

        elif all([dim == 3, type == 'euclidean']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                x1 = loc1[0]
                y1 = loc1[1]
                z1 = loc1[2]

                x2 = loc2[0]
                y2 = loc2[1]
                z2 = loc2[2]

                d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

                results.append(d)


        if decimal is None:
            return sum(results)
        elif decimal == 0:
            return int(round(sum(results)))
        else:
            return round(sum(results), decimal)

    except:
        print('Please use available type and dim, such as "euclidean"(2-dim, 3-dim) and "haversine" (2-dim only), '
              'in calculate_distance().')

def neuron_branchCol_QCol_distance(df, df_path, tree_node_dict, branch_lst, first_fork,child_col, parent_col, type_col='T',decimal=0):
    ### Create distances of branches and create Q col
    length_lst = []    # distance of branch
    length_lst_soma = []  # distance of descendant to soma
    direct_dis_lst_soma = []  # direct distance of descendant to soma
    df_temp = pd.DataFrame()
    for i in branch_lst:
        start = i[0]
        end = i[1]
        soma = tree_node_dict['root'][0]

        path_points = df_path.loc[df_path['descendant'] == start, 'path'].tolist()

        start_idx = path_points.index(start)
        end_idx = path_points.index(end)
        soma_idx = path_points.index(soma)

        path_points_1 = path_points[start_idx: end_idx]         # exclude the end point(for Q)
        path_points_2 = path_points[start_idx: (end_idx + 1)]   # include the end point(for Q, dis)
        path_points_3 = path_points[start_idx: (soma_idx + 1)]  # path to soma


        # Create branch col and Q col
        # branch with end pt != 1 or first_fork == 1 (first_fork == soma)
        if any([end != 1, first_fork == 1]):
            temp_lst_1 = [start] * len(path_points_1)
            temp_lst_2 = list(range(len(path_points_1)))
            df_temp_1 = pd.DataFrame(OrderedDict({child_col: path_points_1, 'branch': temp_lst_1, 'Q': temp_lst_2}))
            df_temp = df_temp.append(df_temp_1)

        else:
            temp_lst_1 = [start] * len(path_points_2)
            temp_lst_2 = list(range(len(path_points_2)))
            df_temp_1 = pd.DataFrame(OrderedDict({child_col: path_points_2, 'branch': temp_lst_1, 'Q': temp_lst_2}))
            df_temp = df_temp.append(df_temp_1)

        # Calculate distance
        positions = df.loc[df[child_col].isin(path_points_2), ['x', 'y', 'z']]
        tuples = [tuple(x) for x in positions.values]
        positions_s = df.loc[df[child_col].isin(path_points_3), ['x', 'y', 'z']]
        tuples_s = [tuple(x) for x in positions_s.values]
        tuples_ds = [tuples_s[0], tuples_s[-1]]

        length = calculate_distance(tuples, decimal)
        length_lst.append(length)

        length_soma = calculate_distance(tuples_s, decimal)
        length_lst_soma.append(length_soma)

        direct_dis_soma = calculate_distance(tuples_ds, decimal)
        direct_dis_lst_soma.append(direct_dis_soma)

    # Merge branch & Q into original df
    df = pd.merge(df, df_temp, how='left', on=child_col)
    if first_fork == 1:
        df.loc[df[parent_col] == -1, ['branch', 'Q']] = [1, 0]  # add soma if first_fork = 1
    df[['branch', 'Q']] = df[['branch', 'Q']].astype('int')


    ### Create df_dis (cols ['len_des_soma', 'des_T'])
    df_dis = pd.DataFrame({'branch': branch_lst, 'len': length_lst, 'len_des_soma': length_lst_soma, 'direct_dis_des_soma': direct_dis_lst_soma})
    df_dis['ancestor'] = [tuple[1] for tuple in branch_lst]
    df_dis['descendant'] = [tuple[0] for tuple in branch_lst]
    df_dis = df_dis.sort_values(['ancestor', 'descendant']).reset_index(drop=True)

    # create type column
    df_t = df[[child_col, type_col]].copy()
    df_t.columns = ['descendant', 'des_T']
    df_dis = pd.merge(df_dis, df_t, how='left', on='descendant')


    ### Reorder columns
    df_dis = df_dis[['branch', 'descendant', 'ancestor', 'len', 'len_des_soma', 'direct_dis_des_soma', 'des_T']]

    return df, df_dis, length_lst

def list_unique(mylist):
    x = np.array(mylist)
    x = list(np.unique(x))
    x.sort()
    return x

def dict_merge_value(d, unique=True):
    lst = []
    for k, v in d.items():
        if lst is None:
            lst = v
        else:
            lst += v

    if unique:
        lst = list_unique(lst)

    return lst

def partition(lst, n=None, pct=None, shuffle_list=True):
    if shuffle_list:
        random.shuffle(lst)

    if all([n is not None, pct is None]):
        division = len(lst) / n
    elif all([n is None, pct > 0, pct < 1]):
        val = 1/pct
        n = round(val)
        division = len(lst)/n
    else:
        sys.exit("\n Use either Number(n=1, 2, 3...) or Percent(pct=0.1, 0.2, 0.3,...) to separate the list.")

    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


def swc_vibration(swc_path,save_path,num,vibrate_amplitude=1/3):
    filenames = os.listdir(swc_path)
    for i in range(len(filenames)):
        filenames[i] = os.path.splitext(filenames[i])[0]
    print('Vibration :')
    for swc in tqdm(filenames) :
        time.sleep(0.5)
        nrn = nm.io.swc.read(swc_path + swc + '.swc')
        df = pd.DataFrame(nrn.data_block, columns=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'])

        # nrn_name
        df["nrn"] = swc
        # Create child number col and dictionary of leaf/fork/root lists
        df, tree_node_dict = neuron_childNumCol(df, child_col='ID', parent_col='PARENT_ID')
        # Create ancestors and path dfs
        df_anc, df_path = neuron_ancestors_and_path(df,child_col='ID', parent_col='PARENT_ID')
        # Create branches (level tuple list)
        branch_lst = neuron_level_branch(df_path, tree_node_dict)
        # Count the level for each point
        df, max_level, first_fork = neuron_levelCol(df, df_anc, df_path, tree_node_dict, branch_lst,child_col='ID', parent_col='PARENT_ID')
        # Create distances of branches and create Q col
        df, df_dis, dis_lst = neuron_branchCol_QCol_distance(df, df_path, tree_node_dict, branch_lst, first_fork,decimal=None, child_col='ID', parent_col='PARENT_ID')
        tnd = tree_node_dict

        for fk_num in range(num) :
            df_0 = df.copy()
            df_dis_0 = df_dis.copy()
            df_0["nrn"] = str(swc) +'_fk'+ str(fk_num)
            df_dis_0["direct_dis_des_anc"] = 0

            nodes0 = dict_merge_value(tnd)
            nodes = partition(nodes0, 3, shuffle_list=True)

            vibrate = ["x", "y", "z"]

            # r0 = min(df_dis["len"])*vibrate_amplitude
            r0 = vibrate_amplitude
            r = [r0, -r0]

            # A.
            # 1. Vibration
            for i in range(3):
                _n = nodes[i]
                _v = random.choice(vibrate)
                _r = random.choice(r)

                df_0[_v] = np.where(df_0['ID'].isin(_n), df_0[_v]+_r, df_0[_v])

            # 2. Calculate the new distance
            for _idx in range(len(df_dis_0)):
                # Calculate distance
                _s = df_0.loc[df_0['ID']==tnd["root"][0], ['x', 'y', 'z']]
                _s = [tuple(x) for x in _s.values]

                _d0 = df_dis_0.loc[_idx, "descendant"]
                _d = df_0.loc[df_0['ID']==_d0, ['x', 'y', 'z']]
                _d = [tuple(x) for x in _d.values]

                _a0 = df_dis_0.loc[_idx, "ancestor"]
                _a = df_0.loc[df_0['ID'] == _a0, ['x', 'y', 'z']]
                _a = [tuple(x) for x in _a.values]

                tuples_ds = _s + _d
                _ds = calculate_distance(tuples_ds)
                df_dis_0["direct_dis_des_soma"] = np.where(df_dis_0["descendant"]==_d0, _ds, df_dis_0["direct_dis_des_soma"])

                tuples_dp = _a + _d
                _dp = calculate_distance(tuples_dp)
                df_dis_0.loc[_idx, "direct_dis_des_anc"] = _dp

            # 3. Find out nodes which violate the rule
            _df_dis = df_dis_0.loc[df_dis_0["len"] < df_dis_0["direct_dis_des_anc"]].reset_index(drop=True)

            # B.
            # Adjust those nodes which violate the rule
            # while len(_df_dis) > 0:
            #
            #     # 1. Recover x,y,z from vibration
            #     for _idx in range(len(_df_dis)):
            #         _d0 = _df_dis.loc[_idx, "descendant"]
            #         _a0 = _df_dis.loc[_idx, "ancestor"]
            #
            #         for _n0 in [_d0, _a0]:
            #             _n = df.loc[df['ID'] == _n0, ['x', 'y', 'z']].values
            #             _row = df_0.index[df['ID'] == _n0].tolist()[0]
            #             df_0.loc[_row, "x"] = _n[0, 0]
            #             df_0.loc[_row, "y"] = _n[0, 1]
            #             df_0.loc[_row, "z"] = _n[0, 2]
            #
            #     # 2. Calculate the new distance
            #     for _idx in range(len(df_dis_0)):
            #         # Calculate distance
            #         _s = df_0.loc[df_0['ID'] == tnd["root"][0], ['x', 'y', 'z']]
            #         _s = [tuple(x) for x in _s.values]
            #
            #         _d0 = df_dis_0.loc[_idx, "descendant"]
            #         _d = df_0.loc[df_0['ID'] == _d0, ['x', 'y', 'z']]
            #         _d = [tuple(x) for x in _d.values]
            #
            #         _a0 = df_dis_0.loc[_idx, "ancestor"]
            #         _a = df_0.loc[df_0['ID'] == _a0, ['x', 'y', 'z']]
            #         _a = [tuple(x) for x in _a.values]
            #
            #         tuples_ds = _s + _d
            #         _ds = calculate_distance(tuples_ds)
            #         df_dis_0["direct_dis_des_soma"] = np.where(df_dis_0["descendant"] == _d0, _ds,
            #                                                     df_dis_0["direct_dis_des_soma"])
            #
            #         tuples_dp = _a + _d
            #         _dp = calculate_distance(tuples_dp)
            #         df_dis_0.loc[_idx, "direct_dis_des_anc"] = _dp
            #
            #     # 3. Find out nodes which violate the rule
            #     _df_dis = df_dis_0.loc[df_dis_0["len"] < df_dis_0["direct_dis_des_anc"]].reset_index(drop=True)

            _df_dis = _df_dis.drop(["direct_dis_des_anc"], 1)

            # save data
            df_0 = df_0.drop(columns=['nrn','NC','level','dp_level','branch','Q'])
            df_0 = df_0.sort_index()
            df_0[['ID','T','PARENT_ID']] = df_0[['ID','T','PARENT_ID']].astype(int)
            df_0 = df_0[['ID','T','x','y','z','R','PARENT_ID']]
            df_0.to_csv(str(save_path)+str(swc)+'_fk'+str(fk_num)+".swc",index=False,sep=' ')

    time.sleep(0.5)
    return

def reflection_core(mapp, option):
    # generate the reflected map
    _map = copy.deepcopy(mapp)
    for axis in option:
        _map[axis] = np.flip(_map[axis], 1)
        _map[(axis+2) % 3] = np.flip(_map[(axis+2) % 3], 0)
    return _map


def rotation_core(mapp, option):
    # generate the rotation map
    _map = copy.deepcopy(mapp)
    for i in range(3):
        _map[i] = _map[i].transpose()
    ex_index = [(option[0]) % 3, (option[1]) % 3]

    temp1 = copy.deepcopy(_map[ex_index[0]])
    temp2 = copy.deepcopy(_map[ex_index[1]])
    _map[ex_index[0]], _map[ex_index[1]] = temp2, temp1
    return _map


def orientation_key(key):
    indexR_lst = [[], [0, 1], [0, 2], [1, 2]]
    indexL_lst = [[0], [1], [2], [0, 1, 2]]

    if "ex" in key:
        option = [int(key[-1])-1, int(key[-1])]
        if "R" in key:
            return lambda x: reflection_core(rotation_core(x, option), indexR_lst[int(key[1])-1])
        elif "L" in key:
            return lambda x: reflection_core(rotation_core(x, option), indexL_lst[int(key[1])-1])
    else:
        if "R" in key:
            return lambda x: reflection_core(x, indexR_lst[int(key[1])-1])
        elif "L" in key:
            return lambda x: reflection_core(x, indexL_lst[int(key[1])-1])

    return None


def load_data_CNN(region_dict, info_dict, load_path):
    data_input = []
    data_label = []

    # load the map data in the file list of map_dict
    nrn_list = []
    for nrns in region_dict.values():
        for i in range(len(nrns)):
            nrn_list.append(nrns[i])
    region_dict = {vi: k for k, v in region_dict.items() for vi in v}
    key_list = list(info_dict.keys())
    map_dict = {}

    # laod maps
    print("load maps: ")
    time.sleep(0.5)
    for nrn_ID in tqdm(nrn_list):
        with open(load_path + nrn_ID + ".pkl", "rb") as file:
            nrn = pickle.load(file)
        map_dict[nrn_ID] = {}
        for key in key_list:
            map_dict[nrn_ID][key] = nrn[key]
    del nrn

    # create pairs of neurons
    _nrn_list = list(it.combinations(nrn_list, 2))
    nrn_list = []
    region_list = []

    # transport the information list in each weighting method
    _info_dict = {}
    for key in info_dict.keys():
        _info_dict[key] = list(map(list, zip(*info_dict[key])))

    # check the order of pairs
    for i in range(len(_nrn_list)):
        nrn1 = _nrn_list[i][0]
        nrn2 = _nrn_list[i][1]
        region1 = region_dict[nrn1]
        region2 = region_dict[nrn2]

        # create data --> pair of neurons
        if "fc" in region1 and "em" in region2:
            # name of neurons
            nrn_list.append([nrn1, nrn2])
  
            # Region of neuron
            region_list.append(region1 + region2)

            # input of neurons
            for i in range(len(key_list)):
                if nrn1 + "_" + nrn2 in _info_dict[key_list[i]][0]:
                    index = _info_dict[key_list[i]][0].index(nrn1 + "_" + nrn2)
                    right_key = _info_dict[key_list[i]][1][index]

                    if i == 0:
                        _map = map_dict[nrn1][key_list[i]]
                    else:
                        _map = np.vstack((_map, map_dict[nrn1][key_list[i]]))
                    operation = orientation_key(right_key)
                    _map = np.vstack((_map, operation(map_dict[nrn2][key_list[i]])))
                elif nrn2 + "_" + nrn1 in _info_dict[key_list[i]][0]:
                    index = _info_dict[key_list[i]][0].index(nrn2 + "_" + nrn1)
                    right_key = _info_dict[key_list[i]][1][index]

                    operation = orientation_key(right_key)
                    if i == 0:
                        _map = operation(map_dict[nrn1][key_list[i]])
                    else:
                        _map = np.vstack((_map, operation(map_dict[nrn1][key_list[i]])))
                    _map = np.vstack((_map, map_dict[nrn2][key_list[i]]))
                else:
                    print("lack of ranking data")

            data_input.append(_map)

            # label of neurons
            if region1[region1.index("c")+1:] == region2[region2.index("m")+1:]:
                data_label.append(1.0)
            else:
                data_label.append(0.0)

        elif "fc" in region2 and "em" in region1:
            # name of neurons
            nrn_list.append([nrn2, nrn1])

            # Region of neuron
            region_list.append(region2 + region1)

            # input of neurons
            for i in range(len(key_list)):
                if nrn1 + "_" + nrn2 in _info_dict[key_list[i]][0]:
                    index = _info_dict[key_list[i]][0].index(nrn1 + "_" + nrn2)
                    right_key = _info_dict[key_list[i]][1][index]

                    operation = orientation_key(right_key)
                    if i == 0:
                        _map = operation(map_dict[nrn2][key_list[i]])
                    else:
                        _map = np.vstack((_map, operation(map_dict[nrn2][key_list[i]])))
                    _map = np.vstack((_map, map_dict[nrn1][key_list[i]]))
                elif nrn2 + "_" + nrn1 in _info_dict[key_list[i]][0]:
                    index = _info_dict[key_list[i]][0].index(nrn2 + "_" + nrn1)
                    right_key = _info_dict[key_list[i]][1][index]

                    if i == 0:
                        _map = map_dict[nrn2][key_list[i]]
                    else:
                        _map = np.vstack((_map, map_dict[nrn2][key_list[i]]))
                    operation = orientation_key(right_key)
                    _map = np.vstack((_map, operation(map_dict[nrn1][key_list[i]])))

                else:
                    print("lack of ranking data")

            data_input.append(_map)

            # label of neurons
            if region2[region2.index("c")+1:] == region1[region1.index("m")+1:]:
                data_label.append(1.0)
            else:
                data_label.append(0.0)

    # change into the data format of array
    data_input = np.array(data_input)
    data_label = np.array(data_label)

    return data_input, data_label, region_list, nrn_list


def NCWH_to_NWHC(array):
    new_array = np.zeros((array.shape[0], array.shape[2], array.shape[3], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, :, :, j] = array[i, j, :, :]

    return new_array


def my_loss(ground_truth, prediction, gamma=2):
    alpha = 10 / 10
    condition1 = tf.equal(ground_truth, tf.constant(1.0))
    condition2 = tf.equal(ground_truth, tf.constant(0.0))
    # condition = tf.equal(ground_truth, tf.constant(1.0))

    prob = tf.abs(tf.subtract(tf.add(prediction, ground_truth), tf.constant(1.0)))

    true_res1 = 1.0 * tf.multiply(tf.pow(tf.math.abs(tf.subtract(1.0, prob)), gamma), tf.math.log(prob))
    true_res2 = 1.0 * alpha * tf.multiply(tf.pow(tf.math.abs(tf.subtract(1.0, prob)), gamma), tf.math.log(prob))
    false_res = tf.add(tf.multiply(0.0, ground_truth), 0.0)

    res = tf.add(tf.where(condition1, true_res1, false_res),
                 tf.where(condition2, true_res2, false_res))

    return res


def numpy_col_inner_many_to_one_join(ar1, ar2):
    # Select connectable rows of ar1 and ar2 (ie. ar1 last_col = ar2 first col)
    ar1 = ar1[np.in1d(ar1[:, -1], ar2[:, 0])]   # error occurred if ar1 is empty.
    ar2 = ar2[np.in1d(ar2[:, 0], ar1[:, -1])]

    # if int >= 0, else otherwise
    if 'int' in ar1.dtype.name and ar1[:, -1].min() >= 0:
        bins = np.bincount(ar1[:, -1])
        counts = bins[bins.nonzero()[0]]
    else:
        # order of np is "-int -> 0 -> +int -> other type"
        counts = np.unique(ar1[:, -1], False, False, True)[1]

    # Reorder array with np's order rule
    left = ar1[ar1[:, -1].argsort()]
    right = ar2[ar2[:, 0].argsort()]

    # Connect the rows of ar1 & ar2
    return np.concatenate([left[:, :-1],right[np.repeat(np.arange(right.shape[0]),counts)]], 1)


def trace_nodes(edges):
    # Top layer
    mask = np.in1d(edges[:, 1], edges[:, 0])    # parent with/without further ancestor
    gen_branches = edges[~mask]   # branch of parent without further ancestor
    edges = edges[mask]   # branch of otherwise
    yield gen_branches    # generate result

    # Successor layers
    while edges.size != 0:
        mask = np.in1d(edges[:, 1], edges[:, 0])    # parent with/without further ancestor
        next_gen = edges[~mask]   # branch of parent without further ancestor
        gen_branches = numpy_col_inner_many_to_one_join(next_gen, gen_branches)  # connect with further ancestors
        edges = edges[mask]   # branch of otherwise
        yield gen_branches    # generate result


def neuron_first_fork(root_lst, fork_lst, branch_lst):
    soma = root_lst[0]
    if soma in fork_lst:
        first_fork = soma
    else:
        result = [t for t in branch_lst if all([t[1] == soma, t[0] in fork_lst])]
        if len(result) == 1:
            first_fork = result[0][0]
        elif len(result) == 0:
            sys.exit("\n No fork in the neuron! Check 'neuron_first_fork()'.")
        else:
            sys.exit("\n Multiple first_fork in the neuron! Check 'neuron_first_fork()'.")

    return first_fork


def zero_list_maker(n):
    list_of_zeros = [0] * n
    return list_of_zeros


def neuron_tree_node_dict(df, child_col, parent_col, childNum_col='NC'):
    ### Create list of leaf/fork/root
    leaf_lst = df.loc[df[childNum_col] == 0, child_col].tolist()
    fork_lst = df.loc[df[childNum_col] > 1, child_col].tolist()
    root_lst = df.loc[df[parent_col] == -1, child_col].tolist()
    if len(root_lst) == 1:
        pass
    else:
        sys.exit("\n Multiple roots(somas) in a neuron. Check 'neuron_num_of_child()'.")

    tree_node_dict = {'root': root_lst, 'fork': fork_lst, 'leaf': leaf_lst}

    return tree_node_dict


def neuron_childNumCol(df, child_col, parent_col, output_col='NC'):
    ### Create child number col
    df_freq = pd.value_counts(df[parent_col]).to_frame().reset_index()
    df_freq.columns = [child_col, output_col]
    df_freq = df_freq.sort_values([child_col]).reset_index(drop=True)

    df = pd.merge(df, df_freq, how='left', on=child_col)
    df[output_col] = np.where(np.isnan(df[output_col]), 0, df[output_col])
    df[output_col] = df[output_col].astype(int)

    ### Create list of leaf/fork/root
    tree_node_dict = neuron_tree_node_dict(df, child_col, parent_col, childNum_col=output_col)

    return df, tree_node_dict


def neuron_ancestors_and_path(df, child_col, parent_col):
    df_anc = df[[child_col, parent_col]]

    ### Need to drop row that "PARENT_ID = -1" first
    df_anc = df_anc[df_anc[parent_col] != -1]

    edges = df_anc.values

    ancestors = []
    path = []
    for ar in trace_nodes(edges):
        ancestors.append(np.c_[np.repeat(ar[:, 0], ar.shape[1] - 1),
                               ar[:, 1:].flatten()])
        path.append(np.c_[np.repeat(ar[:, 0], ar.shape[1]),
                          ar[:, :].flatten()])

    return pd.DataFrame(np.concatenate(ancestors),columns=['descendant', 'ancestor']), \
           pd.DataFrame(np.concatenate(path),columns=['descendant', 'path'])


def neuron_level_branch(df_path, tree_node_dict):
    leaf_lst = tree_node_dict['leaf']
    fork_lst = tree_node_dict['fork']
    root_lst = tree_node_dict['root']
    ### Create branches (level tuple list)
    level_points = list(set().union(leaf_lst, fork_lst, root_lst))
    df_level = df_path.loc[df_path['path'].isin(level_points)]
    df_level = df_level.loc[df_level['descendant'].isin(leaf_lst)]
    df_level = df_level.reset_index(drop=True)
    # df_level = df_level.sort_values(['descendant', 'path']).reset_index(drop=True)

    branches = []
    for i in df_level.descendant.unique().tolist():
        lst = df_level.loc[df_level['descendant'] == i, 'path'].tolist()
        branches = list(set().union(branches, zip(lst, lst[1:])))

    return branches


def neuron_levelCol(df, df_anc, df_path, tree_node_dict, branch_lst, child_col, parent_col):
    ### cihld_col is from "df"
    leaf_lst = tree_node_dict['leaf']
    fork_lst = tree_node_dict['fork']
    root_lst = tree_node_dict['root']

    ### 1. Points btw root and first_fork
    first_fork = neuron_first_fork(root_lst, fork_lst, branch_lst)
    temp_lst = df_anc.loc[df_anc['descendant'] == first_fork, 'ancestor'].tolist()
    temp_lst.append(first_fork)
    zero_lst = zero_list_maker(len(temp_lst))
    df_temp_1 = pd.DataFrame({child_col: temp_lst, 'level': zero_lst})

    ### 2. Points after first_fork
    df_temp_2 = df_anc.loc[df_anc['ancestor'].isin(fork_lst)]
    df_temp_2 = pd.value_counts(df_temp_2.descendant).to_frame().reset_index()
    df_temp_2.columns = [child_col, 'level']

    ### Merge 1. & 2. to df
    df_temp_1 = pd.concat([df_temp_1, df_temp_2])
    df_temp_1 = df_temp_1.sort_values([child_col]).reset_index(drop=True)

    df = pd.merge(df, df_temp_1, how='left', on=child_col)

    max_level = max(df['level'])


    ### 3. Find deepest level of each point
    df['dp_level'] = 0
    # df_path = df_path.loc[df_path['path'] != 1]

    df_temp = df.loc[df[child_col].isin(leaf_lst), [child_col, 'level']].sort_values(['level'], ascending=False).reset_index(drop=True)
    for l in df_temp[child_col]:
        path = df_path.loc[df_path['descendant'] == l, 'path'].tolist()
        level = df_temp.loc[df_temp[child_col] == l, 'level'].values[0]
        df['dp_level'] = np.where((df[child_col].isin(path)) & (df['dp_level'] < level),
                                  level, df['dp_level'])

    return df, max_level, first_fork


def calculate_distance(positions, decimal=None, type='euclidean'):
    results = []

    # Detect dimension of tuples in the positions
    try:
        if all(len(tup) == 2 for tup in positions):
            dim = 2
        elif all(len(tup) == 3 for tup in positions):
            dim = 3
    except:
        print('Dimension of positions must be same in calculate_distance()!')


    # Calculate distance
    try:
        if all([dim == 2, type == 'haversine']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                lat1 = loc1[0]
                lng1 = loc1[1]

                lat2 = loc2[0]
                lng2 = loc2[1]

                degreesToRadians = (math.pi / 180)
                latrad1 = lat1 * degreesToRadians
                latrad2 = lat2 * degreesToRadians
                dlat = (lat2 - lat1) * degreesToRadians
                dlng = (lng2 - lng1) * degreesToRadians

                a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(latrad1) * \
                    math.cos(latrad2) * math.sin(dlng / 2) * math.sin(dlng / 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                r = 6371000

                results.append(r * c)

        elif all([dim == 2, type == 'euclidean']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                x1 = loc1[0]
                y1 = loc1[1]

                x2 = loc2[0]
                y2 = loc2[1]

                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                results.append(d)

        elif all([dim == 3, type == 'euclidean']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                x1 = loc1[0]
                y1 = loc1[1]
                z1 = loc1[2]

                x2 = loc2[0]
                y2 = loc2[1]
                z2 = loc2[2]

                d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

                results.append(d)


        if decimal is None:
            return sum(results)
        elif decimal == 0:
            return int(round(sum(results)))
        else:
            return round(sum(results), decimal)

    except:
        print('Please use available type and dim, such as "euclidean"(2-dim, 3-dim) and "haversine" (2-dim only), '
              'in calculate_distance().')


def neuron_branchCol_QCol_distance(df, df_path, tree_node_dict, branch_lst, first_fork,child_col, parent_col, type_col='T',decimal=0):
    ### Create distances of branches and create Q col
    length_lst = []    # distance of branch
    length_lst_soma = []  # distance of descendant to soma
    direct_dis_lst_soma = []  # direct distance of descendant to soma
    df_temp = pd.DataFrame()
    for i in branch_lst:
        start = i[0]
        end = i[1]
        soma = tree_node_dict['root'][0]

        path_points = df_path.loc[df_path['descendant'] == start, 'path'].tolist()

        start_idx = path_points.index(start)
        end_idx = path_points.index(end)
        soma_idx = path_points.index(soma)

        path_points_1 = path_points[start_idx: end_idx]         # exclude the end point(for Q)
        path_points_2 = path_points[start_idx: (end_idx + 1)]   # include the end point(for Q, dis)
        path_points_3 = path_points[start_idx: (soma_idx + 1)]  # path to soma


        # Create branch col and Q col
        # branch with end pt != 1 or first_fork == 1 (first_fork == soma)
        if any([end != 1, first_fork == 1]):
            temp_lst_1 = [start] * len(path_points_1)
            temp_lst_2 = list(range(len(path_points_1)))
            df_temp_1 = pd.DataFrame(OrderedDict({child_col: path_points_1, 'branch': temp_lst_1, 'Q': temp_lst_2}))
            df_temp = df_temp.append(df_temp_1)

        else:
            temp_lst_1 = [start] * len(path_points_2)
            temp_lst_2 = list(range(len(path_points_2)))
            df_temp_1 = pd.DataFrame(OrderedDict({child_col: path_points_2, 'branch': temp_lst_1, 'Q': temp_lst_2}))
            df_temp = df_temp.append(df_temp_1)

        # Calculate distance
        positions = df.loc[df[child_col].isin(path_points_2), ['x', 'y', 'z']]
        tuples = [tuple(x) for x in positions.values]
        positions_s = df.loc[df[child_col].isin(path_points_3), ['x', 'y', 'z']]
        tuples_s = [tuple(x) for x in positions_s.values]
        tuples_ds = [tuples_s[0], tuples_s[-1]]

        length = calculate_distance(tuples, decimal)
        length_lst.append(length)

        length_soma = calculate_distance(tuples_s, decimal)
        length_lst_soma.append(length_soma)

        direct_dis_soma = calculate_distance(tuples_ds, decimal)
        direct_dis_lst_soma.append(direct_dis_soma)

    # Merge branch & Q into original df
    df = pd.merge(df, df_temp, how='left', on=child_col)
    if first_fork == 1:
        df.loc[df[parent_col] == -1, ['branch', 'Q']] = [1, 0]  # add soma if first_fork = 1
    df[['branch', 'Q']] = df[['branch', 'Q']].astype('int')


    ### Create df_dis (cols ['len_des_soma', 'des_T'])
    df_dis = pd.DataFrame({'branch': branch_lst, 'len': length_lst, 'len_des_soma': length_lst_soma, 'direct_dis_des_soma': direct_dis_lst_soma})
    df_dis['ancestor'] = [tuple[1] for tuple in branch_lst]
    df_dis['descendant'] = [tuple[0] for tuple in branch_lst]
    df_dis = df_dis.sort_values(['ancestor', 'descendant']).reset_index(drop=True)

    # create type column
    df_t = df[[child_col, type_col]].copy()
    df_t.columns = ['descendant', 'des_T']
    df_dis = pd.merge(df_dis, df_t, how='left', on='descendant')


    ### Reorder columns
    df_dis = df_dis[['branch', 'descendant', 'ancestor', 'len', 'len_des_soma', 'direct_dis_des_soma', 'des_T']]

    return df, df_dis, length_lst


def list_unique(mylist):
    x = np.array(mylist)
    x = list(np.unique(x))
    x.sort()
    return x


def dict_merge_value(d, unique=True):
    lst = []
    for k, v in d.items():
        if lst is None:
            lst = v
        else:
            lst += v

    if unique:
        lst = list_unique(lst)

    return lst


def partition(lst, n=None, pct=None, shuffle_list=True):
    if shuffle_list:
        random.shuffle(lst)

    if all([n is not None, pct is None]):
        division = len(lst) / n
    elif all([n is None, pct > 0, pct < 1]):
        val = 1/pct
        n = round(val)
        division = len(lst)/n
    else:
        sys.exit("\n Use either Number(n=1, 2, 3...) or Percent(pct=0.1, 0.2, 0.3,...) to separate the list.")

    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


def swc_vibration(path_dict, num, vibrate_amplitude):
    keys = os.listdir(path_dict["import_swc"])
    if num == 0:
        print('Aug_num is zero.No Augmentation.')
    else:
        for key_t in keys:
            filenames = os.listdir(path_dict["import_swc"]+str(key_t))
            for i in range(len(filenames)):
                filenames[i] = os.path.splitext(filenames[i])[0]
            print('Vibration for '+str(key_t)+'...')
            time.sleep(0.5)
            for swc in tqdm(filenames):
                nrn = nm.io.swc.read(path_dict["import_swc"] + str(key_t) + path_dict["sep"] + swc + '.swc')
                df = pd.DataFrame(nrn.data_block, columns=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'])

                # nrn_name
                df["nrn"] = swc
                # Create child number col and dictionary of leaf/fork/root lists
                df, tree_node_dict = neuron_childNumCol(df, child_col='ID', parent_col='PARENT_ID')
                # Create ancestors and path dfs
                df_anc, df_path = neuron_ancestors_and_path(df,child_col='ID', parent_col='PARENT_ID')
                # Create branches (level tuple list)
                branch_lst = neuron_level_branch(df_path, tree_node_dict)
                # Count the level for each point
                df, max_level, first_fork = neuron_levelCol(df, df_anc, df_path, tree_node_dict, branch_lst,child_col='ID', parent_col='PARENT_ID')
                # Create distances of branches and create Q col
                df, df_dis, dis_lst = neuron_branchCol_QCol_distance(df, df_path, tree_node_dict, branch_lst, first_fork,decimal=None, child_col='ID', parent_col='PARENT_ID')
                tnd = tree_node_dict

                for fk_num in range(num) :
                    df_0 = df.copy()
                    df_dis_0 = df_dis.copy()
                    df_0["nrn"] = str(swc) +'_fk'+ str(fk_num)
                    df_dis_0["direct_dis_des_anc"] = 0

                    nodes0 = dict_merge_value(tnd)
                    nodes = partition(nodes0, 3, shuffle_list=True)

                    vibrate = ["x", "y", "z"]

                    # vibrate : min len time amplitude
                    # r0 = min(df_dis["len"])*vibrate_amplitude[key_t]
                    # vibrate : amplitude --> 20 for fc ; 2000 for em
                    r0 = vibrate_amplitude[key_t]
                    r = [r0, -r0]

                    # A.
                    # 1. Vibration
                    for i in range(3):
                        _n = nodes[i]
                        _v = random.choice(vibrate)
                        _r = random.choice(r)

                        df_0[_v] = np.where(df_0['ID'].isin(_n), df_0[_v]+_r, df_0[_v])

                    # 2. Calculate the new distance
                    for _idx in range(len(df_dis_0)):
                        # Calculate distance
                        _s = df_0.loc[df_0['ID']==tnd["root"][0], ['x', 'y', 'z']]
                        _s = [tuple(x) for x in _s.values]

                        _d0 = df_dis_0.loc[_idx, "descendant"]
                        _d = df_0.loc[df_0['ID']==_d0, ['x', 'y', 'z']]
                        _d = [tuple(x) for x in _d.values]

                        _a0 = df_dis_0.loc[_idx, "ancestor"]
                        _a = df_0.loc[df_0['ID'] == _a0, ['x', 'y', 'z']]
                        _a = [tuple(x) for x in _a.values]

                        tuples_ds = _s + _d
                        _ds = calculate_distance(tuples_ds)
                        df_dis_0["direct_dis_des_soma"] = np.where(df_dis_0["descendant"]==_d0, _ds, df_dis_0["direct_dis_des_soma"])

                        tuples_dp = _a + _d
                        _dp = calculate_distance(tuples_dp)
                        df_dis_0.loc[_idx, "direct_dis_des_anc"] = _dp

                    # 3. Find out nodes which violate the rule
                    _df_dis = df_dis_0.loc[df_dis_0["len"] < df_dis_0["direct_dis_des_anc"]].reset_index(drop=True)

                    # # B.
                    # # Adjust those nodes which violate the rule
                    # while len(_df_dis) > 0:
                    #
                    #     # 1. Recover x,y,z from vibration
                    #     for _idx in range(len(_df_dis)):
                    #         _d0 = _df_dis.loc[_idx, "descendant"]
                    #         _a0 = _df_dis.loc[_idx, "ancestor"]
                    #
                    #         for _n0 in [_d0, _a0]:
                    #             _n = df.loc[df['ID'] == _n0, ['x', 'y', 'z']].values
                    #             _row = df_0.index[df['ID'] == _n0].tolist()[0]
                    #             df_0.loc[_row, "x"] = _n[0, 0]
                    #             df_0.loc[_row, "y"] = _n[0, 1]
                    #             df_0.loc[_row, "z"] = _n[0, 2]
                    #
                    #     # 2. Calculate the new distance
                    #     for _idx in range(len(df_dis_0)):
                    #         # Calculate distance
                    #         _s = df_0.loc[df_0['ID'] == tnd["root"][0], ['x', 'y', 'z']]
                    #         _s = [tuple(x) for x in _s.values]
                    #
                    #         _d0 = df_dis_0.loc[_idx, "descendant"]
                    #         _d = df_0.loc[df_0['ID'] == _d0, ['x', 'y', 'z']]
                    #         _d = [tuple(x) for x in _d.values]
                    #
                    #         _a0 = df_dis_0.loc[_idx, "ancestor"]
                    #         _a = df_0.loc[df_0['ID'] == _a0, ['x', 'y', 'z']]
                    #         _a = [tuple(x) for x in _a.values]
                    #
                    #         tuples_ds = _s + _d
                    #         _ds = calculate_distance(tuples_ds)
                    #         df_dis_0["direct_dis_des_soma"] = np.where(df_dis_0["descendant"] == _d0, _ds,
                    #                                                    df_dis_0["direct_dis_des_soma"])
                    #
                    #         tuples_dp = _a + _d
                    #         _dp = calculate_distance(tuples_dp)
                    #         df_dis_0.loc[_idx, "direct_dis_des_anc"] = _dp
                    #
                    #     # 3. Find out nodes which violate the rule
                    #     _df_dis = df_dis_0.loc[df_dis_0["len"] < df_dis_0["direct_dis_des_anc"]].reset_index(drop=True)
                    #
                    _df_dis = _df_dis.drop(["direct_dis_des_anc"], 1)

                    # save data
                    df_0 = df_0.drop(columns=['nrn','NC','level','dp_level','branch','Q'])
                    df_0 = df_0.sort_index()
                    df_0[['ID','T','PARENT_ID']] = df_0[['ID','T','PARENT_ID']].astype(int)
                    df_0 = df_0[['ID','T','x','y','z','R','PARENT_ID']]
                    df_0.to_csv(str(path_dict["aug"])+str(key_t)+str(swc)+'_fk'+str(fk_num)+".swc",index=False,sep=' ')

    time.sleep(0.5)
    return None


def operation_decide(key):
    if "ex" in key:
        if "R" in key:
            indexR_lst = [[], [0, 1], [0, 2], [1, 2]]
            reflection_key = int(key[1])-1
            rotation_key = int(key[-1])
            if rotation_key == 0:
                return lambda x: reflection_core(rotation_core(x, [0, 1]), indexR_lst[reflection_key])
            elif rotation_key == 1:
                return lambda x: reflection_core(rotation_core(x, [1, 2]), indexR_lst[reflection_key])
        if "L" in key:
            indexL_lst = [[0], [1], [2], [0, 1, 2]]
            reflection_key = int(key[1])-1
            rotation_key = int(key[-1])
            if rotation_key == 0:
                return lambda x: reflection_core(rotation_core(x, [0, 1]), indexL_lst[reflection_key])
            elif rotation_key == 1:
                return lambda x: reflection_core(rotation_core(x, [1, 2]), indexL_lst[reflection_key])

    else:
        if "R" in key:
            indexR_lst = [[], [0, 1], [0, 2], [1, 2]]
            reflection_key = int(key.replace("R", ""))-1
            return lambda x: reflection_core(x, indexR_lst[reflection_key])
        if "L" in key:
            indexL_lst = [[0], [1], [2], [0, 1, 2]]
            reflection_key = int(key.replace("L", ""))-1
            return lambda x: reflection_core(x, indexL_lst[reflection_key])


def res_pkl2grp(path_pkl, num_best, name_group=""):
    # load the comparison result of the target neuron
    with open(path_pkl, "rb") as file:
        res = pickle.load(file)["sn"]

    # the name of .yml group
    if name_group == "":
        name_group = res[0][0]

    # build the text
    text = ""
    text += name_group + ":\n"
#    for i in range(num_best):


def ignore_soma_branch(lst):
    index = 0

    for i in range(lst.shape[0]):
        if lst[i][4] >= 2:
            index = i
            break

    lst = lst[index:lst.shape[0]]
    return lst


def map_max_score(map1, map2, key):
    if key == "unit":
        max_score = np.max((np.count_nonzero(map1), np.count_nonzero(map2)))
        return max_score

    if key == "sn":
        max_score = np.max((np.sum(np.power(map1, 2)),
                            np.sum(np.power(map2, 2))))
        return max_score

    if key == "rsn":
        max_score = np.max((np.sum(np.power(map1, 2)),
                            np.sum(np.power(map2, 2))))
        return max_score


# ----------------------------------------------------------------

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


def load_pkl(path):
    if path[-4:] != '.pkl':
        # print('Check the file type')
        path += '.pkl'
    with open(path,'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data


def traversal_folder(path, with_extension=False):
    file_lst = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name[0] != '.':             # without '.gitkeep', '.DS_Store' or anything else
                if with_extension:
                    file_lst.append(file_name)
                else:
                    file_lst.append(file_name[:-4]) # without '.swc'
    return file_lst


def get_key(dictionary, value):     # use value to get key from dictionary
    return [k for k, v in dictionary.items() if value in v]