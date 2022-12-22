from util import *
import ranking_method as rk


class NrnRanking:
    """
    provide the individual and the batch processing versions of ranking system between neurons
    and it can output figures of 2D-mapping with ranking result
    """
    def __init__(self, path_dict, grid_num, weighting_keys,
                 ranking_method=rk.sig_ave_mask, coordinate_selection="fixed_moi"):
        # system configuration
        self.path = path_dict

        # dynamical variables
        self.name = ""
        self.file_list = []
        self.file_list2 = []
        self.corresponding_neuron = []
        self.corresponding_map = {}

        # user define
        self.grid_num = grid_num
        self.weight_keys = weighting_keys
        self.ranking_method = ranking_method
        self.axis_selection = 0
        self.batch_num = 5
        self.coord_sel = coordinate_selection

        # preprocessing-required information
        self.info_lst = []
        self.I_dict = {}
        self.cm_dict = {}
        self.rmax_i = {}
        self.rmax_s = {}
        self.coord = {}
        self.total_nrnID = []

        self.__load_para()

    def __load_para(self):
        with open(self.path["stats"] + "nI_dict.pkl", "rb") as file:
            self.I_dict = pickle.load(file)
        with open(self.path["stats"] + "cm_dict.pkl", "rb") as file:
            self.cm_dict = pickle.load(file)
        with open(self.path["stats"] + "rmax_individual.pkl", "rb") as file:
            self.rmax_i = pickle.load(file)
        with open(self.path["stats"] + "rmax_standard.pkl", "rb") as file:
            self.rmax_s = pickle.load(file)
        with open(self.path["stats"] + "coord_dict.pkl", "rb") as file:
            self.coord = pickle.load(file)
        self.total_nrnID = list(self.I_dict[self.weight_keys[0]].keys())

    def generate_dataset(self, key_w):
        if self.axis_selection == 0:
            # todo: change axes
            num = len(self.file_list) // self.batch_num + int(len(self.file_list) % self.batch_num != 0)
            candidate_prin_vec = self.coord[key_w][self.file_list2[0]][:, 0][np.newaxis, :]
            candidate_vec = self.I_dict[key_w][self.file_list2[0]][np.newaxis, :]
            candidate_cm = self.cm_dict[key_w][self.file_list2[0]][np.newaxis, :]
            for k in range(1, len(self.file_list2)):
                candidate_prin_vec = np.vstack((candidate_prin_vec, self.coord[key_w][self.file_list2[k]][:, 0]))
                candidate_vec = np.vstack((candidate_vec, self.I_dict[key_w][self.file_list2[k]]))
                candidate_cm = np.vstack((candidate_cm, self.cm_dict[key_w][self.file_list2[k]]))
            candidate_prin_vec = candidate_prin_vec[np.newaxis]
            candidate_vec = candidate_vec[np.newaxis]
            candidate_cm = candidate_cm[np.newaxis]
            _candidate_prin_vec = np.tile(candidate_prin_vec, (self.batch_num, 1, 1))
            _candidate_vec = np.tile(candidate_vec, (self.batch_num, 1, 1))
            _candidate_cm = np.tile(candidate_cm, (self.batch_num, 1, 1))

            for i in range(num):
                if (i != num-1) or (int(len(self.file_list) % self.batch_num) == 0):
                    try:
                        target_prin_vec = self.coord[key_w][self.file_list[i*self.batch_num]][:, 0]
                        target_vec = self.I_dict[key_w][self.file_list[i*self.batch_num]]
                        target_cm = self.cm_dict[key_w][self.file_list[i*self.batch_num]]
                    except:
                        print("the given name does not exist in the I list : ", self.file_list[i*self.batch_num])
                        return None
                    target_prin_vec = np.tile(target_prin_vec, (len(self.file_list2), 1))[np.newaxis]
                    target_vec = np.tile(target_vec, (len(self.file_list2), 1))[np.newaxis]
                    target_cm = np.tile(target_cm, (len(self.file_list2), 1))[np.newaxis]
                    for j in range(1, self.batch_num):
                        _target_prin_vec = np.tile(self.coord[key_w][self.file_list[i*self.batch_num+j]][:, 0],
                                                   (len(self.file_list2), 1))[np.newaxis]
                        _target_vec = np.tile(self.I_dict[key_w][self.file_list[i*self.batch_num+j]],
                                              (len(self.file_list2), 1))[np.newaxis]
                        _target_cm = np.tile(self.cm_dict[key_w][self.file_list[i*self.batch_num+j]],
                                             (len(self.file_list2), 1))[np.newaxis]
                        target_prin_vec = np.concatenate((target_prin_vec, _target_prin_vec), 0)
                        target_vec = np.concatenate((target_vec, _target_vec), 0)
                        target_cm = np.concatenate((target_cm, _target_cm), 0)
                    del _target_prin_vec, _target_vec, _target_cm

                else:
                    num2 = int(len(self.file_list) % self.batch_num)
                    _candidate_prin_vec = np.tile(candidate_prin_vec, (num2, 1, 1))
                    _candidate_vec = np.tile(candidate_vec, (num2, 1, 1))
                    _candidate_cm = np.tile(candidate_cm, (num2, 1, 1))
                    try:
                        target_prin_vec = self.coord[key_w][self.file_list[i*self.batch_num]][:, 0]
                        target_vec = self.I_dict[key_w][self.file_list[i*self.batch_num]]
                        target_cm = self.cm_dict[key_w][self.file_list[i*self.batch_num]]
                    except:
                        print("the given name does not exist in the I list : ", self.file_list[i*self.batch_num])
                        return None

                    target_prin_vec = np.tile(target_prin_vec, (len(self.file_list2), 1))[np.newaxis]
                    target_vec = np.tile(target_vec, (len(self.file_list2), 1))[np.newaxis]
                    target_cm = np.tile(target_cm, (len(self.file_list2), 1))[np.newaxis]

                    for j in range(1, num2):
                        _target_prin_vec = np.tile(self.coord[key_w][self.file_list[i*self.batch_num+j]][:, 0],
                                                   (len(self.file_list2), 1))[np.newaxis]
                        _target_vec = np.tile(self.I_dict[key_w][self.file_list[i*self.batch_num+j]],
                                              (len(self.file_list2), 1))[np.newaxis]
                        _target_cm = np.tile(self.cm_dict[key_w][self.file_list[i*self.batch_num+j]],
                                             (len(self.file_list2), 1))[np.newaxis]
                        target_prin_vec = np.concatenate((target_prin_vec, _target_prin_vec), 0)
                        target_vec = np.concatenate((target_vec, _target_vec), 0)
                        target_cm = np.concatenate((target_cm, _target_cm), 0)
                    try:
                        del _target_prin_vec, _target_vec, _target_cm
                    except:
                        pass
                #print(target_prin_vec.shape, _candidate_prin_vec.shape)
                yield target_prin_vec, target_vec, target_cm, _candidate_prin_vec, _candidate_vec, _candidate_cm

        elif self.axis_selection == 1:
            num = len(self.file_list2) // self.batch_num + int(len(self.file_list2) % self.batch_num != 0)
            for i in range(len(self.file_list)):
                try:
                    target_prin_vec = self.coord[key_w][self.file_list[i]][:, 0]
                    target_vec = self.I_dict[key_w][self.file_list[i]]
                    target_cm = self.cm_dict[key_w][self.file_list[i]]
                except:
                    print("the given name does not exist in the I list : ", self.file_list[i])
                    return None

                _target_prin_vec = np.tile(target_prin_vec, (self.batch_num, 1))
                _target_vec = np.tile(target_vec, (self.batch_num, 1))
                _target_cm = np.tile(target_cm, (self.batch_num, 1))
                for j in range(num):
                    if j != num-1:
                        candidate_prin_vec = self.coord[key_w][self.file_list2[j*self.batch_num]][:, 0][np.newaxis, :]
                        candidate_vec = self.I_dict[key_w][self.file_list2[j*self.batch_num]][np.newaxis, :]
                        candidate_cm = self.cm_dict[key_w][self.file_list2[j*self.batch_num]][np.newaxis, :]
                        for k in range(1, self.batch_num):
                            candidate_prin_vec = np.vstack((candidate_prin_vec, self.coord[key_w][self.file_list2[j*self.batch_num+k]][:, 0]))
                            candidate_vec = np.vstack((candidate_vec, self.I_dict[key_w][self.file_list2[j*self.batch_num+k]]))
                            candidate_cm = np.vstack((candidate_cm, self.cm_dict[key_w][self.file_list2[j*self.batch_num+k]]))
                    else:
                        _target_prin_vec = np.tile(target_prin_vec, (int(len(self.file_list2) % self.batch_num), 1))
                        _target_vec = np.tile(target_vec, (int(len(self.file_list2) % self.batch_num), 1))
                        _target_cm = np.tile(target_cm, (int(len(self.file_list2) % self.batch_num), 1))
                        candidate_prin_vec = self.coord[key_w][self.file_list2[j*self.batch_num]][0, :][np.newaxis, :]
                        candidate_vec = self.I_dict[key_w][self.file_list2[j*self.batch_num]][np.newaxis, :]
                        candidate_cm = self.cm_dict[key_w][self.file_list2[j*self.batch_num]][np.newaxis, :]
                        for k in range(1, int(len(self.file_list2) % self.batch_num)):
                            candidate_prin_vec = np.vstack((candidate_prin_vec, self.coord[key_w][self.file_list2[j*self.batch_num+k]][:, 0]))
                            candidate_vec = np.vstack((candidate_vec, self.I_dict[key_w][self.file_list2[j*self.batch_num+k]]))
                            candidate_cm = np.vstack((candidate_cm, self.cm_dict[key_w][self.file_list2[j*self.batch_num+k]]))
                    yield _target_prin_vec, _target_vec, _target_cm, candidate_prin_vec, candidate_vec, candidate_cm

    def match(self, data, threshold_I, threshold_dis, threshold_in, gate="and"):
        v_pv, v_I, v_cm, c_pv, c_I, c_cm = data

        if self.axis_selection == 0:
            value_in = np.abs(np.sum(v_pv*c_pv, 2))
            value_I = np.sqrt(np.sum(np.power((v_I-c_I), 2), 2))
            value_cm = np.sqrt(np.sum(np.power((v_cm-c_cm), 2), 2))

            if gate == "and":
                bool_lst = (value_in > threshold_in) & (value_I < threshold_I) & (value_cm < threshold_dis)
                index = np.nonzero(bool_lst)
            elif gate == "or":
                bool_lst = (value_in > threshold_in) | (value_I < threshold_I) | (value_cm < threshold_dis)
                index = np.nonzero(bool_lst)

            return index

        elif self.axis_selection == 1:
            value_in = np.abs(np.sum(v_pv*c_pv, 1))
            value_I = np.sqrt(np.sum(np.power((v_I-c_I), 2), 1))
            value_cm = np.sqrt(np.sum(np.power((v_cm-c_cm), 2), 1))

            if gate == "and":
                bool_lst = (value_in > threshold_in) & (value_I < threshold_I) & (value_cm < threshold_dis)
                index = np.nonzero(bool_lst)
            elif gate == "or":
                bool_lst = (value_in > threshold_in) | (value_I < threshold_I) | (value_cm < threshold_dis)
                index = np.nonzero(bool_lst)

            return index

    def orientation_generator(self, key_w, corre_name, threshold, mirror=False):
        # enumerate the possibilities of axis exchanging
        option_rotation = []
        if self.I_dict[key_w][corre_name][1] - self.I_dict[key_w][corre_name][0] < threshold:
            option_rotation.append([0, 1])
        if self.I_dict[key_w][corre_name][2] - self.I_dict[key_w][corre_name][1] < threshold:
            option_rotation.append([1, 2])

        # enumerate possible combination of coordinates and create its operator
        i = 0
        indexR_lst = [[], [0, 1], [0, 2], [1, 2]]
        indexL_lst = [[0], [1], [2], [0, 1, 2]]
        while i < 4:
            yield "R" + str(i+1), lambda x: reflection_core(x, indexR_lst[i])
            if mirror:
                yield "L" + str(i+1), lambda x: reflection_core(x, indexL_lst[i])
            i += 1
        # extra combinations when eigenvalues of moment of inertia are close
        for option in option_rotation:
            i = 0
            indexR_lst = [[], [0, 1], [0, 2], [1, 2]]
            indexL_lst = [[0], [1], [2], [0, 1, 2]]
            while i < 4:
                yield "R" + str(i+1) + "_ex" + str(option[1]), lambda x: reflection_core(rotation_core(x, option), indexR_lst[i])
                if mirror:
                    yield "L" + str(i+1) + "_ex" + str(option[1]), lambda x: reflection_core(rotation_core(x, option), indexL_lst[i])
                i += 1

    def morphological_ranking(self, key_w, threshold, save_path, plot=True):
        # load the target map
        with open(self.path["map"] + self.name + ".pkl", "rb") as file:
            target_map = pickle.load(file)
            target_map = target_map[key_w]

        # search the best combination
        info_list = []
        time.sleep(0.5)
        for i in range(len(self.corresponding_neuron)):
            score_max = 0
            right_key = ""
            right_map = np.array([])
            with open(self.path["map"] + self.corresponding_neuron[i] + ".pkl", "rb") as file:
                _maps = pickle.load(file)
                _maps = _maps[key_w]

            # ranking process
            r_max = np.max((self.rmax_i[key_w][self.corresponding_neuron[i]], self.rmax_i[key_w][self.name]))
            _map1 = mapping_core(target_map, self.grid_num, r_max)
            _map2 = mapping_core(_maps, self.grid_num, r_max)
            ts = time.time()
            for key, trans in self.orientation_generator(key_w, self.corresponding_neuron[i], threshold):
                _map2t = trans(_map2)
                _score, _shift = self.ranking_method(_map1, _map2t)
                if plot == "complete":
                    figure_output([self.name, self.corresponding_neuron[i]],
                                  [_map1, _map2t],
                                  [key, _score], _map1[0].shape[0], save_path, key_w,
                                  sub_name=["X-Y", "Y-Z", "Z-X"], I_dict=self.I_dict)

                if _score > score_max:
                        right_key = key
                        score_max = _score
                        right_map = _map2t
                        right_shift = _shift
            info_list.append([self.name, self.corresponding_neuron[i], right_key, score_max, right_shift, r_max])
            print("It costs %.2f second for comparing one pair of neurons." % (time.time()-ts))

            if plot == "best":
                self.figure_output([self.name, self.corresponding_neuron[i]],
                                   [_map1, right_map],
                                   [right_key, score_max, right_shift], _map1[0].shape[0], save_path, key_w)

        info_list.sort(key=lambda x: (x[0], -x[3]))
        time.sleep(0.5)

        return info_list

    def figure_output(self, nrn_ID, maps, res, N, save_path, key_g, method="normal"):
        # order of axes
        sub_name = ["Y-Z", "Z-X", "X-Y"]

        x = np.arange(0, N, 1)
        y = np.arange(0, N, 1)
        X, Y = np.meshgrid(x, y)
        fig = figure.Figure()
        ax = fig.subplots(2, 3)

        if method == "normal":
            for i in range(2):
                for j in range(3):
                    ax[i][j].pcolormesh(X, Y, maps[i][j], vmin=np.min(maps[i][j]), vmax=np.max(maps[i][j]), shading='auto')
                    ax[i][j].set_title(r"$I_%d: %.3f$" % (j+1, self.I_dict[key_g][nrn_ID[i]][j]))
                    if i == 1:
                        ax[i][j].set_xlabel(sub_name[j])
                    if j == 0:
                        ax[i][j].set_ylabel(nrn_ID[i])

        if method == "shift":
            for i in range(2):
                if i == 0:
                    for j in range(3):
                        ax[i][j].pcolormesh(X, Y, maps[i][j], vmin=np.min(maps[i][j]), vmax=np.max(maps[i][j]), shading='auto')
                        ax[i][j].set_title(r"$I_%d: %.3f$" % (j+1, self.I_dict[key_g][nrn_ID[i]][j]))
                        if i == 1:
                            ax[i][j].set_xlabel(sub_name[j])
                        if j == 0:
                            ax[i][j].set_ylabel(nrn_ID[i])
                elif i == 1:
                    ax[1][0].pcolormesh(X, Y, rk.map_shift(maps[1][0], res[2][1], res[2][2]), vmin=np.min(maps[1][0]), vmax=np.max(maps[1][0]), shading='auto')
                    ax[1][0].set_title(r"$I_%d: %.3f$" % (0+1, self.I_dict[key_g][nrn_ID[1]][0]))
                    ax[1][0].set_xlabel(sub_name[0])
                    ax[1][0].set_ylabel(nrn_ID[1])

                    ax[1][1].pcolormesh(X, Y, rk.map_shift(maps[1][1], res[2][2], res[2][0]), vmin=np.min(maps[1][1]), vmax=np.max(maps[1][1]), shading='auto')
                    ax[1][1].set_title(r"$I_%d: %.3f$" % (1+1, self.I_dict[key_g][nrn_ID[1]][1]))
                    ax[1][1].set_xlabel(sub_name[1])

                    ax[1][2].pcolormesh(X, Y, rk.map_shift(maps[1][2], res[2][0], res[2][1]), vmin=np.min(maps[1][2]), vmax=np.max(maps[1][2]), shading='auto')
                    ax[1][2].set_title(r"$I_%d: %.3f$" % (2+1, self.I_dict[key_g][nrn_ID[1]][2]))
                    ax[1][2].set_xlabel(sub_name[2])

        fig.suptitle("score: %.3f, weighting: %s" % (res[1], res[0]), y=0.96, fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(save_path + nrn_ID[0] + "_" + nrn_ID[1] + "_" + key_g + "_" + method + ".png", dpi=200)
        fig.clf()
        plt.close()

        return None

    def batch_matching_process(self, overwrite, target_list=["FC"], candidate_list=["EM"],
                               threshold_I=0.4, threshold_dis=100.0, threshold_in=np.cos(np.pi*50/180)):
        # process begins
        time.sleep(0.5)
        print("Matching Processor : ")
        time.sleep(0.5)

        # define matching groups
        with open(self.path["stats"] + "group_dict.pkl", "rb") as file:
            group_dict = pickle.load(file)
        for key in target_list:
            self.file_list += group_dict[key]
        for key in candidate_list:
            self.file_list2 += group_dict[key]

        if overwrite:
            matching_dict = {}

            for key in self.weight_keys:
                matching_dict[key] = {}
                for key_id in self.file_list:
                    matching_dict[key][key_id] = []

                num_rep = 0
                for data in self.generate_dataset(key):
                    res = self.match(data, threshold_I, threshold_dis, threshold_in)
                    for i in range(len(res[0])):
                        matching_dict[key][self.file_list[res[0][i]+num_rep*self.batch_num]].append(self.file_list2[res[1][i]])
                    num_rep += 1

                print("  Done!")

            with open(self.path["stats"] + "match_dict.pkl", "wb") as file:
                pickle.dump(matching_dict, file)
        else:
            print("  Done!")

    def morphological_ranking_fix(self, key_g, c_lst, saving, plot, shift_delta=30):
        # target nrn information
        coor_target = self.coord[key_g][self.name]
        cm_target = self.cm_dict[key_g][self.name]
        r_max_target = self.rmax_i[key_g][self.name]
        with open(self.path["map"] + self.name + ".pkl", "rb") as file:
            map_base_target = pickle.load(file)[key_g]

        # candidates
        info_list = []
        mapping_list = []
        candidates = c_lst
        for name_c in candidates:
            with open(self.path["convert"] + name_c + ".pkl", "rb") as file:
                nrn_c = pickle.load(file).loc[:, ["x", "y", "z", "base"]].to_numpy()
            r_max_c, map_base_c = mapping_rule(nrn_c, coor_target, cm_target, key_g)
            r_max = max(r_max_target, r_max_c)

            map_target = mapping_core(map_base_target, self.grid_num, r_max)
            map_c = mapping_core(map_base_c, self.grid_num, r_max)

            max_score = map_max_score(map_target, map_c, key_g)
            shift_grid = np.ceil(shift_delta/(r_max/self.grid_num)).astype("int")
            _score, _shift, _score_x, _score_y, _score_z = self.ranking_method(map_target, map_c, max_score, shift_grid)

            if plot:
                self.figure_output([self.name, name_c],
                                   [map_target, map_c],
                                   [key_g, _score, _shift], map_target[0].shape[0], self.path["plot_pair_neuron"], key_g)

            if saving:
                mapping_list.append([self.name, name_c, _score, map_target, map_c])
            info_list.append([self.name, name_c, _score, _shift, _score_x, _score_y, _score_z])

        if saving:
            return info_list, mapping_list
        else:
            return info_list

    def morphological_ranking3(self, key_w, c_lst, shift_delta=30):
        # target nrn information
        coor_target = self.coord[key_w][self.name]
        cm_target = self.cm_dict[key_w][self.name]
        r_max_target = self.rmax_s[key_w][self.name]
        with open(self.path["map2"] + self.name + ".pkl", "rb") as file:
            map_target = pickle.load(file)[key_w]

        # candidates
        info_list = []
        candidates = c_lst
        time.sleep(0.5)
        map_candidates = []
        for name_c in tqdm(candidates):
            with open(self.path["map2"] + name_c + ".pkl", "rb") as file:
                map_c = pickle.load(file)["unit"]
            #map_candidates.append(map_c)

            max_score = map_max_score(map_target, map_c, key_w)
            shift_grid = 5
            _score, _shift, _score_x, _score_y, _score_z = self.ranking_method(map_target, map_c, max_score, shift_grid)

            info_list.append([self.name, name_c, _score, _shift, _score_x, _score_y, _score_z])
        return info_list

    def batch_ranking_process(self, overwrite, map_data_saving, plot):
        with open(self.path["stats"] + "match_dict.pkl", "rb") as file:
            match_dict = pickle.load(file)

        keys = match_dict.keys()

        if self.coord_sel == "target-orientation":
            if overwrite:
                for key_g in keys:
                    information_list = []
                    mapping_list = []
                    print(key_g, " :")

                    match_lst = list(match_dict[key_g].keys())
                    for i in range(len(match_lst)):
                        print("%d / %d" % (i+1,len(match_lst)))
                        t_s = time.time()
                        self.name = match_lst[i]

                        if map_data_saving:
                            _information_list, _mapping_data = self.morphological_ranking_fix(key_g, match_dict[key_g][self.name], map_data_saving, plot)
                            information_list += _information_list
                            mapping_list += _mapping_data
                        else:
                            information_list += self.morphological_ranking_fix(key_g, match_dict[key_g][self.name], map_data_saving, plot)

                        print("Time Cost: ", "%.3f" % (time.time() - t_s), " sec")

                    if map_data_saving:
                        with open(self.path["stats"] + "mapping_data_" + key_g + ".pkl", "wb") as file:
                            pickle.dump(mapping_list, file)

                    with open(self.path["stats"] + "info_list_" + key_g + ".csv", "w", newline="") as file:
                        writer = csv.writer(file)

                        # title
                        writer.writerow(["target_ID", "candidate_ID", "score", "shift", "score_x", "score_y", "score_z"])

                        # values
                        for row in information_list:
                            writer.writerow(row)

        if self.coord_sel == "coordinate-orientation":
            if overwrite:
                information_dict = {}

                for key in keys:
                    information_dict[key] = []
                    #print(key, " :")

                    match_lst = list(match_dict[key].keys())
                    for i in range(len(match_lst)):
                        self.name = match_lst[i]
                        print("target: ", self.name, "  searching")
                        information_dict[key] += self.morphological_ranking3(key, match_dict[key][self.name])

                    lst = information_dict[key]
                    lst.sort(key=lambda x: -x[2])
                    length = min(5, len(lst))
                    for k in range(length):
                        print("rank ", k+1, ": ", lst[k][1], "\n score: ", lst[k][2])