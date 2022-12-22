from util import *


class NrnMapping:
    """
    provide the individual and the batch processing versions of 2D-neuron mapping
    """
    def __init__(self, path_dict, file_lst, weighting_keys, grid_num):
        # system configuration
        self.path = path_dict

        if not os.path.isdir(path_dict["map"]):
            os.makedirs(path_dict["map"])
        if not os.path.isdir(path_dict["stats"]):
            os.makedirs(path_dict["stats"])

        # dynamical variables
        self.nrn = []
        self.name = ""

        # user define
        self.file_list = file_lst
        self.max_sn = 0
        self.num_grid = grid_num
        self.weight_keys = weighting_keys

        # preprocessing-required information
        self.data = {"I_dict": {},
                     "nI_dict": {},
                     "coord_dict": {},
                     "cm_dict": {}}
        for key_g in weighting_keys:
            for key_d in self.data.keys():
                self.data[key_d][key_g] = {}

    def load_data(self, ignore_soma=False, reduced_sn=False, num_re=3):
        # data standardization
        with open(self.path["convert"] + self.name + ".pkl", "rb") as file:
            _df = pickle.load(file)
        self.nrn = _df.loc[:, ["x", "y", "z", "sn", "CN"]].to_numpy()

        # ignore soma branch
        if ignore_soma:
            self.nrn = ignore_soma_branch(self.nrn)

        # sn reduced process
        if reduced_sn:
            self.nrn = reduce_mapping(self.nrn, num_re)

    def diag_process(self, norm=True, normI=False):
        # set the upper limit of sn num
        self.nrn = baseline_rule(self.nrn, self.max_sn, norm)
        with open(self.path["convert"] + self.name + ".pkl", "wb") as file:
            pickle.dump(pd.DataFrame(self.nrn, columns=["x", "y", "z", "sn", "CN", "base"]), file)

        # define the coordinate of the neuron
        coord_dict = coordinate_rule(self.nrn, normI, self.weight_keys)

        return coord_dict

    def map_process(self, cor, cm, key):
        # set the upper limit of sn num
        with open(self.path["convert"] + self.name + ".pkl", "rb") as file:
            self.nrn = pickle.load(file).loc[:, ["x", "y", "z", "base"]].to_numpy()

        r_max, maps = mapping_rule(self.nrn, cor, cm, key)

        return r_max, maps

    def map_process_s(self, cm, key):
        # set the upper limit of sn num
        with open(self.path["convert"] + self.name + ".pkl", "rb") as file:
            self.nrn = pickle.load(file).loc[:, ["x", "y", "z", "base"]].to_numpy()

        r_max, maps = mapping_rule_standard(self.nrn, cm, key, self.num_grid)

        return r_max, maps

    def batch_coordinate_process(self, overwrite, max_sn,
                                 normalization_moi=True, normalization_sn=True, ignore_soma=False, reduced_sn=False, num_re=3):
        # todo: concurrency-multithreading
        # load
        self.max_sn = max_sn

        # automatically process all the given data
        time.sleep(0.5)
        print("Coordinate Processor : ")
        time.sleep(0.5)

        if overwrite:
            buffer = []
        else:
            # load existing data
            _lists = []
            for key_d in self.data.keys():
                if os.path.isfile(self.path["stats"] + key_d + ".pkl"):
                    with open(self.path["stats"] + key_d + ".pkl", "rb") as file:
                        _data_dict = pickle.load(file)
                    for key_g in self.weight_keys:
                        if key_g in _data_dict.keys():
                            self.data[key_d][key_g] = _data_dict[key_g]
                            _lists.append(list(self.data[key_d][key_g].keys()))
                        else:
                            _lists.append([])
                    del _data_dict

            # co-existing files
            if len(_lists) == 4*len(self.weight_keys):
                try:
                    buffer = list(set(_lists.pop()).intersection(*map(set, _lists)))
                except IndexError:
                    return list()
            else:
                buffer = []
            del _lists

        _file_lst = list(set(self.file_list)-set(buffer))

        if len(_file_lst) == 0:
            print("  Done!")
            return None

        for i in tqdm(_file_lst):
            self.name = i
            self.load_data(ignore_soma, reduced_sn, num_re)
            _dict = self.diag_process(norm=normalization_sn, normI=normalization_moi)

            for key_g in self.weight_keys:
                self.data["nI_dict"][key_g][self.name] = _dict[key_g][0]
                self.data["I_dict"][key_g][self.name] = _dict[key_g][1]
                self.data["coord_dict"][key_g][self.name] = _dict[key_g][2]
                self.data["cm_dict"][key_g][self.name] = _dict[key_g][3]

        # save data
        for key_d in self.data.keys():
            with open(self.path["stats"] + key_d + ".pkl", "wb") as file:
                pickle.dump(self.data[key_d], file)

        return None

    def batch_mapping_process(self, overwrite,
                              normalization_sn=True, ignore_soma=False, reduced_sn=False, num_re=3):
        # todo: process standard coordinate
        # load data
        for key_d in self.data.keys():
            with open(self.path["stats"] + key_d + ".pkl", "rb") as file:
                self.data[key_d] = pickle.load(file)

        time.sleep(0.5)
        print("Mapping Processor : ")
        time.sleep(0.5)

        if overwrite:
            buffer = []
            _rmax_i = {i: {} for i in self.weight_keys}
            _rmax_s = {i: {} for i in self.weight_keys}

        else:
            _buffer = []
            if os.path.isfile(self.path["stats"]+"rmax_individual"+".pkl"):
                with open(self.path["stats"]+"rmax_individual"+".pkl", "rb") as file:
                    _rmax_i = pickle.load(file)

                for key_g in self.weight_keys:
                    if key_g in _rmax_i:
                        _buffer.append(_rmax_i[key_g].keys())
                    else:
                        _rmax_i[key_g] = {}
                        _buffer.append([])
                _buffer.append(_rmax_i[self.weight_keys[0]].keys())

            else:
                _rmax_i = {i: {} for i in self.weight_keys}
                _buffer.append([])

            if os.path.isfile(self.path["stats"]+"rmax_standard"+".pkl"):
                with open(self.path["stats"]+"rmax_standard"+".pkl", "rb") as file:
                    _rmax_s = pickle.load(file)

                for key_g in self.weight_keys:
                    if key_g in _rmax_s:
                        _buffer.append(_rmax_s[key_g].keys())

                    else:
                        _rmax_s[key_g] = {}
                        _buffer.append([])
                _buffer.append(_rmax_s[self.weight_keys[0]].keys())

            else:
                _rmax_s = {i: {} for i in self.weight_keys}
                _buffer.append([])

            buffer = list(set(_buffer.pop()).intersection(*map(set, _buffer)))
            del _buffer

        _file_lst = list(set(self.file_list)-set(buffer))

        if len(_file_lst) == 0:
            print("  Done!")
            return None

        for i in tqdm(_file_lst):
            self.name = i
            self.load_data(ignore_soma, reduced_sn, num_re)
            _map = {}
            for key_g in self.weight_keys:
                _rmax_i[key_g][i], _map[key_g] = self.map_process(self.data["coord_dict"][key_g][i],
                                                                  self.data["cm_dict"][key_g][i],
                                                                  key_g)

            with open(self.path["map"] + self.name + ".pkl", "wb") as file:
                pickle.dump(_map, file)

            _map = {}
            for key in self.weight_keys:
                _rmax_s[key][i], _map[key] = self.map_process_s(self.data["cm_dict"][key][i], key)

            with open(self.path["map2"] + self.name + ".pkl", "wb") as file:
                pickle.dump(_map, file)

#            if plot:
#                self.figure_output(_map, self.path["plot_single_neuron"])
        with open(self.path["stats"] + "rmax_individual" + ".pkl", "wb") as file:
            pickle.dump(_rmax_i, file)
        with open(self.path["stats"] + "rmax_standard" + ".pkl", "wb") as file:
            pickle.dump(_rmax_s, file)

        return None
