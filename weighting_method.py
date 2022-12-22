import numpy as np

class Weight:
    key = "unit"

    def __init__(self, attribute={"sn": 2, "rsn": 1}):
        self.func = lambda x: x
        self.attri = attribute

    def __init__func(self, attribute={}):
        self.attri = attribute
        if self.key == "unit":
            self.func = lambda x:np.ones(x.shape[0], dtype="int")

        elif self.key == "sn":
            self.func = lambda x:np.power(x, self.attri[self.key])

        elif self.key == "rsn":
            self.func = lambda x:np.power(x, self.attri[self.key])