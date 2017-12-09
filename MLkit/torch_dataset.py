import torch as th
from MLkit.dataset import DataSet


class TorchDataSet:
    def __init__(self, dataset: DataSet, cuda: bool = True):
        if cuda:
            tensor = th.cuda.FloatTensor
        else:
            tensor = th.FloatTensor
        self.x = tensor(dataset.x)
        self.y = tensor(dataset.y)

