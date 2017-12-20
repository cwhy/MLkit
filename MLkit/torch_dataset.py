import torch as th
from MLkit.dataset import DataSet
import numpy as np


class TorchDataSet:
    def __init__(self, x: np.ndarray, y: np.ndarray, cuda: bool = True):
        if cuda:
            tensor = th.cuda.FloatTensor
        else:
            tensor = th.FloatTensor
        self.x = tensor(x)
        self.y = tensor(y)
        self.n_samples = x.shape[0]

    @classmethod
    def from_DataSet(cls, dataset: DataSet, cuda: bool = True):
        return cls(x=dataset.x, y=dataset.y, cuda=cuda)
