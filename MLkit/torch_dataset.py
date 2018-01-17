from typing import Union, Callable, Optional

import torch as th
from MLkit.dataset import DataSet
import numpy as np

Tensor = Union[th.cuda.FloatTensor, th.FloatTensor]


class TorchDataSet:
    def __init__(self, x: np.ndarray,
                 y: np.ndarray,
                 cuda: bool = True,
                 name: Optional[str] = None):
        if name is not None:
            self.name = name
        else:
            self.name = "Unnamed"
        if cuda:
            tensor = th.cuda.FloatTensor
        else:
            tensor = th.FloatTensor
        self.x: Tensor = tensor(x)
        self.y: Tensor = tensor(y)
        self.n_samples = x.shape[0]

    @classmethod
    def from_DataSet(cls, dataset: DataSet, cuda: bool = True):
        return cls(x=dataset.x, y=dataset.y, cuda=cuda, name=dataset.name)

    def to_DataSet(self):
        return DataSet(x=self.x.numpy(), y=self.y.numpy(), name=self.name)
