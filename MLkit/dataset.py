from typing import List, Union, Optional, Tuple
import numpy as np
from MLkit.color import get_color_gen

Dimensions = Union[List[int], None, np.ndarray]
DataSetType = Union['DataSet', 'CategoricalDataSet']


class DataSet:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 dim_X: Dimensions = None,
                 dim_Y: Dimensions = None,
                 name: Optional[str] = None,
                 shuffle: bool = True):

        if name is not None:
            self.name = name
        else:
            self.name = "Unnamed"

        self.n_samples = x.shape[0]
        if shuffle:
            _idx = np.random.permutation(self.n_samples)
            self.x = x[_idx, :]
            self.y = y[_idx, :]
        else:
            self.x = x
            self.y = y

        self.dim_x = np.asscalar(np.prod(x.shape[1:]))
        self.dim_y = np.asscalar(np.prod(y.shape[1:]))

        if dim_X is not None:
            self.dim_X = list(dim_X)
        else:
            self.dim_X = [self.dim_x]

        if dim_Y is not None:
            self.dim_Y = list(dim_Y)
        else:
            self.dim_Y = [self.dim_y]

    def flatten_x(self) -> DataSetType:
        x_new = self.x.reshape((-1, np.prod(self.dim_x)))
        return type(self)(x_new, self.y)

    def get_x_by_y(self, y):
        idx = np.ravel(self.y == y)
        return self.x[idx, :]

    def subset(self, index: Union[np.ndarray, int, List, slice],
               shuffle: bool = False) -> DataSetType:
        return type(self)(self.x[index, :], self.y[index, :], shuffle=shuffle)

    def sample(self, size):
        _idx = np.random.randint(self.n_samples, size=size)
        return self.subset(_idx, shuffle=True)

    def random_split(self, ratio):
        part_1_size = int(ratio * self.n_samples)
        part_1_idx = np.random.choice(self.n_samples, size=part_1_size, replace=False)
        part_2_idx = [i for i in range(self.n_samples) if i not in part_1_idx]
        return self.subset(part_1_idx), self.subset(part_2_idx)

    def next_batch(self, mb_size: int,
                   progress: Tuple[int, int]) -> (Tuple[int, int], DataSetType):
        (epoch, marker) = progress
        _ids = []
        for _idx in range(mb_size):
            _ids.append(marker)
            marker += 1
            if marker == self.n_samples:
                marker = 0
                epoch += 1

        return (epoch, marker), self.subset(_ids)

    def __str__(self):
        property_str = ", ".join([f"dim_x={self.dim_x}",
                                  f"dim_y={self.dim_y}",
                                  f"n_samples={self.n_samples}"])
        return f"{self.name}: {property_str}"

    def __repr__(self):
        return self.__str__()

    def __add__(self, d2: DataSetType, shuffle=True) -> DataSetType:
        if not (self.dim_x == d2.dim_x
                and self.dim_y == d2.dim_y):
            raise ValueError("Dimension mismatch")
        else:
            x_new = np.vstack((self.x, d2.x))
            y_new = np.vstack((self.y, d2.y))
            return type(self)(x_new, y_new, shuffle=shuffle)

    def __radd__(self, d2, shuffle=True):
        if d2 == 0:
            return self
        else:
            return self.__add__(d2, shuffle=shuffle)


class CategoricalDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = self.get_y_categories().ravel().shape[0]
        color_gen = get_color_gen()
        self.class_colors = [next(color_gen) for c in range(self.n_classes)]
        self.c = (
            self.get_y_1hot()[:, :, np.newaxis] * np.array(
                [self.class_colors])).sum(axis=1)

    @classmethod
    def from_1hot(cls,
                  x: np.ndarray,
                  y: np.ndarray, *args, **kwargs):
        y_dense = y @ np.arange(y.shape[1])[:, np.newaxis]
        return cls(x=x, y=y_dense, *args, **kwargs)

    def get_sign_encoded_y(self):
        assert self.n_classes == 2
        return self.y * 2 - 1

    def get_y_1hot(self):
        if self.n_classes == 1:
            return self.y
        else:
            y_ = np.eye(self.n_classes)[self.y.astype(np.int8).ravel()]
            return y_

    def update_y_1hot(self):
        return type(self)(self.x, self.get_y_1hot(), shuffle=False)

    def get_y_categories(self):
        return np.unique(self.y.ravel())
