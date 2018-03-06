from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple
import numpy as np
from MLkit.color import get_color_gen
from enum import Enum, auto


class Encoding(Enum):
    T_1HOT = auto()
    T_SIGN = auto()
    T_DENSE = auto()


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

        self.dim_x = int(np.asscalar(np.prod(x.shape[1:])))
        self.dim_y = int(np.asscalar(np.prod(y.shape[1:])))

        if dim_X is not None:
            self.dim_X = list(dim_X)
        else:
            self.dim_X = [self.dim_x]

        if dim_Y is not None:
            self.dim_Y = list(dim_Y)
        else:
            self.dim_Y = [self.dim_y]

    def flatten_x(self) -> DataSetType:
        x_new = self.x.reshape([-1, self.dim_x])
        return self.new(x_new, self.y, shuffle=True, name_mod='flattened')

    def get_x_by_y(self, y):
        idx = np.ravel(self.y == y)
        return self.x[idx, :]

    def subset(self, index: Union[np.ndarray, int, List, slice],
               name_mod: Optional[str] = 'subset',
               shuffle: bool = False) -> DataSetType:
        return self.new(self.x[index, :], self.y[index, :], shuffle, name_mod)

    def sample(self, size, name_mod: Optional[str] = 'sample') -> DataSetType:
        _idx = np.random.randint(self.n_samples, size=size)
        return self.subset(_idx, name_mod=name_mod, shuffle=True)

    def random_split(self, ratio,
                     name_mods: Optional[Tuple[str, str]] = None):
        part_1_size = int(ratio * self.n_samples)
        part_1_idx = np.random.choice(self.n_samples, size=part_1_size, replace=False)
        part_2_idx = [i for i in range(self.n_samples) if i not in part_1_idx]
        if name_mods is None:
            return (self.subset(part_1_idx),
                    self.subset(part_2_idx))
        else:
            return (self.subset(part_1_idx, name_mods[0]),
                    self.subset(part_2_idx, name_mods[1]))

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

    def new(self, x: np.ndarray, y: np.ndarray,
            shuffle: bool, name_mod: Optional[str] = None) -> DataSetType:
        if name_mod is None:
            new_name = self.name
        else:
            new_name = self.name + '_' + name_mod
        return DataSet(x, y,
                       shuffle=shuffle,
                       name=new_name)

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
    def __init__(self, *args,
                 y_encoding: Encoding = Encoding.T_DENSE,
                 n_classes: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.y_encoding = y_encoding
        if n_classes is not None:
            self.n_classes = n_classes
        elif y_encoding == Encoding.T_DENSE:
            self.n_classes = self.get_y_categories().ravel().shape[0]
        elif y_encoding == Encoding.T_1HOT:
            self.n_classes = self.y.shape[-1]
        elif y_encoding == Encoding.T_SIGN:
            self.n_classes = 2
        else:
            raise ValueError("Encoding must be one of the encoding type")

        color_gen = get_color_gen()
        self.class_colors = [next(color_gen) for c in range(self.n_classes)]
        self.c = (
                self.get_y_1hot()[:, :, np.newaxis] * np.array(
            [self.class_colors])).sum(axis=1)

    # @classmethod
    # def from_1hot(cls,
    #               x: np.ndarray,
    #               y: np.ndarray, *args, **kwargs):
    #     y_dense = y @ np.arange(y.shape[1])[:, np.newaxis]
    #     return cls(x=x, y=y_dense, *args, **kwargs)

    def get_sign_encoded_y(self):
        assert self.n_classes == 2
        assert self.y_encoding == Encoding.T_DENSE
        return self.y * 2 - 1

    def update_y_sign_encoded(self):
        assert self.n_classes == 2
        assert self.y_encoding == Encoding.T_DENSE
        return type(self)(self.x,
                          self.get_sign_encoded_y(),
                          y_encoding=Encoding.T_SIGN,
                          n_classes=self.n_classes,
                          shuffle=False)

    def get_y_1hot(self):
        if self.y_encoding == Encoding.T_1HOT:
            return self.y
        elif self.y_encoding == Encoding.T_DENSE:
            if self.n_classes == 1:
                return self.y
            else:
                y_ = np.eye(self.n_classes)[self.y.astype(np.int8).ravel()]
                return y_
        else:  # self.y_encoding == Encoding.T_SIGN
            y_d = (self.y + 1) / 2
            y_ = np.eye(self.n_classes)[y_d.astype(np.int8).ravel()]
            return y_

    def update_y_1hot(self):
        return type(self)(self.x,
                          self.get_y_1hot(),
                          y_encoding=Encoding.T_1HOT,
                          n_classes=self.n_classes,
                          shuffle=False)

    def get_y_categories(self):
        assert self.y_encoding == Encoding.T_DENSE
        return np.unique(self.y.ravel())

    def get_y_intID(self):
        assert self.y_encoding == Encoding.T_DENSE
        _, y_ids = np.unique(self.y, return_inverse=True)
        return y_ids

    def new(self, x: np.ndarray, y: np.ndarray,
            shuffle: bool, name_mod: Optional[str] = None) -> DataSetType:
        if name_mod is None:
            new_name = self.name
        else:
            new_name = self.name + '_' + name_mod
        return CategoricalDataSet(x, y,
                                  shuffle=shuffle,
                                  n_classes=self.n_classes,
                                  y_encoding=self.y_encoding,
                                  name=new_name)


class BaseDataSet(metaclass=ABCMeta):

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def dim_X(self):
        pass

    @property
    @abstractmethod
    def dim_Y(self):
        pass

    @property
    @abstractmethod
    def dim_x(self):
        pass

    @property
    @abstractmethod
    def dim_y(self):
        pass

    @abstractmethod
    def sample(self, mb_size, name_mod: Optional[str]) -> DataSetType:
        pass


class FixedDataSet(BaseDataSet):
    @abstractmethod
    def next_batch(self, mb_size: int,
                   progress: Tuple[int, int]) -> (Tuple[int, int], DataSetType):
        pass
