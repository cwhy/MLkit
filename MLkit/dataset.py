from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple
import numpy as np
from MLkit.color import get_N, ColorF
from enum import Enum, auto


class Encoding(Enum):
    T_1HOT = auto()
    T_SIGN = auto()
    T_DENSE = auto()
    T_NAME = auto()


Dimensions = Union[List[int], None, np.ndarray]
DataSetType = Union['DataSet', 'CategoricalDataSet']


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


class DataSet(BaseDataSet):
    @property
    def dim_y(self):
        return int(np.asscalar(np.prod(self.dim_Y)))

    @property
    def dim_x(self):
        return int(np.asscalar(np.prod(self.dim_X)))

    @property
    def dim_Y(self):
        return list(self._dim_Y)

    @property
    def dim_X(self):
        return list(self._dim_X)

    @property
    def name(self):
        return self._name

    @property
    def n_samples(self):
        return self.x.shape[0]

    @property
    def Y(self):
        return self.y.reshape([self.n_samples] + self.dim_Y)

    @property
    def X(self):
        return self.x.reshape([self.n_samples] + self.dim_X)

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 dim_X: Dimensions = None,
                 dim_Y: Dimensions = None,
                 name: Optional[str] = None,
                 shuffle: bool = True):

        if name is not None:
            self._name = name
        else:
            self._name = "Unnamed_DataSet"

        if dim_X is not None:
            self._dim_X = dim_X
        else:
            self._dim_X = x.shape[1:]

        if dim_Y is not None:
            self._dim_Y = dim_Y
        else:
            self._dim_Y = y.shape[1:]

        self.x = x.reshape([-1, self.dim_x])
        self.y = y.reshape([-1, self.dim_y])

        if shuffle:
            _idx = np.random.permutation(self.n_samples)
            self.x = self.x[_idx, :]
            self.y = self.y[_idx, :]

    def get_x_by_y(self, y):
        idx = np.ravel(self.y == y)
        return self.x[idx, :]

    def count_y(self, y):
        return np.sum(np.ravel(self.y == y))

    def subset(self, index: Union[np.ndarray, int, List, slice],
               name_mod: Optional[str] = 'subset',
               shuffle: bool = False) -> DataSetType:
        if name_mod is None:
            new_name = self.name
        else:
            new_name = self.name + '_' + name_mod
        return self.new(self.x[index, :], self.y[index, :], shuffle,
                        dim_X=self.dim_X,
                        dim_Y=self.dim_Y,
                        name=new_name)

    def sample(self, size, name_mod: Optional[str] = 'sample') -> DataSetType:
        _idx = np.random.randint(self.n_samples, size=size)
        return self.subset(_idx, name_mod=name_mod, shuffle=True)

    def random_split(self, ratio,
                     name_mods: Optional[Tuple[str, str]] = None):
        part_1_size = int(ratio * self.n_samples)
        part_1_idx = np.random.choice(self.n_samples, size=part_1_size, replace=False)
        part_2_idx = np.array([i for i in range(self.n_samples) if i not in part_1_idx])
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

    def rename(self, name):
        return self.new(self.x, self.y, shuffle=False, name=name)

    def new(self, x: np.ndarray, y: np.ndarray,
            shuffle: bool,
            dim_X: Dimensions = None,
            dim_Y: Dimensions = None,
            name: Optional[str] = None) -> DataSetType:
        if name is None:
            name = self.name
        return DataSet(x, y,
                       dim_X=dim_X,
                       dim_Y=dim_Y,
                       shuffle=shuffle,
                       name=name)

    def __str__(self):
        property_str = ", ".join([f"dim_x={self.dim_x}",
                                  f"dim_y={self.dim_y}",
                                  f"n_samples={self.n_samples}"])
        return f"{self.name}: {property_str}"

    def __repr__(self):
        return self.__str__()

    def __add__(self, d2: DataSetType, shuffle=True) -> DataSetType:
        assert self.dim_x == d2.dim_x
        assert self.dim_y == d2.dim_y
        x_new = np.vstack((self.x, d2.x))
        y_new = np.vstack((self.y, d2.y))
        return self.new(x_new, y_new, shuffle=shuffle)

    def __radd__(self, d2, shuffle=True):
        if d2 == 0:
            return self
        else:
            return self.__add__(d2, shuffle=shuffle)


class CategoricalDataSet(DataSet):
    def __init__(self, *args,
                 y_in_encoding: Encoding = Encoding.T_NAME,
                 n_classes: Optional[int] = None,
                 class_colors: Optional[ColorF] = None,
                 category_names: Optional[List[Union[str]]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if y_in_encoding == Encoding.T_NAME and self.dim_y == 1:
            assert category_names is None
            _ids, self.y = np.unique(self.y, return_inverse=True)
            self.y = self.y[:, np.newaxis]
            self.category_names = list(map(str, _ids))
            self.y_encoding = Encoding.T_DENSE
            self.n_classes = len(_ids)
        else:
            self.y_encoding = y_in_encoding
            if y_in_encoding == Encoding.T_DENSE:
                assert self.dim_y == 1
                if n_classes is not None:
                    self.n_classes = n_classes
                else:
                    self.n_classes = self.get_y_categories().ravel().shape[0]
            elif y_in_encoding == Encoding.T_1HOT:
                assert len(self.dim_Y) == 1
                self.n_classes = self.y.shape[-1]
            elif y_in_encoding == Encoding.T_SIGN:
                self.n_classes = 2
            else:
                raise ValueError("Encoding must be one of the encoding type")
            if category_names is None:
                self.category_names = list(map(str, range(self.n_classes)))
            else:
                self.category_names = category_names

        if class_colors is not None:
            self.class_colors = class_colors
        else:
            self.class_colors: ColorF = get_N(self.n_classes, phase=0.4)
        self.c = (
                self.get_y_1hot()[:, :, np.newaxis] * np.array(
            [self.class_colors])).sum(axis=1)

    #   @classmethod
    #   def from_name(cls,
    #                 x: np.ndarray,
    #                 y: np.ndarray, *args, **kwargs):
    #       y_dense = y @ np.arange(y.shape[1])[:, np.newaxis]
    #       return cls(x=x, y=y_dense, *args, **kwargs)

    def count_y(self, y):
        if self.y_encoding == Encoding.T_1HOT:
            return self.update_y_dense().count_y(y)
        elif self.y_encoding == Encoding.T_DENSE:
            return super().count_y(y)
        else:  # self.y_encoding == Encoding.T_SIGN
            assert y == 1 or y == -1
            return super().count_y(y)

    def count_y_name(self, y_name):
        y = self.category_names.index(y_name)
        if self.y_encoding == Encoding.T_1HOT:
            return self.update_y_dense().count_y(y)
        elif self.y_encoding == Encoding.T_DENSE:
            return super().count_y(y)
        else:  # self.y_encoding == Encoding.T_SIGN
            assert y_name == 1 or y_name == -1
            return super().count_y(y)

    def get_sign_encoded_y(self):
        assert self.n_classes == 2
        assert self.y_encoding == Encoding.T_DENSE
        return self.y * 2 - 1

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

    def get_y_dense(self):
        if (self.y_encoding == Encoding.T_DENSE
                or self.y_encoding == Encoding.T_SIGN):
            return self.y
        else:  # self.y_encoding == Encoding.T_1HOT:
            if self.n_classes == 1:
                return self.y
            else:
                y_ = np.where(self.y)[-1]
                # y_ = self.y @ np.arange(self.y.shape[1])[:, np.newaxis]
                return y_

    def update_y_sign_encoded(self):
        assert self.n_classes == 2
        assert self.y_encoding == Encoding.T_DENSE
        return self.new(self.x,
                        self.get_sign_encoded_y(),
                        shuffle=False,
                        y_encoding=Encoding.T_SIGN,
                        n_classes=self.n_classes)

    def update_y_1hot(self):
        return self.new(self.x,
                        self.get_y_1hot(),
                        shuffle=False,
                        y_encoding=Encoding.T_1HOT,
                        n_classes=self.n_classes)

    def update_y_dense(self):
        return self.new(self.x,
                        self.get_y_dense(),
                        shuffle=False,
                        y_encoding=Encoding.T_DENSE)

    def get_y_categories(self):
        assert self.y_encoding == Encoding.T_DENSE
        return np.unique(self.y.ravel())

    def new(self, x: np.ndarray, y: np.ndarray,
            shuffle: bool,
            dim_X: Dimensions = None,
            dim_Y: Dimensions = None,
            y_encoding: Optional[Encoding] = None,
            n_classes: Optional[int] = None,
            name: Optional[str] = None) -> 'CategoricalDataSet':
        if name is None:
            new_name = self.name
        else:
            new_name = name
        if y_encoding is None:
            y_encoding = self.y_encoding
        if n_classes is None:
            n_classes = self.n_classes

        return CategoricalDataSet(x, y,
                                  shuffle=shuffle,
                                  n_classes=n_classes,
                                  category_names=self.category_names,
                                  class_colors=self.class_colors,
                                  y_in_encoding=y_encoding,
                                  dim_X=dim_X,
                                  dim_Y=dim_Y,
                                  name=new_name)

    def __add__(self, d2: DataSetType, shuffle=True) -> 'CategoricalDataSet':
        raise NotImplementedError('Not Properly implemented')


class FixedDataSet(BaseDataSet):
    @abstractmethod
    def next_batch(self, mb_size: int,
                   progress: Tuple[int, int]) -> (Tuple[int, int], DataSetType):
        pass
