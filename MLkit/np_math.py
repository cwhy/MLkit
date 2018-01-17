import numpy as np


def add1(X: np.ndarray, axis: int = 0) -> np.ndarray:
    new_shape = list(X.shape)
    new_shape[axis] += 1
    X1 = np.ones(new_shape)
    slices = tuple(slice(None, -1, None)
                   if i == axis else
                   slice(None, None, None)
                   for i, _ in enumerate(new_shape))

    X1.__setitem__(slices, X)
    return X1


def accuracy(_y: np.ndarray, _y_hat: np.ndarray) -> float:
    return float(np.mean(_y == _y_hat))
