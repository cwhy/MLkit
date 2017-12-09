import numpy as np
import tensorflow as tf
from typing import Union, Callable

tf_Tensor = Union[tf.Tensor, tf.Variable, tf.constant]


def add_offset(_x: tf_Tensor) -> tf_Tensor:
    # The hack by https://github.com/tensorflow/tensorflow/issues/956
    return tf.pad(_x - 1, [[0, 0], [1, 0]], 'CONSTANT') + 1


def mixed_moments(_t: tf_Tensor, order: int) -> tf_Tensor:
    if order == 1:
        return _t
    n_vars = _t.get_shape()[1]
    r = [tf.expand_dims(_t[:, i], axis=1) * mixed_moments(_t[:, i:], order - 1) for i in range(n_vars)]
    return tf.concat(r, axis=1)


# Find number of mix-moments
def comb_with_rep(n: int, r: int) -> int:
    # When n >> r
    _p1 = np.prod(np.arange(n, n + r))
    r_fac = np.prod(np.arange(1, r + 1))
    return (_p1 / r_fac).__int__()


def get_n_moment_terms(x_dim: int, order: int) -> int:
    return comb_with_rep(x_dim + 1, order)


def make_moment_fn(order: int) -> Callable[[tf_Tensor], tf_Tensor]:
    def _moments_fn(_x: tf_Tensor) -> tf_Tensor:
        _t = add_offset(_x)
        return mixed_moments(_t, order)

    return _moments_fn


def accuracy(y_hat: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.arg_max(y_hat, 1),
                         tf.arg_max(y, 1)), tf.float32))


def n_wrong(y_hat: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(
        tf.cast(tf.not_equal(tf.arg_max(y_hat, 1),
                             tf.arg_max(y, 1)), tf.float32))
