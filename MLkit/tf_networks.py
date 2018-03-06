import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.contrib import slim
import tensorflow.contrib.layers as cl
from typing import Tuple, List, Optional, Callable


def xavier_init(size: Tuple[int, int]) -> Tensor:
    in_dim = float(size[0])
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)


def scewl(logits: Tensor, labels: Tensor) -> Tensor:
    # A hacky wrapper of tf.nn.sigmoid_cross_entropy_with_logits
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=labels)


def smcewl(logits: Tensor, labels: Tensor) -> Tensor:
    # A hacky wrapper of tf.nn.sigmoid_cross_entropy_with_logits
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels)


def flatten(_input: Tensor) -> Tensor:
    print('Unfinished')
    dim = tf.reduce_prod(tf.shape(_input)[1:])
    return tf.reshape(_input, [-1, dim])


def simple_net(_in, n_in, n_out, n_hidden=128) -> Tensor:
    _W1 = tf.get_variable('W1', initializer=xavier_init((n_in, n_hidden)))
    _b1 = tf.get_variable('b1', initializer=tf.zeros(shape=[n_hidden]))

    _W2 = tf.get_variable('W2', initializer=xavier_init((n_hidden, n_out)))
    _b2 = tf.get_variable('b2', initializer=tf.zeros(shape=[n_out]))

    _h1 = tf.nn.relu(tf.matmul(_in, _W1) + _b1)
    _logit = tf.matmul(_h1, _W2) + _b2
    return _logit


def layer_dense(_in: Tensor, n_out: int, scope_name: str) -> Tensor:
    with tf.variable_scope(scope_name):
        _in_dim = int(_in.get_shape()[-1])
        _W = tf.get_variable('W', shape=(_in_dim, n_out),
                             initializer=cl.xavier_initializer(uniform=False))
        _b = tf.get_variable('b', initializer=tf.zeros(shape=[n_out]))
    return tf.matmul(_in, _W) + _b


def dense_net(z: Tensor,
              n_units: List[int],
              activation_fn: Callable[[Tensor], Tensor] = tf.nn.relu,
              drop_out: Optional[float] = None,
              batch_norm: bool = False,
              is_train: bool = False) -> Tensor:
    _flow = z
    for i, n in enumerate(n_units[:-1]):
        _flow = activation_fn(layer_dense(_flow, n, 'fc' + str(i)))
        if batch_norm:
            _flow = tf.layers.batch_normalization(_flow, training=is_train)
        if drop_out is not None:
            _flow = tf.layers.dropout(_flow, rate=drop_out, training=is_train)
    _flow = layer_dense(_flow, n_units[-1], 'fcFinal')
    return _flow


def le_conv_tune(x__: tf.Tensor, n_out: int,
                 activation_fn=tf.nn.relu,
                 drop_out: Optional[float] = None,
                 batch_norm=False,
                 is_train=True
                 ):
    net = tf.layers.conv2d(x__, 20, 5, activation=activation_fn, name='conv1')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool1')
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.layers.conv2d(net, 50, 5, activation=activation_fn, name='conv2')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool2')
    net = cl.flatten(net)
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    if drop_out is not None:
        net = tf.layers.dropout(net, rate=drop_out, training=is_train)
    _logits = net
    return _logits


def le_conv_tune_64(x__: tf.Tensor, n_out: int,
                    activation_fn=tf.nn.relu,
                    drop_out: Optional[float] = None,
                    batch_norm=False,
                    is_train=True
                    ):
    net = tf.layers.conv2d(x__, 16, 5, activation=activation_fn, name='conv1')
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool1')
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.layers.conv2d(net, 32, 5, activation=activation_fn, name='conv2')
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool2')
    net = tf.layers.conv2d(net, 64, 3, activation=activation_fn, name='conv3')
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool3')
    net = cl.flatten(net)
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    print(net.shape)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    if drop_out is not None:
        net = tf.layers.dropout(net, rate=drop_out, training=is_train)
    _logits = net
    return _logits


def conv64(x__: Tensor, n_out: int) -> Tensor:
    # In: 64x64
    net = tf.layers.conv2d(x__, 16, 5, activation=tf.nn.relu, name='conv1')
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool1')
    # 29x29
    net = tf.layers.conv2d(net, 32, 5, activation=tf.nn.relu, name='conv2')
    # 25x25
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool2')
    # 12x12
    net = tf.layers.conv2d(net, 64, 5, activation=tf.nn.relu, name='conv3')
    # 8x8
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool3')
    # 3x3
    net = cl.flatten(net)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    _logits = net
    return _logits


def le_cov_pp(x__, n_out):
    # In: 28x28
    net = slim.layers.conv2d(x__, 20, 5, scope='conv1')
    net = slim.layers.max_pool2d(net, 2, scope='pool1')
    net = slim.layers.conv2d(net, 50, 5, scope='conv2')
    net = slim.batch_norm(net, scale=True)
    net = slim.layers.max_pool2d(net, 2, scope='pool2')
    net = slim.layers.flatten(net)
    net = tf.layers.dense(net, 500, kernel_initializer=slim.initializers.xavier_initializer(False))
    net = slim.batch_norm(net, scale=True)
    net = tf.nn.dropout(net, 0.6)
    net = tf.layers.dense(net, n_out, kernel_initializer=slim.initializers.xavier_initializer(False))
    _logits = net
    return _logits


def DCGAN_D(_in):
    base_ch = 8
    filter_size = [4, 4]
    net = slim.layers.conv2d(_in, base_ch, filter_size, 2, scope='conv1', activation_fn=None)
    net = slim.batch_norm(net, scale=True)
    net = tf.nn.relu(net)
    net = slim.layers.conv2d(net, base_ch * 2, filter_size, 2, scope='conv2', activation_fn=None)
    net = slim.batch_norm(net, scale=True)
    net = tf.nn.relu(net)
    net = slim.layers.conv2d(net, base_ch * 4, filter_size, 2, scope='conv3', activation_fn=None)
    net = slim.batch_norm(net, scale=True)
    net = tf.nn.relu(net)
    net = slim.layers.conv2d(net, base_ch * 8, filter_size, 2, scope='conv4', activation_fn=None)
    net = slim.batch_norm(net, scale=True)
    net = tf.nn.relu(net)
    _logit = slim.layers.conv2d(net, 1, filter_size, scope='conv5', activation_fn=None)
    return _logit


def la_cov(_in, n_out, base_ch=16, filter_size=(4, 4)):
    net = slim.layers.conv2d(_in, base_ch, filter_size, 2, scope='conv1', activation_fn=None)
    net = slim.batch_norm(net, scale=True)
    net = tf.nn.relu(net)
    net = slim.layers.conv2d(net, base_ch * 2, filter_size, 2, scope='conv2', activation_fn=None)
    net = slim.batch_norm(net, scale=True)
    net = tf.nn.relu(net)
    net = slim.layers.flatten(net, scope='flatten3')
    _logit = layer_dense(net, n_out, 'linear_map')
    return _logit


def deconv64(_in: Tensor):
    net = tf.layers.dense(_in, 2 * 2 * 256, activation=tf.nn.relu,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    net = tf.reshape(net, [-1, 2, 2, 256])
    net = tf.layers.conv2d_transpose(net, 128, 5, strides=2,
                                     activation=tf.nn.relu, name='conv1')
    # 7x7 ->
    net = tf.layers.conv2d_transpose(net, 64, 4, strides=2,
                                     activation=tf.nn.relu, name='conv2')
    # 16x16 ->
    net = tf.layers.conv2d_transpose(net, 32, 2, strides=2,
                                     activation=tf.nn.relu, name='conv3')
    # 32x32 ->
    net = tf.layers.conv2d_transpose(net, 1, 2, strides=2,
                                     activation=tf.nn.relu, name='conv4')
    # 64x64
    return net


def conv28(x__: Tensor, n_out: int) -> Tensor:
    # In: 28x28 to 32x32
    net = tf.layers.conv2d(x__, 16, 5, activation=tf.nn.relu, name='conv1')
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
    net = tf.layers.conv2d(net, 64, 5, activation=tf.nn.relu, name='conv2')
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool2')
    net = cl.flatten(net)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    _logits = net
    return _logits

# Alias:
strange_net = conv64
conv32 = conv28
cifar_net = conv32
