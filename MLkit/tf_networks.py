import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.contrib import slim
import tensorflow.contrib.layers as cl
from typing import Tuple, List, Optional, Callable

ef = lambda x: tf.expand_dims(x, -1)  # expand forward
eb = lambda x: tf.expand_dims(x, 0)  # expand backward


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


def mini_rnn(_in: Tensor, out_dim_t: int, state_size: int):
    cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    rnn_out, _ = tf.nn.dynamic_rnn(cell, tf.expand_dims(_in, -1), dtype=tf.float32)
    W = tf.get_variable('output_embedding', (state_size, out_dim_t),
                        initializer=tf.truncated_normal_initializer())
    out_pred_logit = tf.reduce_sum(ef(rnn_out) * eb(eb(W)), axis=-2)
    return cl.flatten(out_pred_logit)


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


def conv64(x__: Tensor, n_out: int, is_train=True) -> Tensor:
    net = tf.layers.batch_normalization(x__, training=is_train)
    # In: 64x64 to 80x80
    net = tf.layers.conv2d(net, 16, 5, name='conv1')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool1')
    # 29x29 to 37x37
    net = tf.layers.conv2d(net, 32, 5, name='conv2')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    # 25x25 to 33x33
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool2')
    # 12x12 to 16x16
    net = tf.layers.conv2d(net, 64, 5, name='conv3')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    # 8x8 to 12x12
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool3')
    # 3x3 to 5x5
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
    # In: 28x28 <= shape <= 32x32
    net = tf.layers.conv2d(x__, 16, 5, activation=tf.nn.relu, name='conv1')
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
    net = tf.layers.conv2d(net, 64, 5, activation=tf.nn.relu, name='conv2')
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool2')
    net = cl.flatten(net)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    _logits = net
    return _logits


def conv_only28(x__: Tensor, n_out: int) -> Tensor:
    net = tf.layers.conv2d(x__, 32, 4, 1, activation=tf.nn.relu, name='conv1')
    net = tf.layers.conv2d(net, 32, 4, 1, activation=tf.nn.relu, name='conv2')
    net = tf.layers.conv2d(net, 32, 4, 2, activation=tf.nn.relu, name='conv3')
    net = tf.layers.conv2d(net, 32, 4, 2, activation=tf.nn.relu, name='conv4')
    net = cl.flatten(net)

    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    _logits = net
    return _logits


def deconv28(in_: Tensor, is_train: bool = False) -> Tensor:
    net = tf.layers.dense(in_, 32 * 4 * 4, activation=tf.nn.relu,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    net = tf.reshape(net, [-1, 4, 4, 32])
    net = tf.layers.conv2d_transpose(net, 32, 4, strides=2, name='deconv1')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, 32, 4, strides=2, name='deconv2')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, 32, 4, strides=1, name='deconv3')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, 1, 4, strides=1,
                                     activation=None, name='deconv4')
    _logits = net
    return _logits


def deconv80(in_: Tensor, out_channels: int = 1, is_train: bool = False) -> Tensor:
    size0 = 6
    net = tf.layers.dense(in_, 64 * size0 * size0, activation=tf.nn.relu,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    net = tf.reshape(net, [-1, size0, size0, 64])
    net = tf.layers.conv2d_transpose(net, 64, 4, strides=1, name='deconv1')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, 32, 4, strides=2, name='deconv2')
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, 32, 2, strides=2, name='deconv3')
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, out_channels, 2, strides=2, name='deconv4')
    _logits = net
    return _logits


def get_variational_layer(_logits: Tensor, dim_z: int) -> Tuple[Tensor, Tensor]:
    Z_mean = tf.layers.dense(_logits, dim_z, activation=None,
                             kernel_initializer=cl.xavier_initializer(uniform=False))
    Z_logvar = tf.layers.dense(_logits, dim_z, activation=None,
                               kernel_initializer=cl.xavier_initializer(uniform=False))
    eps = tf.random_normal(shape=tf.shape(Z_mean))
    Z = Z_mean + eps * (tf.exp(Z_logvar / 2))

    kl_losses = 0.5 * tf.reduce_sum(tf.exp(Z_logvar) + Z_mean ** 2 - 1. - Z_logvar, 1)
    return Z, kl_losses


# Alias:
strange_net = conv64
conv32 = conv28
conv80 = conv64
cifar_net = conv32
