import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.layers as cl
from typing import Tuple, Union


def xavier_init(size: Tuple[int, int]):
    in_dim = float(size[0])
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)


def flatten(_input: tf.Tensor):
    print('Unfinished')
    dim = tf.reduce_prod(tf.shape(_input)[1:])
    return tf.reshape(_input, [-1, dim])


def simple_net(_in, n_in, n_out, n_hidden=128):
    _W1 = tf.get_variable('W1', initializer=xavier_init((n_in, n_hidden)))
    _b1 = tf.get_variable('b1', initializer=tf.zeros(shape=[n_hidden]))

    _W2 = tf.get_variable('W2', initializer=xavier_init((n_hidden, n_out)))
    _b2 = tf.get_variable('b2', initializer=tf.zeros(shape=[n_out]))

    _h1 = tf.nn.relu(tf.matmul(_in, _W1) + _b1)
    _logit = tf.matmul(_h1, _W2) + _b2
    return _logit


def layer_dense(_in, n_out, scope_name):
    with tf.variable_scope(scope_name):
        _in_dim = int(_in.get_shape()[-1])
        _W = tf.get_variable('W', shape=(_in_dim, n_out),
                             initializer=cl.xavier_initializer(uniform=False))
        _b = tf.get_variable('b', initializer=tf.zeros(shape=[n_out]))
    return tf.matmul(_in, _W) + _b


def dense_net(z, n_units,
              activation_fn=tf.nn.relu,
              drop_out: Union[None, float] = None,
              batch_norm=False,
              is_train=False):
    _flow = z
    for i, n in enumerate(n_units[:-1]):
        _flow = activation_fn(layer_dense(_flow, n, 'fc' + str(i)))
        if batch_norm:
            _flow = tf.layers.batch_normalization(_flow, training=is_train)
        if drop_out is not None:
            _flow = tf.layers.dropout(_flow, rate=drop_out, training=is_train)
    _flow = layer_dense(_flow, n_units[-1], 'fcFinal')
    return _flow


def le_conv(x__: tf.Tensor, n_out: int):
    net = tf.layers.conv2d(x__, 20, 5, activation=tf.nn.relu, name='conv1')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool1')
    net = tf.layers.conv2d(net, 50, 5, activation=tf.nn.relu, name='conv2')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool2')
    net = cl.flatten(net)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    _logits = net
    return _logits


def le_cov_pp(x__, n_out):
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
