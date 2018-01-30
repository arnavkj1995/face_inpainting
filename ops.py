import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

F = tf.flags.FLAGS

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def minibatch_disc(input, num_kernels=10, kernel_size=5, scope="m_bat"):
    '''
    Modified from http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
    '''
    tensor_out = linear(input, num_kernels * kernel_size, scope=scope)  
    tensor_out = tf.reshape(tensor_out, (-1, num_kernels, kernel_size))  # [bat, B, C]
    tensor_out = tf.expand_dims(tensor_out, 3)  # [bat, B, C, 1]
    diffs = tensor_out - tf.transpose(tensor_out, [3, 1, 2, 0])  # [bat, B, C, bat]
    l1_norm = tf.reduce_sum(tf.abs(diffs), 2)  # [bat, B, bat]
    mb_feats = tf.reduce_sum(tf.exp(-l1_norm), 2)  # [bat, B]
    return tf.concat([input, mb_feats], 1)



def instance_norm(x):
    # instance normalization 
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))




def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                                (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, bias=True, pad='SAME',
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=pad)

        if bias == True:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             start_bias=0.0, bias=True, padding="SAME", name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding=padding)

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        if bias == True:
            biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(start_bias))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.1, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, bias=True,  stddev=0.01, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    #print ('In linear:::::::::::::      ', input_.get_shape())

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer() if F.dataset == "mnist"
                                 else tf.random_normal_initializer(stddev=stddev))
        if bias == True:
            bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
