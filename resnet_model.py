#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
import json

from tensorpack.utils import logger
from tensorpack.tfutils.argscope import argscope
from tensorpack.models import (
    Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm, FullyConnected, BNReLU)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(ret, name='block_output')


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                current_stride = stride if i == 0 else 1
                l = block_func(l, features, current_stride)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, use_bias=False,
                     kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits


def denoising(name, l, denoise_func_str):
    with tf.variable_scope(name):
        # assign different denoise function
        if denoise_func_str.startswith('nonlocal'):
            _, embed, softmax, maxpool, avgpool = denoise_func_str.split(".")
            f = non_local_op(l, json.loads(embed.split("_")[1].lower()), json.loads(softmax.split("_")[1].lower()), int(maxpool.split("_")[1]), int(avgpool.split("_")[1]))
        elif denoise_func_str.startswith('void'):
            f = l
        elif denoise_func_str.startswith('avgpool'):
            _, pooling_size = denoise_func_str.split(".")
            f = avgpool_op(l, int(pooling_size.split("_")[1]))
        elif denoise_func_str.startswith('globalavgpool'):
            f = globalavgpool_op(l)
        elif denoise_func_str.startswith('medianpool'):
            _, pooling_size = denoise_func_str.split(".")
            f = medianpool_op(l, int(pooling_size.split("_")[1]))
        f = Conv2D('conv', f, l.get_shape()[1], 1, strides=1, activation=get_bn(zero_init=True))
        l = l + f
    return l


def non_local_op(l, embed, softmax, maxpool, avgpool):
    if embed:
        n_in = l.get_shape().as_list()[1]
        theta = Conv2D('embedding_theta', l, n_in/2, 1, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        phi = Conv2D('embedding_phi', l, n_in/2, 1, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    else:
        theta, phi = l, l
    g_orig = g = l
    # whether apply pooling function
    assert (avgpool == 1 or maxpool == 1)
    if maxpool > 1:
       phi = MaxPooling('pool_phi', phi, pool_size=maxpool, stride=maxpool)
       g = MaxPooling('pool_g', g, pool_size=maxpool, stride=maxpool)
    if avgpool > 1:
       phi = AvgPooling('pool_phi', phi, pool_size=avgpool, stride=avgpool)
       g = AvgPooling('pool_g', g, pool_size=avgpool, stride=avgpool)
    # flatten tensors
    theta_flat = tf.reshape(theta, [tf.shape(theta)[0], tf.shape(theta)[1], -1])
    phi_flat = tf.reshape(phi, [tf.shape(phi)[0], tf.shape(phi)[1], -1])
    g_flat = tf.reshape(g, [tf.shape(g)[0], tf.shape(g)[1], -1])
    theta_flat.set_shape([theta.shape[0], theta.shape[1], theta.shape[2] * theta.shape[3] if None not in theta.shape[2:] else None])
    phi_flat.set_shape([phi.shape[0], phi.shape[1], phi.shape[2] * phi.shape[3] if None not in phi.shape[2:] else None])
    g_flat.set_shape([g.shape[0], g.shape[1], g.shape[2] * g.shape[3] if None not in g.shape[2:] else None])
    # Compute production
    f = tf.matmul(theta_flat, phi_flat, transpose_a=True)
    if softmax:
        f = tf.nn.softmax(f)
    else:
        f = f / tf.cast(tf.shape(f)[-1], f.dtype)
    out = tf.matmul(g_flat, f, transpose_b=True)
    return tf.reshape(out, tf.shape(g_orig))


def resnet_denoising_backbone(image, num_blocks, group_func, block_func, denoise_func_str):
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, use_bias=False,
                     kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = denoising('group0_denoise_{}'.format(denoise_func_str), l, denoise_func_str)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = denoising('group1_denoise_{}'.format(denoise_func_str), l, denoise_func_str)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = denoising('group2_denoise_{}'.format(denoise_func_str), l, denoise_func_str)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = denoising('group3_denoise_{}'.format(denoise_func_str), l, denoise_func_str)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits
