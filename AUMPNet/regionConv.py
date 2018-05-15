#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-05-14"
__email__ = "cxweieee@126.com"

"""
Utils for region convolution
"""

import os, glob
import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim import layers, arg_scope, variable_scope, batch_norm

def region_conv(inputs, block_size = 4, scope="region_conv"):
    """
    @param inputs, input tensor with batch_size x 128 x 128 x 16 shape
    @param, block_size, the block size should be dividable by the image feature map size 128
    """
    with variable_scope.variable_scope(scope, "region_conv", [inputs]) as sc:
        input_shape = inputs.get_shape()
        if input_shape[1] % block_size != 0:
            padding = (int(input_shape[1] / block_size) + 1) * block_size
            padding = int((padding - input_shape[1]) / 2)
            paddings = [[0,0],[padding, padding], [padding, padding], [0, 0]]
            inputs = tf.pad(inputs, paddings)
        input_shape = inputs.get_shape()
        size = input_shape[1] / block_size
        slice_starts = list(range(0, input_shape[1], size))
        nets = []
        for i in slice_starts:
            for j in slice_starts:
                tmp_input = tf.slice(inputs, [0,slice_starts[i], slice_starts[j],0], 
                                    [input_shape[0], size, size, input_shape[-1]])
                tmp_net = layers.repeat(tmp_input, 1, layers.conv2d, 16, [3, 3], activation_fn=None, scope="conv-%d-%d"%(i,j))
                tmp_net = batch_norm(tmp_net, scope="batch-%d-%d"%(i,j))
                tmp_net = PReLU(tmp_net, scope="prelu-%d-%d"%(i,j))
                nets.append(tmp_net)
        nets2 = []
        for i in range(0, block_size * block_size, block_size):
            tmp_net2 = tf.concat(nets[i:i+block_size], 2)
            nets2.append(tmp_net2)
        net = tf.concat(nets2, 1)
        return net



