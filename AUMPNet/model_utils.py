#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-05-14"
__email__ = "cxweieee@126.com"

"""
Some useful functions.
"""

import os, glob
import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim import layers, arg_scope, variable_scope

def PReLU(inputs, scope="prelu"):
    """
    @param inputs, input tensor which will be used for prelu activation
    """
    with variable_scope.variable_scope(scope, "prelu", [inputs]) as sc:
        _alpha = tf.get_variable('alpha', shape=inputs.get_shape()[-1], 
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
        pos = tf.nn.relu(inputs)
        neg = _alpha * tf.nn.relu(-1 * inputs)
        net = pos + neg
        return net

