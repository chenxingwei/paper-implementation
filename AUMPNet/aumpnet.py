#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-05-14"
__email__ = "cxweieee@126.com"

import os, glob, math, random
import tensorflow as tf
from tensorflow.contrib.slim import layers, arg_scope, variable_scope, batch_norm
import numpy as np

from model_utils import PReLU

"""
Implementation of AUMPNet proposed by the paper:

AUMPNet: simultaneous Action Units detection and intensity estimation on multipose facial images using a single convolutional neural network

for multi-pose face expression recognition
"""

def aumpnet(inputs, 
            num_classes=54, 
            num_poses=4,
            is_training=True, 
            dropout_keep_prob=0.6,
            scope="aumpnet"):
    """
    @param inputs, input tensor with shape [batch_size, height, width, channel]
                    inputs suggested have size "132 x 132 x 3"
    @param num_classes, the number of expression classes
    @param num_poses, the number of poses
    @param is_training, whether the process update the weights
    @param dropout_keep_prob, the probability to keep the nodes when using dropout
    """
    with variable_scope.variable_scope(scope, "aumpnet", [inputs]) as sc:
        net = layers.repeat(inputs, 1, layers.conv2d, 16, [5,5], padding="VALID", 
                            activation_fn=None, scope="conv1")
        # 128 * 128 * 16

        net_up = batch_norm(net, scope="up-batchnorm1")
        net_up = PReLU(net_up, scope="up-prelu1")
        net_up = region_conv(net_up, scope="up-regionconv")
        net_up = net_up + net
        net_up = batch_norm(net_up, scope="up-batchnorm2")
        net_up = layers.max_pool2d(net_up, [2, 2], scope="up-pool1")
        net_up = layers.repeat(net_up, 1, layers.conv2d, 32, [5, 5], 
                            activation_fn=None, scope="up-conv1")
        net_up = PReLU(net_up, scope="up-prelu1")
        net_up = layers.max_pool2d(net_up, [2, 2], scope="up-pool2")
        net_up = layers.repeat(net_up, 1, layers.conv2d, 64, [3, 3], 
                            activation_fn=None, scope="up-conv2")
        net_up = PReLU(net_up, scope="up-prelu2")
        # 32 x 32 x 64

        net_down = layers.max_pool2d(net, [2, 2], scope="down-pool1")
        net_down = layers.repeat(net_down, 1, layers.conv2d, 32, [3, 3], 
                            activation_fn=None, scope="down-conv1")
        net_down = PReLU(net_down, scope="down-prelu1")
        net_down = layers.max_pool2d(net_down, [2, 2], scope="down-pool2")
        net_down = layers.repeat(net_down, 1, layers.conv2d, 64, [3, 3], 
                            activation_fn=None, scope="down-conv2")
        net_down = PReLU(net_down, scope="down-prelu2")
        # 32 x 32 x 64

        net_concat = tf.concat([net_up, net_down], -1)
        # 32 x 32 x 128
        
        net_concat = layers.flatten(net_concat, scope="flatten")
        net_pose = layers.fully_connected(net_concat, 160, 
                                activation_fn=tf.nn.relu, scope="pfc6")
        net_pose = layers.dropout(net_pose, dropout_keep_prob, 
                                is_training=is_training, scope="pdropout")
        net_pose = layers.fully_connected(net_pose, 160,
                                activation_fn=tf.nn.relu, scope="pfc7")
        net_pose = layers.fully_connected(net_pose, num_poses,
                                actifvation_fn=tf.nn.relu, scope="pfc8")

        net_expr = layers.fully_connected(net_concat, 2000, 
                                activation_fn=tf.nn.relu, scope="fc6")
        net_expr = layers.dropout(net_expr, dropout_keep_prob, 
                                is_training=is_training, scope="dropout")
        net_expr = layers.fully_connected(net_expr, 2000,
                                activation_fn=tf.nn.relu, scope="fc7")
        net_expr = layers.fully_connected(net_expr, num_classes,
                                actifvation_fn=tf.nn.relu, scope="fc8")
        return [net_expr, net_pose]

