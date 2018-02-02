#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-01-23"
__email__ = "cxweieee@126.com"

import os, glob
import tensorflow as tf
from tensorflow.contrib.slim import layers, arg_scope, variable_scope, losses, l2_regularizer
import tensorlayer as tl
import h5py, random
import numpy as np

def CReLU(inputs):
    """
    """
    net1 = inputs * (-1)
    net1 = tf.nn.relu(net1)
    net = tf.concat([inputs, net1], 3)
    net = tf.nn.relu(net)
    return net

def oneHot(labels, num_classes=7):
    """
    """
    n = len(labels)
    res = [[0 for i in xrange(num_classes)] for j in xrange(n)]
    for i in xrange(n):
        res[i][labels[i]] = 1
    return np.array(res)
    


def holoNet(inputs, num_classes=7, is_training=True, dropout_keep_prob=0.5, scope="holonet"):
    """
    Implementation of the HoloNet proposed at:
    HoloNet: Towards Robust Emotion Recognition in the Wild
    """
    with variable_scope.variable_scope(scope, "holonet", [inputs]) as sc:
        net = layers.repeat(inputs, 1, layers.conv2d, 8, [7,7], activation_fn=None, scope="conv1")
        net = CReLU(net)
        # 128 * 128 * 16
        net = layers.max_pool2d(net, [3, 3], padding='SAME', scope="pool1")

        net_res1 = net
        net_res1 = layers.repeat(net_res1, 1, layers.conv2d, 32, [1, 1], activation_fn=None, scope="side-conv1")
        net = layers.repeat(net, 1, layers.conv2d, 12, [1, 1], activation_fn=None, scope="res1-conv1")
        net = layers.repeat(net, 1, layers.conv2d, 12, [3, 3], activation_fn=None, scope="res1-conv2")
        net = CReLU(net)
        net = layers.repeat(net, 1, layers.conv2d, 32, [1, 1], activation_fn=None, scope="res1-conv3")
        net = net + net_res1
        net_res2 = net
        net = layers.repeat(net, 1, layers.conv2d, 12, [1, 1], activation_fn=None, scope="res2-conv1")
        net = layers.repeat(net, 1, layers.conv2d, 12, [3, 3], activation_fn=None, scope="res2-conv2")
        net = CReLU(net)
        net = layers.repeat(net, 1, layers.conv2d, 32, [1, 1], activation_fn=None, scope="res2-conv3")
        net = net + net_res2
        net_res3 = net
        net_res3 = layers.repeat(net_res3, 1, layers.conv2d, 48, [1, 1], stride=2, activation_fn=None, scope="side-conv2")
        net = layers.repeat(net, 1, layers.conv2d, 16, [1, 1], stride=2, activation_fn=None, scope="res3-conv1")
        net = layers.repeat(net, 1, layers.conv2d, 16, [3, 3], activation_fn=None, scope="res3-conv2")
        net = CReLU(net)
        net = layers.repeat(net, 1, layers.conv2d, 48, [1, 1], activation_fn=None, scope="res3-conv3")
        net = net + net_res3
        net_res4 = net
        net = layers.repeat(net, 1, layers.conv2d, 16, [1, 1], activation_fn=None, scope="res4-conv1")
        net = layers.repeat(net, 1, layers.conv2d, 16, [3, 3], activation_fn=None, scope="res4-conv2")
        net = CReLU(net)
        net = layers.repeat(net, 1, layers.conv2d, 48, [1, 1], activation_fn=None, scope="res4-conv3")
        net = net + net_res4
 
        net_inception = layers.repeat(net, 1, layers.conv2d, 64, [1, 1], stride=2, scope="inception-conv1")
        net1 = layers.repeat(net, 1, layers.conv2d, 24, [1, 1], stride=2, scope="inception-conv1-1")
        net2 = layers.repeat(net, 1, layers.conv2d, 16, [1, 1], stride=2, scope="inception-conv2-1")
        net2 = layers.repeat(net2, 1, layers.conv2d, 32, [3, 3], scope="inception-conv2-2")
        net3 = layers.repeat(net, 1, layers.conv2d, 12, [1, 1], stride=2, scope="inception-conv3-1")
        net3 = layers.repeat(net3, 1, layers.conv2d, 16, [3, 3], scope="inception-conv3-2")
        net3 = layers.repeat(net3, 1, layers.conv2d, 16, [3, 3], scope="inception-conv3-3")
        net4 = layers.max_pool2d(net, [3, 3], padding='SAME', scope="pool2")
        net4 = layers.repeat(net4, 1, layers.conv2d, 32, [1, 1], scope="inception-conv4-1")
        net = tf.concat([net1, net2, net3, net4], 3)
        net = layers.repeat(net, 1, layers.conv2d, 64, [1, 1],scope="inception-conv2")
        net = net + net_inception
        net = layers.max_pool2d(net, [2, 2], padding='SAME', scope="pool3")

        net = layers.flatten(net, scope="flatten")
        net = layers.fully_connected(net, 1024, activation_fn=tf.nn.relu, weights_regularizer=l2_regularizer(0.01), scope="fc5")
        net = layers.dropout(net, dropout_keep_prob, is_training=is_training, scope="dropout")
        net = layers.fully_connected(net, num_classes, activation_fn=None, scope="fc6")

        return net
        
def train_fer2013(train_images, train_labels, validation_images, validation_labels, model_dir="./holonet_models", n_epochs=500, batch_size=64, learning_rate=0.001):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
    keep_prob = tf.placeholder(tf.float32)
    is_training = True
    num_classes = 7
    logits = holoNet(x, num_classes=num_classes, is_training=is_training, dropout_keep_prob=keep_prob)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    loss_regularization = losses.get_regularization_losses()
    total_loss = cross_entropy + loss_regularization
    loss, loss_update_op = tf.metrics.mean(cross_entropy)    

    #y = tf.argmax(tf.nn.softmax(logits), axis=1)
    acc, acc_update_op = tf.metrics.accuracy(tf.argmax(y_,axis=1), tf.argmax(logits, axis=1))

    # tensorboard

    tf.summary.scalar('cross_entropy_loss', loss)
    tf.summary.scalar("accuracy", acc)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./train_op", sess.graph)
    val_writer = tf.summary.FileWriter("./val_op")



    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    sess.run(tf.global_variables_initializer())

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        restore_var_list = [var for var in tf.global_variables() if 'Adam' not in var.op.name and "beta" not in var.op.name]
        restorer = tf.train.Saver(restore_var_list)
        restorer.restore(sess, ckpt.model_checkpoint_path)
    # define variables saver
    save_var_list = [var for var in tf.global_variables() if 'Adam' not in var.op.name and "beta" not in var.op.name]
    saver = tf.train.Saver(save_var_list)
    save_path = os.path.join(model_dir, 'holonet_fer2013.ckpt')


    max_acc = 0
    n_train = len(train_labels)
    n_validation = len(validation_labels)
    for epoch in range(n_epochs):
        n_batch = int(n_train / batch_size)
        sess.run(tf.local_variables_initializer())
        for i, (train_xs, train_ys) in enumerate(tl.iterate.minibatches(train_images, train_labels, batch_size, shuffle=True)):
            feed_dict = {x: train_xs, y_: train_ys, keep_prob:0.5}
            sess.run([train_step, acc_update_op, loss_update_op], feed_dict=feed_dict)
            loss_value, acc_value = sess.run([loss, acc])
        print '[%d/%d] - loss: %.4f - acc: %.4f' % (epoch, n_epochs, loss_value, acc_value)
        summary_str = sess.run(summary_op)
        train_writer.add_summary(summary_str, epoch)

        # validate
        sess.run(tf.local_variables_initializer())
        for i, (validation_xs, validation_ys) in enumerate(tl.iterate.minibatches(validation_images, validation_labels, batch_size, shuffle=False)):
            feed_dict = {x: validation_xs, y_: validation_ys, keep_prob:1.0}
            sess.run([loss_update_op, acc_update_op], feed_dict=feed_dict)
        loss_value, acc_value = sess.run([loss, acc])
        print 'val_loss: %.4f - val_acc %.4f' % (loss_value, acc_value)
        summary_str = sess.run(summary_op)
        val_writer.add_summary(summary_str, epoch)

        if acc_value > max_acc:
            saver.save(sess, save_path)
            max_acc = acc_value 
 
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fp = h5py.File("fer2013_h5py/fer2013.h5")
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = fp["train_images"][:], fp["train_labels"][:], fp["validation_images"][:], fp["validation_labels"][:], fp["test_images"][:], fp["test_labels"][:]
    fp.close()
    train_labels = oneHot(train_labels,num_classes=7)
    validation_labels = oneHot(validation_labels,num_classes=7)
    test_labelsf = oneHot(test_labels, num_classes=7)
    #train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_norm_fer2013("fer2013.csv")
    # change input file fer2013.csv, current read function is slow, improve it soon.

    # train_fer2013
    train_fer2013(train_images, train_labels, validation_images, validation_labels, model_dir="./holonet_models", n_epochs=200, batch_size=128, learning_rate=0.001)
 

