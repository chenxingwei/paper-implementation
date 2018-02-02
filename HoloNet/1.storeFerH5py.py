#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-01-26"
__email__ = "cxweieee@126.com"

import h5py

from holonet_util import *

fp = h5py.File("fer2013_h5py/fer2013.h5", 'w')

train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_norm_fer2013("fer2013.csv")

fp["train_images"] = train_images
fp["train_labels"] = train_labels
fp["validation_images"] = validation_images
fp["validation_labels"] = validation_labels
fp["test_images"] = test_images
fp["test_labels"] = test_labels

fp.close()




