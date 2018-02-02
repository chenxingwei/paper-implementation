#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-01-25"
__email__ = "cxweieee@126.com"

"""
This scripts prepares training, validation and test data of the holoNet.

"""
import numpy as np
from PIL import Image


def trans_label(labels):
    """
    Input label array like [1,2,3,4,0,1,2,3]
    """
    pass
    

def norm_fer2013_one(inlist):
    """
    @param inlist, input 2304 list
    @return 48 * 48 np.array
    """
    a = [float(x) for x in inlist]
    a = np.array(a)
    a = a.reshape((48,48))
    a = np.array(a, np.uint8)
    img = Image.fromarray(a)
    img = img.resize((128,128), resample=Image.BICUBIC)
    a = np.array(img, np.float32)
    a_basic = basic_lbp(a)
    a_mean = mean_lbp(a)
    mean = np.mean(a)
    std = np.std(a, ddof=1)
    if std == 0:
        std = 1.0
    a = (a-mean) / std
    mean = np.mean(a_basic)
    std = np.std(a_basic, ddof=1)
    if std == 0:
        std = 1.0
    a_basic = (a_basic - mean) / std
    mean = np.mean(a_mean)
    std = np.std(a_mean, ddof=1)
    if std == 0:
        std = 1.0
    a_mean = (a_mean - mean) / std
    res = np.stack([a, a_basic, a_mean], axis=2)
    return res

def zscore_array(inarray):
    a = np.array(inarray)
    mean = np.mean(a)
    std = np.std(a, ddof=1)
    a = (a-mean) / std
    return a


def read_norm_fer2013(infile):
    """
    @param infile, fer2013.csv file, each image is 48 * 48 size
    @return normalized (z-score) training, validation data and test data
    Images are normalized to 128 * 128
    """
    data = map(lambda x:x.strip().split(","), open(infile).readlines())
    train_labels = []
    train_images = []
    validation_labels = []
    validation_images = []
    test_labels = []
    test_images = []
    for line in data[1:]:
        label = int(float(line[0]) + 0.00001)
        image = norm_fer2013_one(line[1].split())
        if line[-1] == 'Training':
            train_images.append(image)
            train_labels.append(label)
        elif line[-1] == 'PublicTest':
            validation_images.append(image)
            validation_labels.append(label)
        else:
            test_images.append(image)
            test_labels.append(label)
    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


def convert_int8(bool_list):
    """
    @param bool_list, the bool_list are used, which has length 8.
    return int8
    """
    res = 0
    for x in bool_list:
        res = res * 2 + int(x)
    if res < 0:
        print bool_list
    return res

def basic_lbp(imagearray):
    """
    @param imagearray, 2d array of gray image.
    calculate the basic local binary pattern images of the input image array
    """
    imagearray = np.array(imagearray)
    padarray = np.pad(imagearray, 1, "constant")
    n, m = padarray.shape
    res = np.zeros((n-2, m-2))
    for i in xrange(1, n-1):
        for j in xrange(1, m-1):
            indexes = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1)]
            inlist = [padarray[x][y] for x, y in indexes]
            bool_list = np.array(inlist) > padarray[i][j]
            res[i-1][j-1] = convert_int8(bool_list) 
    return res

    
def mean_lbp(imagearray):
    """
    @param imagearray, 2d array of gray image.
    Implementation of the mean local binary pattern methods here.
    """
    imagearray = np.array(imagearray)
    padarray = np.pad(imagearray, 1, "constant")
    n, m = padarray.shape
    res = np.zeros((n-2, m-2))
    for i in xrange(1, n-1):
        for j in xrange(1, m-1):
            indexes = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1)]
            inlist = [padarray[x][y] for x, y in indexes]
            bool_list = np.array(inlist) > np.mean(inlist)
            res[i-1][j-1] = convert_int8(bool_list) 
    return res




