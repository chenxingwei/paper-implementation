# HoloNet

In this repository, I am trying to implement HoloNet, proposed in "HolotNet: Towards Robust Emotion Recognition in the Wild" by intel.

## Features

1. HoloNet integrated both Inception structure and Residual network into the its deep CNN architecture.

2. HoloNet applied a modified Concatenated Rectified Linear Units (CReLU).

3. HoloNet takes original gray images, corresponding basic Local Binary Pattern (LBP) feature images, and correponding mean LBP images and the three channels of the inputs.

## Some details about algorithms and scripts

### 1. Local Binary Pattern (LBP)

LBP calculates the local feature, following comes the basic LBP feature extraction example. 

![](https://github.com/chenxingwei/paper-implementation/blob/master/HoloNet/images/lbp.png)

LBP compares the gray level value of the around 8 pixels to the center pixel, if the outer pixel is larger than the center pixel, set 1, otherwise, set 0. Then the binary string with length 8 could be transformed to 0-255 value, regarding as the LBP feature.

### 2. mean LBP

Similar to LBP, but the around 8 pixels do compare to the mean value of these 8 pixels instead. 

![](https://github.com/chenxingwei/paper-implementation/blob/master/HoloNet/images/mlmp.png)

### 3. Scripts

holonet_util.py: contains basic functions including LBP and mLBP feature extraction, FER2013 data reading, feature calculating and normalization.

holoNet_train.py: contains the train example for FER2013 datasets


