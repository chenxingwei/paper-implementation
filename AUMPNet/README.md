# Implementation of AUMPNet with tensorflow

## Introduction

The main structure are implementated the same the the paper: "AUMPNet: Simultaneous Action Units Detection and Intensity 
Estimation on Multipose Facial Images Using a Single Convolutional Neural Network". 

I modified the loss function as the sum of cross entropy loss for facial pose and expression class. The region convolution were
implementated similar to the structure proposed in "Deep region and multi-label learning for facial action units detection".

## Files:

aumpnet.py: network structure of AUMPNet.

model_utils.py:
  
regionConv.py:
