################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################
#from pylab import *
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook

from numpy import *
import os
import numpy as np
import time
import matplotlib.image as mpimg
import urllib
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize
from sklearn.externals import joblib
from scipy.ndimage import filters
from numpy import random
from sklearn.svm import SVR
from caffe_classes import class_names
import urllib
import threading

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


DatasetFolder='../Dataset/'
batch_size= 1
queue_capacity=50*batch_size
train_batch_limit=2*batch_size
train=10

# load_data =  np.loadtxt("../Dataset/URLs_1_1.csv",dtype='str,float,float', delimiter=',',usecols=(0,2))
load_data =  np.genfromtxt("../Dataset/URLs_1_1.csv",dtype=None, delimiter=',')
print(load_data[0])
print(load_data.shape[0])

# load_data=np.asarray(load_data)
# print load_data.shape
# print(load_data[0][0])
# print(load_data.dtype)
# print(load_data.shape)
# print (load_data[0,1,2,3,4][0])
# print load_data
# img=np.empty([0,227,227,3])
v_i=np.empty([0,2])
for i in range(10):
  st=load_data[i][0]
  print i
  st=st.replace("https","http")
  print(imread(urllib.urlopen(st)).shape)
  print st
  v_i=np.concatenate((v_i,[[load_data[i][1],load_data[i][2]]]),axis=0)


print(v_i.shape)
print(v_i)

