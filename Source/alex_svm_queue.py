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

# ****************Model Parameters******************8
DatasetFolder='../Dataset/'
batch_size= 1 #feeded to CNN at a go
queue_capacity=50*batch_size
train_batch_limit=2*batch_size #can be anything >=batch_size
train=10 
download_at_a_time=1 #suggested
output_categories=1000 #ALEXNET PROPERTY
# dequeue_size=10

#*****************************QUEUEING***************************************
queue_input_data = tf.placeholder(tf.float32, shape=[None,227,227,3])
queue_input_target = tf.placeholder(tf.float32, shape=[None, 2])
queue = tf.FIFOQueue(capacity=queue_capacity, dtypes=[tf.float32, tf.float32], shapes=[[227,227,3],[2]])
enqueue_op = queue.enqueue_many([queue_input_data,queue_input_target])
dequeue_op = queue.dequeue()
list_files=os.listdir(DatasetFolder)
list_files_len=len(list_files)
data_batch,target_batch = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=train_batch_limit)
ENQUEUE_LEFT=True

def enqueue(sess):
  """ Iterates over our data puts small junks into our queue."""

  sess2=tf.Session()
  for i in range(list_files_len):
    load_data =  np.genfromtxt(DatasetFolder+list_files[i],dtype=None, delimiter=',')
    under = 0
    FLAG=True

    max = (load_data.shape)[0]
    global train
    train=max
    print("Reading File",list_files[i],"started at index",i)
    while FLAG:

      upper = under + download_at_a_time
      # downloaded serially and not parallelys

      if(upper>=max):
        upper=max
        FLAG=False

      curr_data=np.empty([0,227,227,3])
      curr_target=np.empty([0,2])

      while (upper>under):

        if (load_data[under][1]==0 or load_data[under][2]==0):
          continue
        image_url=load_data[under][0]
        image_url=image_url.replace("https","http")
        # working on proxy server
        under=under+1
        try:
          (urllib.urlretrieve(image_url,"Test.jpg"))
			# Pulling image from URL        
        except:
          print("download failed")
          time.sleep(5)    
          continue
        try:
          image_download=imread("Test.jpg")
        except Exception, e:
          print("Bad Image")
          time.sleep(5)
          continue
        image_shape=image_download.shape
        if(len(image_shape)<3 or image_shape[2]!=3):
          continue
        image_download=image_download-mean(image_download)
        image_download[:, :, 0], image_download[:, :, 2] = image_download[:, :, 2], image_download[:, :, 0]
        temp=tf.image.resize_images(image_download,[227,227])
        image_download=sess2.run(temp)
        curr_data=np.concatenate((curr_data,image_download[np.newaxis,:]),axis=0)
        curr_target=np.concatenate((curr_target,[[load_data[i][1],load_data[i][2]]]),axis=0)

      # under=upper
      # print(curr_data.shape)
      # print(curr_target.shape)

      sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
                                        queue_input_target: curr_target})
    print(list_files[i]," read***********************")

  print("finished enqueueing")
  global ENQUEUE_LEFT
  ENQUEUE_LEFT=False
#********************************QUEUING ENDED************************

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)






svm_input=np.empty([0,output_categories])
svm_output=np.empty([0,2])

with tf.Session() as sess:
  init=tf.global_variables_initializer()
  sess.run(init)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)
  enqueue_thread = threading.Thread(target=enqueue, args=[sess])
  enqueue_thread.daemon=True
  enqueue_thread.start()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  counter=0

  while(sess.run("batch/fifo_queue_Size:0")!=0 or ENQUEUE_LEFT==True):
    counter=counter+batch_size
    curr_data_batch,curr_target_batch=sess.run([data_batch,target_batch])
    print(counter)
    svm_input=np.concatenate((svm_input,sess.run(fc8,feed_dict={x:curr_data_batch})),axis=0)
    svm_output=np.concatenate((svm_output,curr_target_batch),axis=0)
    # print(sess.run("batch/fifo_queue_Size:0"))


  sess.run(queue.close(cancel_pending_enqueues=True))
  coord.request_stop()
  coord.join(threads)
  sess.close()


views_per_photos=(svm_output[:,0]/svm_output[:,1])
svr_linear=SVR(kernel='linear',C=1e3)
svr_linear.fit(svm_input,views_per_photos)
path_svm_model='../Trained/'
file_svm_model='model.sav'
full_path_svm_model=path_svm_model+file_svm_model
joblib.dump(svr_linear,full_path_svm_model)
print("Saved Model")


# init = tf.initialize_all_variables()
# sess = tf.Session()

# direc_list=os.listdir(direc1)

# svm_input=np.empty([0,1000])
# direc_list_len=len(direc_list)
# direc_list_len=min(1,direc_list_len)
# for directories_iterator in range (direc_list_len):
#   directories=direc_list[directories_iterator]
#   image_list=os.listdir(direc1+directories)
#   image_arr=np.empty([0,227,227,3])

#   for images in image_list:
#     im1 = (imread(direc1+directories+"/"+images))
#     print(im1.shape)
#     # print(im1)
#     if (len(im1.shape)<3 or (im1.shape)[2]!=3):
#       continue
#     im1=im1-mean(im1)
#     im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#     temp=tf.image.resize_images(im1,[227,227])
#     # b=(tf.Session().run(temp))
#     sess.run(init)
#     im1=sess.run(temp)
#     print(image_arr.shape)
#     print(im1[np.newaxis,:].shape)
#     image_arr=np.concatenate((image_arr,im1[np.newaxis,:]),axis=0)
#     # t = time.time()
#   output = sess.run(fc8, feed_dict = {x:image_arr})
#   svm_input=np.concatenate((svm_input,output),axis=0)
#   print(directories)
#   print("*********************************************************")

# instances,not_required=svm_input.shape
# views=np.random.rand(instances,)*100
# photos=np.random.rand(instances,)*13

# # print(views.shape)

# # print instances

# views_per_photos=views/photos
# svr_linear=SVR(kernel='linear',C=1e3)
# svr_linear.fit(svm_input,views_per_photos)
# file_svm_model='model.sav'
# joblib.dump(svr_linear,file_svm_model)

# loaded_model=joblib.load(filename)
# result=loaded_model.score(X_test,Y_test)
# print(result)

