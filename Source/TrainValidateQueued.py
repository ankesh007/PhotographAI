#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import csv
import os
import threading

# *********Setting folder names***********************
DatasetFolder='Dataset/Mode/'
model_dir="ThursTest/"
checkpoint_folder="Trained/"+model_dir
checkpoint_name="model.ckpt"

#***********Setting DNN parametes*********************
input_parameters=95
output_parameters=1
hidden_layer1_nodes=2*input_parameters
# hidden_layer2_nodes=2*input_parameters
learning_rate=0.001
train=30000


checkpoint_iterations=10 
log_every=1
lamda=0.0 #for regularization
max_checkpoint_to_keep=5 #number of last checkpoints to be kept
batch_size=1000 #training batch_size
placeholder_limit=batch_size 
queue_capacity=batch_size
train_batch_limit=batch_size
compute_validation_loss_after=10



queue_input_data = tf.placeholder(tf.float32, shape=[None,input_parameters])
queue_input_target = tf.placeholder(tf.float32, shape=[None, output_parameters])

queue = tf.FIFOQueue(capacity=queue_capacity, dtypes=[tf.float32, tf.float32], shapes=[[input_parameters], [output_parameters]])

enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
dequeue_op = queue.dequeue()
list_files=os.listdir(DatasetFolder)
print(list_files)
list_files_len=len(list_files)
# tensorflow recommendation:
# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=train_batch_limit)
# use this to shuffle batches:
# data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)
ToEnqueue=True
def enqueue(sess):
	""" Iterates over our data puts small junks into our queue."""
	while(ToEnqueue):
		for i in range(list_files_len-1):
			load_data = np.genfromtxt(DatasetFolder+list_files[i], delimiter=',')+2
			max_val=np.amax(load_data,axis=0)
			global train
			load_data=np.divide(load_data,max_val)
			raw_target = load_data[:,38][:,np.newaxis]
			raw_data = load_data[:,0:38]
			raw_data = np.concatenate((raw_data,load_data[:,39:input_parameters+1]),axis=1)


			under = 0
			FLAG=True

			max = len(raw_data)
			train=max
			print("Reading File",list_files[i],"started at index",i)
			while True:
				# print("starting to write into queue")
				upper = under + queue_capacity
				# print("try to enqueue ", under, " to ", upper)
				if (upper <= max):
					curr_data = raw_data[under:upper,:]
					curr_target = raw_target[under:upper,:]
					under = upper
				else:
					rest = upper - max
					curr_data = raw_data[under:max]
					curr_target = raw_target[under:max]
					FLAG=False

				sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
			                                    queue_input_target: curr_target})
				# print("added to the queue")
				if(FLAG==False):
					# print("Reading File",list_files[i],"completed")
					break
	print("finished enqueueing")

# start the threads for our FIFOQueue and batch
# sess = tf.Session()


def add_layer(in_nodes,out_nodes,inputs,n_layer,activation_function=None):

	layer_name="layer"+str(n_layer)
		
	with tf.name_scope(layer_name):
		weights=tf.Variable(tf.random_normal([in_nodes,out_nodes]))
		biases=tf.Variable(tf.random_normal([1,out_nodes]))
		output=tf.add(tf.matmul(inputs,weights),biases)
		regularizer=tf.nn.l2_loss(weights)
	# print output

		if activation_function is None:
			return [output,regularizer]

		else:
			return [activation_function(output),regularizer]

def get_test_inputs():
	load_data = np.genfromtxt(DatasetFolder+list_files[list_files_len-1], delimiter=',')+2
	max_val=np.amax(load_data,axis=0)
	load_data=np.divide(load_data,max_val)
	raw_target = load_data[:,38][:,np.newaxis]
	raw_data = load_data[:,0:38]
	raw_data = np.concatenate((raw_data,load_data[:,39:input_parameters+1]),axis=1)
	return raw_data, raw_target

x_test,y_test=get_test_inputs()



xs=tf.placeholder(tf.float32,[None,input_parameters])
ys=tf.placeholder(tf.float32,[None,output_parameters])
samples=tf.placeholder(tf.float32)

[layer1,regularizer_loss1]=add_layer(input_parameters,hidden_layer1_nodes,xs,1,activation_function=tf.nn.relu)
[prediction,regularizer_loss2]=add_layer(hidden_layer1_nodes,output_parameters,layer1,2)
regularizers=regularizer_loss1+regularizer_loss2
# layer2=add_layer(20,10,layer1,activation_function=tf.nn.relu)

global_step=tf.Variable(initial_value=0)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
loss=tf.reduce_mean(loss+lamda*regularizers)
train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)




saver=tf.train.Saver(max_to_keep=max_checkpoint_to_keep)
with tf.Session() as sess:


	if not (os.path.exists(checkpoint_folder)):
		os.makedirs(checkpoint_folder)
		# init=tf.initialize_all_variables()
		init=tf.global_variables_initializer()
		sess.run(init)

	elif(tf.train.latest_checkpoint(checkpoint_folder) is None):
		# init=tf.initialize_all_variables()
		init=tf.global_variables_initializer()
		sess.run(init)

	else:
		saver.restore(sess,tf.train.latest_checkpoint(checkpoint_folder))

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	enqueue_thread = threading.Thread(target=enqueue, args=[sess])
	enqueue_thread.daemon=True
	enqueue_thread.start()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	


	first_eval=True
	cost=0
	counter=0

	while(True):

		counter=counter+1
		for i in range(list_files_len-1):
			j=0

			# One file completely trained in batches
			while(j<train):
				maxy=min(train,j+batch_size)
				curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch])
				# sess.run(train_step,feed_dict={xs:x_train[j:maxy,:],ys:y_train[j:maxy,:]})
				sess.run(train_step,feed_dict={xs:curr_data_batch,ys:curr_target_batch})
				j=j+batch_size

			if((i+1)%log_every==0):

				print("loss: ",sess.run(loss,feed_dict={xs:curr_data_batch,ys:curr_target_batch})," steps:",sess.run(global_step))

			print("Saving Checkpoint",sess.run(global_step))
			saver.save(sess,checkpoint_folder+checkpoint_name,global_step=sess.run(global_step))

		lossComp=sess.run(loss,feed_dict={xs:x_test,ys:y_test})
		print("**********************Computing Loss on Test********************************")
		print("loss: ",lossComp," prevLoss: ",cost," global steps: ",sess.run(global_step))

			
		if(counter%compute_validation_loss_after==0):

			if(first_eval):
				first_eval=False
				cost=lossComp

			elif(lossComp<cost):
				cost=lossComp

			else:
				# global ToEnqueue
				print("Ended")
				ToEnqueue=False
				break

	sess.run(queue.close(cancel_pending_enqueues=True))
	coord.request_stop()
	coord.join(threads)
	sess.close()