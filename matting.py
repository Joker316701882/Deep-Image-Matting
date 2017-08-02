import tensorflow as tf
import numpy as np
import random
from scipy import misc

def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    
	with tf.variable_scope(scope):
		input_shape = pool.get_shape().as_list()
		output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

		flat_input_size = np.prod(input_shape)
		flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

		pool_ = tf.reshape(pool, [flat_input_size])
		batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
		b = tf.ones_like(ind) * batch_range
		b = tf.reshape(b, [flat_input_size, 1])
		ind_ = tf.reshape(ind, [flat_input_size, 1])
		ind_ = tf.concat([b, ind_], 1)

		ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
		ret = tf.reshape(ret, output_shape)
		return ret

def preprocessing(sess,image_batch, trimap_batch, GTmatte_batch, batch_size = 128, image_size=320):
	'''
	input: 
		batch of image patch       
		batch of corresponding trimap             suppose unknown region = 0.5
		batch of ground truth alpha matte
	output:
		batch of 4 channel data and corresponding alpha matte
	'''

	g_mean = np.array([123.68, 116.779, 103.939]).reshape([1,1,3])
	image_batch_shape = image_batch.get_shape().as_list()

	train_batch_pre = tf.concat([image_batch,trimap_batch,GTmatte_batch],3)
	train_batch = np.ones([batch_size,image_size,image_size,5])
	for i in range(batch_size):
		crop_size = random.choice([320,480,640])
		flip = random.choice([0,1])
		i_padding = center_padding(tf.slice(train_batch_pre,[i,0,0,0],[1,image_batch_shape[1],image_batch_shape[2],5]))
		i_UR_center = UR_center(i_padding)
		if crop_size == 320:
			h_start_index = i_UR_center[0] - 159
			w_start_index = i_UR_center[1] - 159
			tmp = i_padding[h_start_index:h_start_index+320, w_start_index:w_start_index+320, 5]
			if flip:
				tmp = tmp[:,::-1,:]
			tmp[:,:,:3] = tmp[:,:,:3] - mean
			train_batch[i,:,:,:] = tmp.reshape([image_size,image_size,5])
		if crop_size == 480:
			h_start_index = i_UR_center[0] - 239
			w_start_index = i_UR_center[1] - 239
			tmp = i_padding[h_start_index:h_start_index+480, w_start_index:w_start_index+480, 5]
			if flip:
				tmp = tmp[:,::-1,:]
			h1 = misc.imresize(tmp[:,:,:3],[image_size,image_size,3]) - mean
			h2 = misc.imresize(tmp[:,:,3:],[image_size,image_size,2],interp = 'nearest')
			train_batch[i,:,:,:] = np.concatenate([h1,h2],2)
		if crop_size == 640:
			h_start_index = i_UR_center[0] - 319
			w_start_index = i_UR_center[1] - 319
			tmp = i_padding[h_start_index:h_start_index+640, w_start_index:w_start_index+640, 5]
			if flip:
				tmp = tmp[:,::-1,:]
			h1 = misc.imresize(tmp[:,:,:3],[image_size,image_size,3]) - mean
			h2 = misc.imresize(tmp[:,:,3:],[image_size,image_size,2],interp = 'nearest')
			train_batch[i,:,:,:] = np.concatenate([h1,h2],2)
	# else:
	# 	#resize,去均值。 batch?
	# 	#假设是batch
	# 	test_batch_size = image_batch_shape[0]
	# 	test_batch = np.ones([test_batch_size,image_size,image_size,5])
	# 	test_batch_pre = tf.concat([image_batch,trimap_batch,GTmatte_batch],3)
	# 	for i in range(test_batch):
	# 		tf.slice(train_batch_pre,[i,0,0,0],[1,image_batch_shape[1],image_batch_shape[2],5])




def center_padding(sess,image):
	'''
	image consists 5 channel (images, trimap, GT alpha matte)
	padding images to 2000*2000
	'''

	image_shape = image.get_shape().as_list()
	print(image_shape)
	h_center = (image_shape[0]-1)//2
	w_center = (image_shape[1]-1)//2
	pad_image = np.zeros([2000,2000,5])
	h_start_index = 999-h_center 
	h_end_index = h_start_index + image_shape[0]
	w_start_index = 999-w_center 
	w_end_index = w_start_index + image_shape[1]
	pad_image[h_start_index:h_end_index,w_start_index:w_end_index,:] = sess.run(image)
	return pad_image

def UR_center(image):
	'''
	image consists 5 channel (images, trimap, GT alpha matte)
	calculate center of unknown region
	'''
	UR = []
	trimap = image[:,:,3]
	for i in range(len(trimap)):
		for j in range(len(trimap[i])):
			if trimap[i,j]!=0 and trimap[i,j]!=1:
				UR.append([i,j])
	
	return [int(i) for i in np.array(UR).mean(0)]



def center_crop():
	pass

def global_mean():
	pass

def generate_trimaps():
	pass

with tf.Session() as sess:
	image = tf.Variable(tf.truncated_normal([10,10,5],dtype = tf.float32))
	sess.run(tf.global_variables_initializer())
	new_image = center_padding(sess,image)
	print(UR_center(new_image))