import tensorflow as tf
import numpy as np
import random
from scipy import misc,ndimage
import copy
import itertools


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

def preprocessing(image_batch, GTmatte_batch, GTBG_batch, GTFG_batch, image_size=320):
	'''
	input: 
		batch of image patch                  suppose unknown region = 128
		batch of ground truth alpha matte
		batch of BG
		batch of FG
	output:
		batch of 4 channel data and corresponding alpha,FG,BG
	'''

	g_mean = np.array([123.68, 116.779, 103.939]).reshape([1,1,3])
	image_batch_shape = image_batch.get_shape().as_list()
	batch_size = image_batch_shape[0]
	#generate trimap by random size (15~30) erosion and dilation
	kernel = [val for val in range(10,31)]
	trimap_batch = copy.deepcopy(GTmatte_batch)
	for i in range(batch_size):
		k_size = random.choice(kernel)
		trimap_batch[i][np.where((ndimage.grey_dilation(GTmatte_batch[i],size=(k_size,k_size)) - ndimage.grey_erosion(GTmatte_batch[i],size=(k_size,k_size)))!=0)] = 128
	trimap_batch = trimap_batch/255.0
	
	train_batch_pre = tf.concat([image_batch,trimap_batch,GTmatte_batch,GTBG_batch,GTFG_batch],3)
	train_batch = np.zeros([batch_size,image_batch_shape[1],image_batch_shape[2],11])
	for i in range(batch_size):
		crop_size = random.choice([320,480,640])
		flip = random.choice([0,1])
		# i_padding = center_padding(sess,tf.slice(train_batch_pre,[i,0,0,0],[1,image_batch_shape[1],image_batch_shape[2],11]))
		i_padding = tf.py_func(center_padding,[tf.slice(train_batch_pre,[i,0,0,0],[1,image_batch_shape[1],image_batch_shape[2],11])])
		i_UR_center = UR_center(i_padding)
		if crop_size == 320:
			h_start_index = i_UR_center[0] - 159
			w_start_index = i_UR_center[1] - 159
			tmp = i_padding[h_start_index:h_start_index+320, w_start_index:w_start_index+320, :]
			if flip:
				tmp = tmp[:,::-1,:]
			# tmp[:,:,:3] = tmp[:,:,:3] - mean
			train_batch[i,:,:,:] = tmp
		if crop_size == 480:
			h_start_index = i_UR_center[0] - 239
			w_start_index = i_UR_center[1] - 239
			tmp = i_padding[h_start_index:h_start_index+480, w_start_index:w_start_index+480, :]
			if flip:
				tmp = tmp[:,::-1,:]
			tmp = misc.imresize(tmp,[image_size,image_size,11])
			# tmp[:,:,:3] = tmp[:,:,:3] - mean
			train_batch[i,:,:,:] = tmp
		if crop_size == 640:
			h_start_index = i_UR_center[0] - 319
			w_start_index = i_UR_center[1] - 319
			tmp = i_padding[h_start_index:h_start_index+640, w_start_index:w_start_index+640, :]
			if flip:
				tmp = tmp[:,::-1,:]
			tmp = misc.imresize(tmp,[image_size,image_size,11])
			# tmp[:,:,:3] = tmp[:,:,:3] - mean
			train_batch[i,:,:,:] = tmp
	train_batch[:,:,:,:3] = train_batch[:,:,:,:3] - g_mean
	train_batch[:,:,:,4:6] = (train_batch[:,:,:,4:6][np.where(train_batch[:,:,:,4:6] == 255)] + 1)/256
	return train_batch[:,:,:,:4],train_batch[:,:,:,4],train_batch[:,:,:,5:8],train_batch[:,:,:,8:] #return input of CNN, and transformed GT alpha matte, GTBG,GTFG
	

def center_padding(image):
	'''
	image consists 11 channel (images, trimap, GT alpha matte, GTBG_batch, GTFG_batch)
	padding images to 2000*2000
	'''

	# image_shape = image.get_shape().as_list()
	# print(image_shape)
	# h_center = (image_shape[0]-1)//2
	# w_center = (image_shape[1]-1)//2
	# pad_image = np.zeros([2000,2000,11])
	# h_start_index = 999-h_center 
	# h_end_index = h_start_index + image_shape[0]
	# w_start_index = 999-w_center 
	# w_end_index = w_start_index + image_shape[1]
	# pad_image[h_start_index:h_end_index,w_start_index:w_end_index,:] = sess.run(image)
	# return pad_image

	image_shape = image.shape
	print(image_shape)
	h_center = (image_shape[0]-1)//2
	w_center = (image_shape[1]-1)//2
	pad_image = np.zeros([1300,1300,11])
	h_start_index = 649-h_center 
	h_end_index = h_start_index + image_shape[0]
	w_start_index = 649-w_center 
	w_end_index = w_start_index + image_shape[1]
	pad_image[h_start_index:h_end_index,w_start_index:w_end_index,:] = image
	return pad_image

def UR_center(image):
	'''
	image consists 5 channel (images, trimap, GT alpha matte)
	centered on unknown region
	'''
	trimap = image[:,:,3]
	UR = [[i,j] for i, j in itertools.product(range(trimap.shape[0]), range(trimap.shape[1])) if trimap[i,j] == 128]
#	return [int(i) for i in np.array(UR).mean(0)]
	return random.choice(UR)

def composition_RGB(BG,FG,p_matte):
	GB = tf.convert_to_tensor(BG)
	FG = tf.convert_to_tensor(FG)
	return p_matte * FG + (1 - p_matte) * BG
			
def global_mean():
	pass

def load_path(RGB,alpha,FG,BG):
	RGBs_path = os.listdir(dataset_RGB)
	RGBs_abspath = [os.path.join(dataset_RGB,RGB) for RGB in RGBs_path]
	alphas_abspath = [os.path.join(alpha,RGB.split('-')[0]) for RGB in RGBs_path]
	FGs_abspath = [os.path.join(FG,RGB.split('-')[0]) for RGB in RGBs_path]
	BGs_abspath = [os.path.join(BG,RGB.split('-')[0],RGB.split('-')[1]) for RGB in RGBs_path]
	return RGBs_abspath,alphas_abspath,FGs_abspath,BGs_abspath

def load_data(batch_RGB_paths,batch_alpha_paths,batch_FG_paths,batch_BG_paths):
	batch_RGBs = []
	for path in batch_RGB_paths:
		file_contents = tf.read_file(filename)
		image = tf.image.decode_png(file_contents)
		batch_RGBs.append(image)
	batch_RGBs = tf.stack(batch_RGBs)

	batch_alphas = []
	for path in batch_alpha_paths:
		file_contents = tf.read_file(filename)
		image = tf.image.decode_jpg(file_contents)
		batch_alphas.append(image)
	batch_alphas = tf.stack(batch_alphas)

	batch_FGs = []
	for path in batch_FG_paths:
		file_contents = tf.read_file(filename)
		image = tf.image.decode_jpg(file_contents)
		batch_FGs.append(image)
	batch_FGs = tf.stack(batch_FGs)

	batch_BGs = []
	for path in batch_BG_paths:
		file_contents = tf.read_file(filename)
		image = tf.image.decode_jpg(file_contents)
		batch_BGs.append(image)
	batch_BGs = tf.stack(batch_BGs)
	return batch_RGBs,batch_alphas,batch_FGs,batch_BGs
