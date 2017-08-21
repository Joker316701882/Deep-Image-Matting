import tensorflow as tf
import numpy as np
import random
from scipy import misc,ndimage
import copy
import itertools
import os
from sys import getrefcount
import gc

input_image_size = 650

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

	g_mean = np.array(([126.8898,120.2431,112.1959])).reshape([1,1,3])
	image_batch_shape = image_batch.shape
	batch_size = image_batch_shape[0]
	#generate trimap by random size (15~30) erosion and dilation

#	trimap_batch = copy.deepcopy(GTmatte_batch)
	trimap_batch = np.copy(GTmatte_batch)
	#trimap_batch = copy.deepcopy(GTmatte_batch)
	trimap_batch = generate_trimap(trimap_batch,GTmatte_batch,batch_size)

	train_batch_pre = np.concatenate([image_batch,trimap_batch,GTmatte_batch,GTBG_batch,GTFG_batch],3)
	train_batch = np.zeros([batch_size,image_size,image_size,11])
	for i in range(batch_size):
		#print('%dth image is under processing...'%i)
		crop_size = random.choice([320,480,640])
		flip = random.choice([0,1])
		# i_padding = center_padding(sess,tf.slice(train_batch_pre,[i,0,0,0],[1,image_batch_shape[1],image_batch_shape[2],11]))
		i_padding = center_padding(train_batch_pre[i])
		#i_UR_center = UR_center(i_padding)
		i_UR_center = UR_center(i_padding)
		if crop_size == 320:
			h_start_index = i_UR_center[0] - 159
			w_start_index = i_UR_center[1] - 159
			tmp = i_padding[h_start_index:h_start_index+320, w_start_index:w_start_index+320, :]
			if flip:
				tmp = tmp[:,::-1,:]
			# tmp[:,:,:3] = tmp[:,:,:3] - mean
			tmp[:,:,3:5] = tmp[:,:,3:5] / 255.0
			tmp[:,:,:3] -= g_mean
			train_batch[i,:,:,:] = tmp
		if crop_size == 480:
			h_start_index = i_UR_center[0] - 239
			w_start_index = i_UR_center[1] - 239
			tmp = i_padding[h_start_index:h_start_index+480, w_start_index:w_start_index+480, :]
			if flip:
				tmp = tmp[:,::-1,:]
			tmp1 = np.zeros([image_size,image_size,11])
			tmp1[:,:,:3] = misc.imresize(tmp[:,:,:3],[image_size,image_size,3]) - g_mean
			tmp1[:,:,3] = misc.imresize(tmp[:,:,3],[image_size,image_size],interp = 'nearest') / 255.0
			tmp1[:,:,4] = binarilize_alpha(misc.imresize(tmp[:,:,4],[image_size,image_size]),60) / 255.0
			tmp1[:,:,5:8] = misc.imresize(tmp[:,:,5:8],[image_size,image_size,3])
			tmp1[:,:,8:] = misc.imresize(tmp[:,:,8:],[image_size,image_size,3])
			train_batch[i,:,:,:] = tmp1

		if crop_size == 640:
			h_start_index = i_UR_center[0] - 319
			w_start_index = i_UR_center[1] - 319
			tmp = i_padding[h_start_index:h_start_index+640, w_start_index:w_start_index+640, :]
			if flip:
				tmp = tmp[:,::-1,:]
			tmp1 = np.zeros([image_size,image_size,11])
			tmp1[:,:,:3] = misc.imresize(tmp[:,:,:3],[image_size,image_size,3]) - g_mean
			tmp1[:,:,3] = misc.imresize(tmp[:,:,3],[image_size,image_size],interp = 'nearest') / 255.0
			tmp1[:,:,4] = binarilize_alpha(misc.imresize(tmp[:,:,4],[image_size,image_size]),60) / 255.0
			tmp1[:,:,5:8] = misc.imresize(tmp[:,:,5:8],[image_size,image_size,3])
			tmp1[:,:,8:] = misc.imresize(tmp[:,:,8:],[image_size,image_size,3])
			train_batch[i,:,:,:] = tmp1
	gc.collect()
	# print('tmp %d' %getrefcount(tmp))
	# print('tmp %d' %getrefcount(tmp1))
	# print('tmp %d' %getrefcount(train_batch_pre))
	# print('tmp %d' %getrefcount(trimap_batch))
	train_batch = train_batch.astype(np.float32)

	return train_batch[:,:,:,:3],np.expand_dims(train_batch[:,:,:,3],3),np.expand_dims(train_batch[:,:,:,4],3),train_batch[:,:,:,5:8],train_batch[:,:,:,8:] #return input of CNN, and transformed GT alpha matte, GTBG,GTFG
	
def binarilize_alpha(alpha, threshold):
	alpha[np.where(alpha<=threshold)] = 0
	alpha[np.where(alpha>threshold)] = 255
	return alpha

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
	h_center = (image_shape[0]-1)//2
	w_center = (image_shape[1]-1)//2
	pad_image = np.zeros([1300,1300,11])
	h_start_index = 649-h_center 
	h_end_index = h_start_index + image_shape[0]
	w_start_index = 649-w_center 
	w_end_index = w_start_index + image_shape[1]
	pad_image[h_start_index:h_end_index,w_start_index:w_end_index,:] = image
	return pad_image

def show(image):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i][j]!=0 and image[i][j] !=1 and image[i][j] !=0.5: 
				print([i,j])

def UR_center(image):
	'''
	image consists 5 channel (images, trimap, GT alpha matte)
	centered on unknown region
	'''
	trimap = image[:,:,3]
#	UR = [[i,j] for i, j in itertools.product(range(trimap.shape[0]), range(trimap.shape[1])) if trimap[i,j] == 128]
	target = np.where(trimap==127.5)
	# show(trimap)
	index = random.choice([i for i in range(len(target[0]))])
	return  np.array(target)[:,index][:2]
#	return [int(i) for i in np.array(UR).mean(0)]
#	return random.choice(UR)

	# trimap = tf.convert_to_tensor(image[:,:,3])
	# condition = tf.equal(trimap,128)
	# indices = tf.where(condition)
	# return random.choice(indices)

def composition_RGB(BG,FG,p_matte):
	GB = tf.convert_to_tensor(BG)
	FG = tf.convert_to_tensor(FG)
	return p_matte * FG + (1 - p_matte) * BG
			
def global_mean(RGB_folder):
	RGBs = os.listdir(RGB_folder)
	num = len(RGBs)
	ite = num // 100
	sum_tmp = []
	print(ite)
	for i in range(ite):
		print(i)
		batch_tmp = np.zeros([input_image_size,input_image_size,3])
		RGBs_tmp = [os.path.join(RGB_folder,RGB_path) for RGB_path in RGBs[i*100:(i+1)*100]]
		for RGB in RGBs_tmp:
			batch_tmp += misc.imread(RGB)
		sum_tmp.append(batch_tmp.sum(axis = 0).sum(axis = 0) / (input_image_size*input_image_size*100))
	return np.array(sum_tmp).mean(axis = 0)

def load_path(dataset_RGB,alpha,FG,BG):
	RGBs_path = os.listdir(dataset_RGB)
	RGBs_abspath = [os.path.join(dataset_RGB,RGB) for RGB in RGBs_path]
	alphas_abspath = [os.path.join(alpha,RGB.split('-')[0]+'.png') for RGB in RGBs_path]
	FGs_abspath = [os.path.join(FG,RGB.split('-')[0]+'.png') for RGB in RGBs_path]
	BGs_abspath = [os.path.join(BG,RGB.split('-')[0],RGB.split('-')[1][:-3]+'jpg') for RGB in RGBs_path]
	return RGBs_abspath,alphas_abspath,FGs_abspath,BGs_abspath

def load_data(batch_RGB_paths,batch_alpha_paths,batch_FG_paths,batch_BG_paths):
	batch_RGBs = []

	for path in batch_RGB_paths:
		file_contents = tf.read_file(tf.convert_to_tensor(path))
		image = tf.image.decode_png(file_contents)
		batch_RGBs.append(image)
	batch_RGBs = tf.stack(batch_RGBs)

	batch_alphas = []
	for path in batch_alpha_paths:
		file_contents = tf.read_file(tf.convert_to_tensor(path))
		image = tf.image.decode_png(file_contents)
		batch_alphas.append(image)
	batch_alphas = tf.stack(batch_alphas)

	batch_FGs = []
	for path in batch_FG_paths:
		file_contents = tf.read_file(tf.convert_to_tensor(path))
		image = tf.image.decode_png(file_contents)
		batch_FGs.append(image)
	batch_FGs = tf.stack(batch_FGs)

	batch_BGs = []
	for path in batch_BG_paths:
		file_contents = tf.read_file(tf.convert_to_tensor(path))
		image = tf.image.decode_jpeg(file_contents)
		batch_BGs.append(image)
	batch_BGs = tf.stack(batch_BGs)
	return batch_RGBs,batch_alphas,batch_FGs,batch_BGs

def generate_trimap(trimap_batch,GTmatte_batch,batch_size):
	kernel = [val for val in range(15,31)]
	for i in range(batch_size):
		k_size = random.choice(kernel)
		trimap_batch[i][np.where((ndimage.grey_dilation(GTmatte_batch[i][:,:,0],size=(k_size,k_size)) - ndimage.grey_erosion(GTmatte_batch[i][:,:,0],size=(k_size,k_size)))!=0)] = 127.5
	return trimap_batch


