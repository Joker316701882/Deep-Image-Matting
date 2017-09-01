import tensorflow as tf
import numpy as np
import random
from scipy import misc,ndimage
import copy
import itertools
import os
from sys import getrefcount
import gc

trimap_kernel = [val for val in range(20,35)]
g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])

def UR_center(image):
	'''
	image consists 5 channel (images, trimap, GT alpha matte)
	centered on unknown region
	'''
	trimap = image[:,:,3]
	target = np.where(trimap==128)
	index = random.choice([i for i in range(len(target[0]))])
	return  np.array(target)[:,index][:2]

# def composition_RGB(BG,FG,p_matte):

# 	return p_matte * FG + (1 - p_matte) * BG

# def global_mean(RGB_folder):
# 	RGBs = os.listdir(RGB_folder)
# 	num = len(RGBs)
# 	ite = num // 100
# 	sum_tmp = []
# 	print(ite)
# 	for i in range(ite):
# 		print(i)
# 		batch_tmp = np.zeros([320,320,3])
# 		RGBs_tmp = [os.path.join(RGB_folder,RGB_path) for RGB_path in RGBs[i*100:(i+1)*100]]
# 		for RGB in RGBs_tmp:
# 			batch_tmp += misc.imread(RGB)
# 		sum_tmp.append(batch_tmp.sum(axis = 0).sum(axis = 0) / (320*320*100))
# 	return np.array(sum_tmp).mean(axis = 0)

def load_path(dataset_RGB,alpha,FG,BG):
	RGBs_path = os.listdir(dataset_RGB)
	RGBs_abspath = [os.path.join(dataset_RGB,RGB) for RGB in RGBs_path]
	alphas_abspath = [os.path.join(alpha,RGB.split('-')[0],RGB.split('-')[1]) for RGB in RGBs_path]
	FGs_abspath = [os.path.join(FG,RGB.split('-')[0],RGB.split('-')[1]) for RGB in RGBs_path]
	BGs_abspath = [os.path.join(BG,RGB.split('-')[0],RGB.split('-')[1][:-3]+'jpg') for RGB in RGBs_path]
	return np.array(RGBs_abspath),np.array(alphas_abspath),np.array(FGs_abspath),np.array(BGs_abspath)

def load_data(batch_RGB_paths,batch_alpha_paths,batch_FG_paths,batch_BG_paths):
	
	batch_size = batch_RGB_paths.shape[0]
	train_batch = []
	images_without_mean_reduction = []
	for i in range(batch_size):
		comp_RGB = misc.imread(batch_RGB_paths[i]).astype(np.float32)		
		
		alpha = misc.imread(batch_alpha_paths[i],'L').astype(np.float32)

		FG = misc.imread(batch_FG_paths[i]).astype(np.float32)

		BG = misc.imread(batch_BG_paths[i]).astype(np.float32)
		
		batch_i,raw_RGB = preprocessing_single(comp_RGB, alpha, BG, FG,i)	
		train_batch.append(batch_i)
		images_without_mean_reduction.append(raw_RGB)
	train_batch = np.stack(train_batch)
	return train_batch[:,:,:,:3],np.expand_dims(train_batch[:,:,:,3],3),np.expand_dims(train_batch[:,:,:,4],3),train_batch[:,:,:,5:8],train_batch[:,:,:,8:],images_without_mean_reduction

def generate_trimap(trimap,alpha):

	k_size = random.choice(trimap_kernel)
	trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - ndimage.grey_erosion(alpha[:,:,0],size=(k_size,k_size)))!=0)] = 128
	return trimap

def preprocessing_single(comp_RGB, alpha, BG, FG,i,image_size=320):

	alpha = np.expand_dims(alpha,2)
	trimap = np.copy(alpha)
	#trimap_batch = copy.deepcopy(GTmatte_batch)
	trimap = generate_trimap(trimap,alpha)

	train_pre = np.concatenate([comp_RGB,trimap,alpha,BG,FG],2)
	train_data = np.zeros([image_size,image_size,11])
	crop_size = random.choice([320,480,620])
	flip = random.choice([0,1])
	i_UR_center = UR_center(train_pre)


	if crop_size == 320:
		h_start_index = i_UR_center[0] - 159
		w_start_index = i_UR_center[1] - 159
		tmp = train_pre[h_start_index:h_start_index+320, w_start_index:w_start_index+320, :]
		if flip:
			tmp = tmp[:,::-1,:]
		# tmp[:,:,:3] = tmp[:,:,:3] - mean
		tmp[:,:,3:5] = tmp[:,:,3:5] / 256.0
		raw_RGB = tmp[:,:,:3]
		tmp[:,:,:3] -= g_mean
		train_data = tmp

	if crop_size == 480:
		h_start_index = i_UR_center[0] - 239
		w_start_index = i_UR_center[1] - 239
		tmp = train_pre[h_start_index:h_start_index+480, w_start_index:w_start_index+480, :]
		if flip:
			tmp = tmp[:,::-1,:]
		tmp1 = np.zeros([image_size,image_size,11]).astype(np.float32)
		raw_RGB = misc.imresize(tmp[:,:,:3],[image_size,image_size,3])
		tmp1[:,:,:3] = raw_RGB - g_mean
		tmp1[:,:,3] = misc.imresize(tmp[:,:,3],[image_size,image_size],interp = 'nearest') / 256.0
		tmp1[:,:,4] = misc.imresize(tmp[:,:,4],[image_size,image_size]) / 256.0
		tmp1[:,:,5:8] = misc.imresize(tmp[:,:,5:8],[image_size,image_size,3])
		tmp1[:,:,8:] = misc.imresize(tmp[:,:,8:],[image_size,image_size,3])
		train_data = tmp1

	if crop_size == 620:
		h_start_index = i_UR_center[0] - 309
		#boundary security
		if h_start_index<0:
			h_start_index = 0
		w_start_index = i_UR_center[1] - 309
		if w_start_index<0:
			w_start_index = 0
		tmp = train_pre[h_start_index:h_start_index+620, w_start_index:w_start_index+620, :]
		if flip:
			tmp = tmp[:,::-1,:]
		tmp1 = np.zeros([image_size,image_size,11]).astype(np.float32)
		raw_RGB = misc.imresize(tmp[:,:,:3],[image_size,image_size,3])
		tmp1[:,:,:3] = raw_RGB - g_mean
		tmp1[:,:,3] = misc.imresize(tmp[:,:,3],[image_size,image_size],interp = 'nearest') / 256.0
		tmp1[:,:,4] = misc.imresize(tmp[:,:,4],[image_size,image_size]) / 256.0
		tmp1[:,:,5:8] = misc.imresize(tmp[:,:,5:8],[image_size,image_size,3])
		tmp1[:,:,8:] = misc.imresize(tmp[:,:,8:],[image_size,image_size,3])
		train_data = tmp1
	train_data = train_data.astype(np.float32)
	return train_data,raw_RGB

def load_test_data(test_alpha):
	rgb_path = os.path.join(test_alpha,'rgb')
	trimap_path = os.path.join(test_alpha,'trimap')
	alpha_path = os.path.join(test_alpha,'alpha')	
	images = os.listdir(trimap_path)
	test_num = len(images)
	all_shape = []
	rgb_batch = []
	tri_batch = []
	alp_batch = []
	for i in range(test_num):
		rgb = misc.imread(os.path.join(rgb_path,images[i]))
		trimap = misc.imread(os.path.join(trimap_path,images[i]),'L')
		alpha = misc.imread(os.path.join(alpha_path,images[i]),'L')/256.0
		all_shape.append(trimap.shape)
		rgb_batch.append(misc.imresize(rgb,[320,320,3])-g_mean)
		trimap = misc.imresize(trimap,[320,320],interp = 'nearest').astype(np.float32)/256.0

		tri_batch.append(np.expand_dims(trimap,2))
		alp_batch.append(np.expand_dims(alpha,2))
	return np.array(rgb_batch),np.array(tri_batch),np.array(alp_batch),all_shape,images

