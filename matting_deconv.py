import tensorflow as tf
import numpy as np
from matting import unpool,preprocessing,composition_RGB,load_path,load_data
import os

image_size = 320
batch_size = 100
max_epochs = 1000000

#pretrained_vgg_model_path
model_path = './vgg16_weights.npz'

dataset_RGB = './RGB'
dataset_alpha = './alpha'
dataset_FG = './FG'
dataset_BG = './BG'

paths_RGB,paths_alpha,paths_FG,paths_BG = load_path(dataset_RGB,dataset_alpha,dataset_FG,dataset_BG)
range_size = len(paths_RGB)
#range_size/batch_size has to be int
batchs_per_epoch = int(range_size/batch_size) 

index_queue = tf.train.range_input_producer(range_size, num_epochs=None,shuffle=True, seed=None, capacity=32)
index_dequeue_op = index_queue.dequeue_many(batch_size, 'index_dequeue')

image_batch = tf.placeholder(tf.float32, shape=(None,image_size,image_size,3))
GT_matte_batch = tf.placeholder(tf.float32, shape = (None,image_size,image_size,1))
GTBG_batch = tf.placeholder(tf.float32, shape = (None,image_size,image_size,3))
GTFG_batch = tf.placeholder(tf.float32, shape = (None,image_size,image_size,3))
en_parameters = []
if training:
    b_input, b_GTmatte ,b_GTBG, b_GTFG= preprocessing(image_batch,GT_matte_batch,GTBG_batch,GTFG_batch,image_size,)
else:
    preprocessing(image_batch)

batch_size = image_batch.get_shape().as_list()[0]

# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(b_input, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv1_2
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool1
pool1 = tf.nn.max_pool(conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

# conv2_1
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv2_2
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool2
pool2 = tf.nn.max_pool(conv2_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')

# conv3_1
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_2
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv3_3
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool3
pool3 = tf.nn.max_pool(conv3_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool3')

# conv4_1
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_2
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv4_3
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool4
pool4 = tf.nn.max_pool(conv4_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')

# conv5_1
with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_2
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# conv5_3
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name=scope)
    en_parameters += [kernel, biases]

# pool5
pool5 = tf.nn.max_pool(conv5_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')

#deconv6_1
with tf.name_scope('deconv6_1') as scope:
    outputs = tf.layers.conv2d_transpose(pool5, 512, [1, 1], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv6_1 = tf.nn.relu(outputs)

#deconv6_2
with tf.name_scope('deconv6_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv6_1, 512, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv6_2 = tf.nn.relu(outputs)


#deconv5_1
with tf.name_scope('deconv5_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv6_2, 512, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv5_1 = tf.nn.relu(outputs)

#deconv5_2
with tf.name_scope('deconv5_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv5_1, 512, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv5_2 = tf.nn.relu(outputs)

#deconv4_1
with tf.name_scope('deconv4_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv5_2, 256, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv4_1= tf.nn.relu(outputs)

#deconv4_2
with tf.name_scope('deconv4_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv4_1, 256, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv4_2= tf.nn.relu(outputs)

#deconv3_1
with tf.name_scope('deconv3_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv4_2, 128, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv3_1 = tf.nn.relu(outputs)

#deconv3_2
with tf.name_scope('deconv3_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv3_1, 128, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv3_2 = tf.nn.relu(outputs)

#deconv2_1
with tf.name_scope('deconv2_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv3_2, 64, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv2_1 = tf.nn.relu(outputs)

#deconv2_2
with tf.name_scope('deconv2_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv2_1, 64, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv2_2 = tf.nn.relu(outputs)

#deconv1
with tf.name_scope('deconv1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv2_2, 64, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv1 = tf.nn.relu(outputs)

#pred_alpha_matte
with tf.name_scope('pred_alpha') as scope:
    outputs = tf.layers.conv2d_transpose(deconv1, 1, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    pred_mattes = tf.nn.sigmoid(outputs)
    #get binary predicted mattes
    pred_mattes = tf.cast(tf.cast(pred_mattes + 0.5, tf.int32),tf.float32)

with tf.name_scope('loss') as scope:
    alpha_diff = tf.square(b_GTmatte - pred_mattes)
    b_RGB = []
    l_matte = tf.unstack(pred_mattes)
    for i in range(batch_size):
        b_RGB.append(composition_RGB(BG[i],FG[i],l_matte[i]))
    pred_RGB = tf.stack(b_RGB)
    c_diff = tf.square(pred_RGB - tf.convert_to_tensor(b_input))
    wl = np.ones_like(b_GTmatte).fill(0.5)
    wl[np.where(b_GTmatte = 0.5)] = 1.5

    total_loss = tf.reduce_sum(tf.convert_to_tensor(wl) * alpha_diff + tf.convert_to_tensor(2-wl) * c_diff) / batch_size
    train_op = tf.train.AdamOptimizer(learning_rate = 1e-5).minimize(total_loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	batch_num = 0
	epoch_num = 0
	#initialize all parameters in vgg16
	weights = np.load(model_path)
	keys = sorted(weights.keys())
	for i, k in enumerate(keys):
		if i == 26:
			break
		if k == 'conv1_1_W':  
			sess.run(en_parameters[i].assign(np.concatenate([weights[k],np.zeros([3,3,1,64])],axis = 2)))
		else:
			sess.run(en_parameters[i].assign(weights[k]))

	#load train data
	while epoch_num < max_epochs:
		print('epoch %d' % epoch_num)	
		while batch_num < batchs_per_epoch:
			print('batch %d' % batch_num)
			batch_index = sess.run(index_dequeue_op)			
		    batch_RGB_paths = np.array(paths_RGB)[index_epoch]
		    batch_alpha_paths = np.array(paths_alpha)[index_epoch]
		    batch_FG_paths = np.array(paths_FG)[index_epoch]
		    batch_BG_paths = np.array(paths_BG)[index_epoch]
		    
		    batch_RGBs,batch_alphas,batch_FGs,batch_BGs = load_data(batch_RGB_paths,batch_alpha_paths,batch_FG_paths,batch_BG_paths)

			feed_dict = {image_batch:batch_RGBs,GT_matte_batch:batch_alphas,GTBG_batch:batch_BGs,GTFG_batch:batch_FGs}
			_,loss = sess.run([train_op,total_loss],feed_dict = feed_dict)
			print('loss %f \n' %loss)
			batch_num += 1
		epoch_num += 1


