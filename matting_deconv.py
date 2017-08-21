import tensorflow as tf
import numpy as np
from matting import unpool,preprocessing,composition_RGB,load_path,load_data
import os
from scipy import misc

image_size = 320
input_image_size = 650
batch_size = 25
max_epochs = 1000000

#pretrained_vgg_model_path
model_path = './vgg16_weights.npz'
log_dir = './tensor_log'
dataset_RGB = '/data/gezheng/data-matting/comp_RGB'
dataset_alpha = './alpha_final'
dataset_FG = './FG_final'
dataset_BG = '/data/gezheng/data-matting/BG'

paths_RGB,paths_alpha,paths_FG,paths_BG = load_path(dataset_RGB,dataset_alpha,dataset_FG,dataset_BG)

range_size = len(paths_RGB)
#range_size/batch_size has to be int
batchs_per_epoch = int(range_size/batch_size) 

index_queue = tf.train.range_input_producer(range_size, num_epochs=None,shuffle=True, seed=None, capacity=32)
index_dequeue_op = index_queue.dequeue_many(batch_size, 'index_dequeue')

image_batch = tf.placeholder(tf.float32, shape=(batch_size,input_image_size,input_image_size,3))
GT_matte_batch = tf.placeholder(tf.float32, shape = (batch_size,input_image_size,input_image_size,1))
GTBG_batch = tf.placeholder(tf.float32, shape = (batch_size,input_image_size,input_image_size,3))
GTFG_batch = tf.placeholder(tf.float32, shape = (batch_size,input_image_size,input_image_size,3))
is_train = tf.placeholder(tf.bool, name = 'is_train')
en_parameters = []
#if training:
#b_input, b_GTmatte ,b_GTBG, b_GTFG = preprocessing(image_batch,GT_matte_batch,GTBG_batch,GTFG_batch,image_size)
#[b_input, b_GTmatte ,b_GTBG, b_GTFG] = tf.py_func(preprocessing,[image_batch,GT_matte_batch,GTBG_batch,GTFG_batch,image_size],tf.float32)
if is_train:
    b_RGB,b_trimap, b_GTmatte ,b_GTBG, b_GTFG = tf.py_func(preprocessing,[image_batch,GT_matte_batch,GTBG_batch,GTFG_batch,image_size],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
else:
    pass

tf.summary.image('GT_alpha',b_GTmatte,max_outputs = 5)
tf.summary.image('trimap',b_trimap,max_outputs = 5)

b_RGB.set_shape([batch_size,image_size,image_size,3])
b_trimap.set_shape([batch_size,image_size,image_size,1])
b_input = tf.concat([b_RGB,b_trimap],3)
b_GTmatte.set_shape([batch_size,image_size,image_size,1])
b_GTBG.set_shape([batch_size,image_size,image_size,3])
b_GTFG.set_shape([batch_size,image_size,image_size,3])

b_RGB = tf.identity(b_RGB,name = 'b_RGB')
b_trimap = tf.identity(b_trimap,name = 'b_trimap')
b_GTmatte = tf.identity(b_GTmatte,name = 'b_GTmatte')
b_GTBG = tf.identity(b_GTBG,name = 'b_GTBG')
b_GTFG = tf.identity(b_GTFG,name = 'b_GTFG')
#else:
 #   preprocessing(image_batch)

batch_size = image_batch.get_shape().as_list()[0]
# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    tf.summary.histogram('W1_1',kernel)
    # with tf.control_dependencies()
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
    tf.summary.histogram('W1_2',kernel)
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
    tf.summary.histogram('W2_1',kernel)
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
    tf.summary.histogram('W2_2',kernel)
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

training = True

#deconv6_1
with tf.variable_scope('deconv6_1') as scope:
    outputs = tf.layers.conv2d_transpose(pool5, 512, [1, 1], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv6_1 = tf.nn.relu(outputs)
    deconv6_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))
#deconv6_2
with tf.variable_scope('deconv6_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv6_1, 512, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv6_2 = tf.nn.relu(outputs)
    deconv6_2 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv5_1
with tf.variable_scope('deconv5_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv6_2, 512, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv5_1 = tf.nn.relu(outputs)
    deconv5_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv5_2
with tf.variable_scope('deconv5_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv5_1, 512, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv5_2 = tf.nn.relu(outputs)
    deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv4_1
with tf.variable_scope('deconv4_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv5_2, 256, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv4_1= tf.nn.relu(outputs)
    deconv4_1 =tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv4_2
with tf.variable_scope('deconv4_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv4_1, 256, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv4_2= tf.nn.relu(outputs)
    deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv3_1
with tf.variable_scope('deconv3_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv4_2, 128, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv3_1 = tf.nn.relu(outputs)
    deconv3_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv3_2
with tf.variable_scope('deconv3_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv3_1, 128, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv3_2 = tf.nn.relu(outputs)
    deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv2_1
with tf.variable_scope('deconv2_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv3_2, 64, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv2_1 = tf.nn.relu(outputs)
    deconv2_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv2_2
with tf.variable_scope('deconv2_2') as scope:
    outputs = tf.layers.conv2d_transpose(deconv2_1, 64, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
#    deconv2_2 = tf.nn.relu(outputs)
    deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv1
with tf.variable_scope('deconv1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv2_2, 64, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #deconv1 = tf.nn.relu(outputs)
    deconv1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#pred_alpha_matte
with tf.variable_scope('pred_alpha') as scope:
    outputs = tf.layers.conv2d_transpose(deconv1, 1, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    #pred_mattes = tf.nn.sigmoid(outputs)
    pred_mattes = tf.nn.sigmoid(outputs)
    tf.summary.image('pred_alpha',pred_mattes,max_outputs = 5)
    #get binary predicted mattes
    #pred_mattes = tf.cast(tf.cast(pred_mattes + 0.5, tf.int32),tf.float32)
    #pred_mattes = tf.where(tf.less_equal(pred_mattes,0.5),tf.zeros_like(pred_mattes),tf.ones_like(pred_mattes))

with tf.variable_scope('loss') as scope:
    alpha_diff = tf.sqrt(tf.square(b_GTmatte - pred_mattes) + 1e-10)
    tf.summary.scalar('alpha_loss',tf.reduce_sum(alpha_diff))
    p_RGB = []
    l_matte = tf.unstack(pred_mattes)
    BG = tf.unstack(b_GTBG)
    FG = tf.unstack(b_GTFG)
    for i in range(batch_size):
        p_RGB.append(composition_RGB(BG[i],FG[i],l_matte[i]))
    pred_RGB = tf.stack(p_RGB)
    tf.summary.image('pred_RGB',pred_RGB,max_outputs = 5)
    c_diff = tf.sqrt(tf.square(pred_RGB - b_RGB) + 1e-10) / 255
    tf.summary.scalar('comp_loss',tf.reduce_sum(c_diff))
    # wl = np.ones([batch_size,image_size,image_size,1]).fill(0.5)
    # wl[np.where(b_GTmatte == 0.5)] = 1.5
    #wl_tmp = tf.fill([batch_size,image_size,image_size,1],0.5)
    wl = tf.where(tf.equal(b_GTmatte,0.5), tf.fill([batch_size,image_size,image_size,1],1.5) ,tf.fill([batch_size,image_size,image_size,1],0.5))

    total_loss = tf.reduce_sum(wl * alpha_diff + (2-wl) * c_diff) / batch_size
    tf.summary.scalar('total_loss',total_loss)
    global_step = tf.Variable(0,trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate = 1e-6).minimize(total_loss,global_step = global_step)

    coord = tf.train.Coordinator()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(coord=coord,sess=sess)

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
    print('finish loading vgg16 model')
	#load train data
    while epoch_num < max_epochs:
        print('epoch %d' % epoch_num)	
        while batch_num < batchs_per_epoch:
            print('batch %d, loading batch data...' % batch_num)
            batch_index = sess.run(index_dequeue_op)
            batch_RGB_paths = np.array(paths_RGB)[batch_index]
            batch_alpha_paths = np.array(paths_alpha)[batch_index]
            batch_FG_paths = np.array(paths_FG)[batch_index]
            batch_BG_paths = np.array(paths_BG)[batch_index]

            batch_RGBs,batch_alphas,batch_FGs,batch_BGs = load_data(batch_RGB_paths,batch_alpha_paths,batch_FG_paths,batch_BG_paths)
            feed = {image_batch:batch_RGBs.eval(), GT_matte_batch:batch_alphas.eval(), GTBG_batch:batch_BGs.eval(), GTFG_batch:batch_FGs.eval(),is_train:True}
            _,loss,summary_str,step,p_mattes = sess.run([train_op,total_loss,summary_op,global_step,pred_mattes],feed_dict = feed)
            misc.imsave('./predict/alpha.jpg',p_mattes[0,:,:,0])
            summary_writer.add_summary(summary_str,global_step = step)
            print('loss is %f' %loss)
            batch_num += 1
        epoch_num += 1
