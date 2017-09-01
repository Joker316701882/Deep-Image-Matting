'''
deconv_v5.py
change the architecture of loss. 
'''
import tensorflow as tf
import numpy as np
from matting_numpy_v2 import load_path,load_data,load_test_data
import os
from scipy import misc



image_size = 320
train_batch_size = 5
max_epochs = 1000000

#checkpoint file path
pretrained_model = './model/model.ckpt'
#pretrained_model = False
test_dir = './alhpamatting'
test_outdir = './test_predict'

#pretrained_vgg_model_path
model_path = './vgg16_weights.npz'
log_dir = '/data/gezheng/matting_log'
dataset_RGB = '/data/gezheng/data-matting/new/comp_RGB'
dataset_alpha = '/data/gezheng/data-matting/new/alpha_final'
dataset_FG = '/data/gezheng/data-matting/new/FG_final'
dataset_BG = '/data/gezheng/data-matting/new/BG'

paths_RGB,paths_alpha,paths_FG,paths_BG = load_path(dataset_RGB,dataset_alpha,dataset_FG,dataset_BG)

range_size = len(paths_RGB)
#range_size/batch_size has to be int
batchs_per_epoch = int(range_size/train_batch_size) 

index_queue = tf.train.range_input_producer(range_size, num_epochs=None,shuffle=True, seed=None, capacity=32)
index_dequeue_op = index_queue.dequeue_many(train_batch_size, 'index_dequeue')

image_batch = tf.placeholder(tf.float32, shape=(None,image_size,image_size,3))
raw_RGBs = tf.placeholder(tf.float32, shape=(None,image_size,image_size,3))
GT_matte_batch = tf.placeholder(tf.float32, shape = (None,image_size,image_size,1))
GT_trimap = tf.placeholder(tf.float32, shape = (None,image_size,image_size,1))
GTBG_batch = tf.placeholder(tf.float32, shape = (None,image_size,image_size,3))
GTFG_batch = tf.placeholder(tf.float32, shape = (None,image_size,image_size,3))

en_parameters = []

b_RGB = tf.identity(image_batch,name = 'b_RGB')
b_trimap = tf.identity(GT_trimap,name = 'b_trimap')
b_GTmatte = tf.identity(GT_matte_batch,name = 'b_GTmatte')
b_GTBG = tf.identity(GTBG_batch,name = 'b_GTBG')
b_GTFG = tf.identity(GTFG_batch,name = 'b_GTFG')

tf.summary.image('GT_matte_batch',b_GTmatte,max_outputs = 5)
tf.summary.image('trimap',b_trimap,max_outputs = 5)
tf.summary.image('raw_RGBs',raw_RGBs,max_outputs = 5)

b_input = tf.concat([b_RGB,b_trimap],3)

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

# conv6_1
with tf.name_scope('conv6_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv6_1 = tf.nn.relu(out, name='conv6_1')

training = True

#deconv6
with tf.variable_scope('deconv6') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv6 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv6')

#deconv5_1/unpooling
with tf.variable_scope('deconv5_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv6, 512, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv5_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv5_2
with tf.variable_scope('deconv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv5_2')

#deconv4_1/unpooling
with tf.variable_scope('deconv4_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv5_2, 512, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv4_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv4_2
with tf.variable_scope('deconv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv4_2')

#deconv3_1/unpooling
with tf.variable_scope('deconv3_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv4_2, 256, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv3_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv3_2
with tf.variable_scope('deconv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv3_2')

#deconv2_1/unpooling
with tf.variable_scope('deconv2_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv3_2, 128, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv2_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv2_2
with tf.variable_scope('deconv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv2_2')

#deconv1_1/unpooling
with tf.variable_scope('deconv1_1') as scope:
    outputs = tf.layers.conv2d_transpose(deconv2_2, 64, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv1_1 = tf.nn.relu(tf.layers.batch_normalization(outputs,training=training))

#deconv1_2
with tf.variable_scope('deconv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    deconv1_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv1_2')

#pred_alpha_matte
with tf.variable_scope('pred_alpha') as scope:
    outputs = tf.layers.conv2d_transpose(deconv1_2, 1, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    pred_mattes = tf.nn.sigmoid(outputs)

wl_alpha = tf.where(tf.equal(b_trimap,0.5), tf.fill([train_batch_size,image_size,image_size,1],1.) ,b_trimap)
wl_alpha = tf.where(tf.less(b_trimap,0.5), tf.fill([train_batch_size,image_size,image_size,1],0.5) ,wl_alpha)
wl_alpha = tf.where(tf.greater(b_trimap,0.5), tf.fill([train_batch_size,image_size,image_size,1],0.) ,wl_alpha)

wl_RGB = tf.where(tf.equal(b_trimap,0.5), tf.fill([train_batch_size,image_size,image_size,1],0.) ,b_trimap)
wl_RGB = tf.where(tf.greater(b_trimap,0.5), tf.fill([train_batch_size,image_size,image_size,1],1.) ,wl_RGB)
wl_RGB = tf.where(tf.less(b_trimap,0.5), tf.fill([train_batch_size,image_size,image_size,1],0.5) ,wl_RGB)

# tf.summary.image('wl_RGB',wl_RGB,max_outputs = 5)
# tf.summary.image('wl_alpha',wl_alpha,max_outputs = 5)

tf.summary.image('pred_mattes',pred_mattes,max_outputs = 5)

alpha_diff = tf.sqrt(tf.square(pred_mattes - GT_matte_batch)+ 1e-12)

p_RGB = []
pred_mattes.set_shape([train_batch_size,image_size,image_size,1])
b_GTBG.set_shape([train_batch_size,image_size,image_size,3])
b_GTFG.set_shape([train_batch_size,image_size,image_size,3])
raw_RGBs.set_shape([train_batch_size,image_size,image_size,3])


# pred_final =  tf.where(tf.equal(b_trimap,0.5), pred_mattes,b_trimap)
l_matte = tf.unstack(pred_mattes)
BG = tf.unstack(b_GTBG)
FG = tf.unstack(b_GTFG)

for i in range(train_batch_size):
    p_RGB.append(l_matte[i] * FG[i] + (tf.constant(1.) - l_matte[i]) * BG[i])
pred_RGB = tf.stack(p_RGB)

tf.summary.image('pred_RGB',pred_RGB,max_outputs = 5)
c_diff = tf.sqrt(tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12)

alpha_loss = tf.reduce_sum(alpha_diff * wl_alpha)/train_batch_size
comp_loss = tf.reduce_sum(c_diff * wl_RGB)/(train_batch_size*3)

tf.summary.scalar('alpha_loss',alpha_loss)
tf.summary.scalar('comp_loss',comp_loss)

total_loss = (alpha_loss + comp_loss) * 0.5
tf.summary.scalar('total_loss',total_loss)
global_step = tf.Variable(0,trainable=False)

# tf.summary.image('pred_final',pred_final,max_outputs = 5)
train_op = tf.train.AdamOptimizer(learning_rate = 1e-5).minimize(total_loss,global_step = global_step)

saver = tf.train.Saver(tf.trainable_variables() , max_to_keep = 1)

coord = tf.train.Coordinator()
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(coord=coord,sess=sess)
    batch_num = 0
    epoch_num = 0
    #initialize all parameters in vgg16
    if not pretrained_model:
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
    else:
        print('Restoring pretrained model...')
        saver.restore(sess,tf.train.latest_checkpoint('./model'))
    sess.graph.finalize()

    while epoch_num < max_epochs:
        print('epoch %d' % epoch_num)   
        while batch_num < batchs_per_epoch:
            print('batch %d, loading batch data...' % batch_num)
            batch_index = sess.run(index_dequeue_op)
            batch_RGB_paths = paths_RGB[batch_index]
            batch_alpha_paths = paths_alpha[batch_index]
            batch_FG_paths = paths_FG[batch_index]
            batch_BG_paths = paths_BG[batch_index]

            batch_RGBs,batch_trimaps,batch_alphas,batch_BGs,batch_FGs,RGBs_with_mean = load_data(batch_RGB_paths,batch_alpha_paths,batch_FG_paths,batch_BG_paths)

            feed = {image_batch:batch_RGBs, GT_matte_batch:batch_alphas,GT_trimap:batch_trimaps, GTBG_batch:batch_BGs, GTFG_batch:batch_FGs,raw_RGBs:RGBs_with_mean}

            _,loss,summary_str,step= sess.run([train_op,total_loss,summary_op,global_step],feed_dict = feed)
            print('loss is %f' %loss)

            if step%50 == 0:
                print('saving model......')
                saver.save(sess,'./model/model.ckpt',global_step = step, write_meta_graph = False)

                print('test on alphamatting.com...')
                test_RGBs,test_trimaps,test_alphas,all_shape,image_paths = load_test_data(test_dir)
            
                feed = {image_batch:test_RGBs,GT_trimap:test_trimaps}
                test_out = sess.run(pred_mattes,feed_dict = feed)
                vali_diff = []
                for i in range(len(test_out)):
                    i_out = misc.imresize(test_out[i][:,:,0],all_shape[i])
                    vali_diff.append(np.sum(np.abs(i_out/256.0-test_alphas[i])))
                    misc.imsave(os.path.join(test_outdir,image_paths[i]),i_out)
                vali_loss = np.mean(vali_diff)
                print('validation loss is '+ str(vali_loss))
                validation_summary = tf.Summary()
                validation_summary.value.add(tag='validation_loss',simple_value = vali_loss)
                summary_writer.add_summary(validation_summary,step)

            summary_writer.add_summary(summary_str,global_step = step)
            batch_num += 1
        epoch_num += 1


