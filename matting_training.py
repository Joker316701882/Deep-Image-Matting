import tensorflow as tf
import numpy as np
from matting import unpool,preprocessing



image_size = 320
batch_size = 128
#pretrained_vgg_model_path
model_path = './vgg16_weights.npz'
image_patchs = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,3))
trimap_patchs = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,1))

#CNN_input = preprocessing(image_patchs,trimap_patchs)
#for test
CNN_input = tf.concat([image_patchs,trimap_patchs],axis = 3)

en_parameters = []
arg_unpooling = []
# conv1_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(CNN_input, kernel, [1, 1, 1, 1], padding='SAME')
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
pool1,argmax1 = tf.nn.max_pool_with_argmax(conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
arg_unpooling.append(argmax1)

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
pool2,argmax2 = tf.nn.max_pool_with_argmax(conv2_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')
arg_unpooling.append(argmax2)

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
pool3,argmax3 = tf.nn.max_pool_with_argmax(conv3_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool3')
arg_unpooling.append(argmax3)

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
pool4,argmax4 = tf.nn.max_pool_with_argmax(conv4_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')
arg_unpooling.append(argmax4)

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
pool5,argmax5 = tf.nn.max_pool_with_argmax(conv5_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')
arg_unpooling.append(argmax5)

#deconv6
with tf.name_scope('deconv6') as scope:
    outputs = tf.layers.conv2d_transpose(pool5, 512, [1, 1], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv6 = tf.nn.relu(outputs)

#unpool1
unpool1 = unpool(deconv6, arg_unpooling[-1], ksize=[1, 2, 2, 1], scope='unpool1')

#deconv5
with tf.name_scope('deconv5') as scope:
    outputs = tf.layers.conv2d_transpose(unpool1, 512, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv5 = tf.nn.relu(outputs)

#unpool2
unpool2 = unpool(deconv5, arg_unpooling[-2], ksize=[1, 2, 2, 1], scope='unpool2')

#deconv4
with tf.name_scope('deconv4') as scope:
    outputs = tf.layers.conv2d_transpose(unpool2, 256, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv4 = tf.nn.relu(outputs)

#unpool3
unpool3 = unpool(deconv4, arg_unpooling[-3], ksize=[1, 2, 2, 1], scope='unpool3')

#deconv3
with tf.name_scope('deconv3') as scope:
    outputs = tf.layers.conv2d_transpose(unpool3, 128, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv3 = tf.nn.relu(outputs)

#unpool4
unpool4 = unpool(deconv3, arg_unpooling[-4], ksize=[1, 2, 2, 1], scope='unpool4')

#deconv2
with tf.name_scope('deconv2') as scope:
    outputs = tf.layers.conv2d_transpose(unpool4, 64, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv2 = tf.nn.relu(outputs)

#unpool5
unpool5 = unpool(deconv2, arg_unpooling[-5], ksize=[1, 2, 2, 1], scope='unpool5')

#deconv1
with tf.name_scope('deconv1') as scope:
    outputs = tf.layers.conv2d_transpose(unpool5, 64, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    deconv1 = tf.nn.relu(outputs)

#pred_alpha_matte
with tf.name_scope('pred_alpha_matte') as scope:
    outputs = tf.layers.conv2d_transpose(deconv1, 1, [5, 5], strides=(1, 1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
    pred_alpha_matte = tf.nn.sigmoid(outputs)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    weights = np.load(model_path)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        if i == 26:
            break
        if k == 'conv1_1_W':  
            sess.run(en_parameters[i].assign(np.concatenate([weights[k],np.zeros([3,3,1,64])],axis = 2)))
        else:
            sess.run(en_parameters[i].assign(weights[k]))



