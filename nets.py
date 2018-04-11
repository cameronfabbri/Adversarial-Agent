import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tf_ops import *

def netG(frames, noise, num_actions, useNoise):

    if useNoise:
        print('Using noise')
        frames = tf.concat([frames, noise], axis=3)
    else: print('Not using noise')

    print('input:',frames)

    conv1 = tcl.conv2d(frames, 32, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
    conv1 = tcl.batch_norm(conv1)
    print('conv1:',conv1)

    conv2 = tcl.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
    conv2 = tcl.batch_norm(conv2)
    print('conv2:',conv2)
    
    conv3 = tcl.conv2d(conv2, 128, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
    conv3 = tcl.batch_norm(conv3)
    print('conv3:',conv3)

    conv4 = tcl.conv2d(conv3, 256, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')
    conv4 = tcl.batch_norm(conv4)
    print('conv4:',conv4)
    
    conv5 = tcl.conv2d(conv4, 512, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv5')
    conv5 = tcl.batch_norm(conv5)
    print('conv5:',conv5)

    conv6 = tcl.conv2d(conv5, 512, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv6')
    conv6 = tcl.batch_norm(conv6)
    print('conv6:',conv6)
    
    conv7 = tcl.conv2d(conv6, 512, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv7')
    conv7 = tcl.batch_norm(conv7)
    print('conv7:',conv7)
    
    conv8 = tcl.conv2d(conv7, 512, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv8')
    conv8 = tcl.batch_norm(conv8)
    print('conv8', conv8)

    # messes up if batch size is 1, aka testing
    bs = conv8.get_shape().as_list()[0]
    #if bs != 1: conv8 = tf.squeeze(conv8)
    #print('conv8', conv8)
    if bs == 1:
        conv8 = tf.squeeze(conv8)
        conv8 = tf.expand_dims(conv8, 0)
        conv8 = tf.expand_dims(conv8, 0)
        conv8 = tf.expand_dims(conv8, 0)
    print('conv8:',conv8)

    # keyboard actions
    actions = tcl.fully_connected(conv8, num_actions, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_actions')
    actions = tf.squeeze(actions)
    if len(actions.get_shape().as_list()) == 1:
        actions = tf.expand_dims(actions,0)
    print('actions:',actions)
    #exit()
    return actions

def netD(x, y, reuse=False):

    # combine the frames with the actions
    y_dim = int(y.get_shape().as_list()[-1])

    # reshape so it's batchx1x1xy_size
    y = tf.reshape(y, shape=[-1, 1, 1, y_dim])
    input_ = conv_cond_concat(x, y)

    print('netD')
    sc = tf.get_variable_scope()
    with tf.variable_scope(sc, reuse=reuse):

        conv1 = tcl.conv2d(input_, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
        conv1 = lrelu(conv1)
        
        conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
        conv2 = lrelu(conv2)
        
        conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
        conv3 = lrelu(conv3)
        
        conv4 = tcl.conv2d(conv3, 512, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
        conv4 = lrelu(conv4)
        
        conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')

        print('x:',x)
        print('conv1:',conv1)
        print('conv2:',conv2)
        print('conv3:',conv3)
        print('conv4:',conv4)
        print('conv5:',conv5)
        return conv5