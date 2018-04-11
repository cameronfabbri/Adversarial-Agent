from __future__ import print_function

import scipy.misc as misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import skimage.io
import argparse
import random
import cv2
import os
import io

from nets import *
from data_ops import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=False,default='small',type=str,help='Dataset to use for training')
    parser.add_argument('--ganLoss',required=False,default='wgan',type=str,help='Type of GAN loss to use')
    parser.add_argument('--numFrames',required=False,default=1,type=int,help='Number of frames to send in at once')
    parser.add_argument('--useNoise',required=False,default=0,type=int,help='Whether or not to use noise')

    a = parser.parse_args()

    dataset   = a.dataset
    ganLoss   = a.ganLoss
    numFrames = a.numFrames
    useNoise  = bool(a.useNoise)

    checkpoint_dir = 'checkpoints/dataset_'+dataset+'/ganLoss_'+ganLoss+'/numFrames_'+str(numFrames)+'/useNoise_'+str(useNoise)+'/'
    try: os.makedirs(checkpoint_dir)
    except: pass

    print('Loading data...')
    #train_images, train_actions = readData(dataset)
    train_paths_, info_dict = readDataset(dataset)
    num_train = len(train_paths_)
    print('Number of training frames:',num_train)

    # gotta remove the extra \ windows puts in
    train_paths = []
    for t in train_paths_:
        train_paths.append(t.replace('\\','/'))

    BATCH_SIZE = 32
    num_actions = 9

    global_step = tf.Variable(0, name='global_step', trainable=False)

    frames_p     = tf.placeholder(tf.float32, shape=(BATCH_SIZE,256,256,3*numFrames), name='frames_p')
    noise_p      = frames_p_n   = tf.placeholder(tf.float32, shape=(BATCH_SIZE,256,256,1), name='frames_p')
    real_actions = tf.placeholder(tf.float32, shape=(BATCH_SIZE, num_actions), name='real_actions')

    # generate an set of actions given a series of 4 frames
    gen_actions = netG(frames_p, noise_p, num_actions, useNoise)

    # send frames with real actions to D
    D_real = netD(frames_p, real_actions)

    # send frames with generated actions to D
    D_fake = netD(frames_p, gen_actions, reuse=True)

    e = 1e-8
    # cost functions
    if ganLoss == 'wgan':
        errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
        errG = -tf.reduce_mean(D_fake)

        # gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = real_actions*epsilon + (1-epsilon)*gen_actions
        d_hat = netD(frames_p, x_hat, reuse=True)
        gradients = tf.gradients(d_hat, x_hat)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
        errD += gradient_penalty
    if ganLoss == 'gan':
        D_real = tf.nn.sigmoid(D_real)
        D_fake = tf.nn.sigmoid(D_fake)
        errG = tf.reduce_mean(-tf.log(D_fake)+e)
        errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))
    if ganLoss == 'lsgan':
        errD_real = tf.nn.sigmoid(D_real)
        errD_fake = tf.nn.sigmoid(D_fake)
        errG = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
        errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))

    # tensorboard summaries
    tf.summary.scalar('d_loss', tf.reduce_mean(errD))
    tf.summary.scalar('g_loss', tf.reduce_mean(errG))

    # get all trainable variables, and split by network G and network D
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    if ganLoss == 'wgan':
        G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errG, var_list=g_vars, global_step=global_step)
        D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errD, var_list=d_vars)
    else:
        G_train_op = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step)
        D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(errD, var_list=d_vars)

    saver = tf.train.Saver(max_to_keep=1)

    init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess = tf.Session()
    sess.run(init)

    # write out logs for tensorboard to the checkpointSdir
    summary_writer = tf.summary.FileWriter(checkpoint_dir+'/logs/', graph=tf.get_default_graph())

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring previous model...')
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored')
        except:
            print('Could not restore model')
            raise

    step = int(sess.run(global_step))

    merged_summary_op = tf.summary.merge_all()

    epoch_num = int(step/(num_train/BATCH_SIZE))

    num_D = 100

    high = 1
    low  = -1

    while True:

        if step > 25: num_D = 5

        # make sure if using gan or lsgan num_D is always 1
        if ganLoss == 'gan' or ganLoss == 'lsgan': num_D = 1

        epoch_num = int(step/(num_train/BATCH_SIZE))

        current_size = 0

        batchFrames  = []
        batchNoise   = []
        batchActions = []
        while current_size < BATCH_SIZE:
            # pick a random starting point
            start_idx = random.randint(0,num_train)
            end_idx   = start_idx + (numFrames-1)

            # end case check
            if end_idx >= num_train:
                end_idx = num_train-1
                start_idx = num_train-numFrames

            #a = cv2.imread(train_paths[start_idx])
            #b = misc.imread(train_paths[start_idx])
            #cv2.imshow('image',b)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #print(a)
            #print('---')
            #print(b)
            #exit()

            # get first frame
            frames = preprocess(cv2.imread(train_paths[start_idx]))
            #print(frames)
            #exit()
            # get the rest of the frames if there are more
            if numFrames > 1:
                for i in range(numFrames-1):
                    frames = np.dstack((frames, preprocess(cv2.imread(train_paths[start_idx+i+1]))))
            #frames = np.dstack((frames, preprocess(cv2.imread(train_paths[start_idx+2]))))
            #frames = np.dstack((frames, preprocess(cv2.imread(train_paths[start_idx+3]))))

            #frames = preprocess(frames)

            # add noise
            noise = np.random.normal(-1.0, 1.0, size=[256,256,1]).astype(np.float32)
            batchNoise.append(noise)

            # get actions - only get first action - put in range [-1, 1]
            #action = train_actions[start_idx]
            action = info_dict[train_paths[start_idx]]
            action = ((high-low)*(action-np.min(action))/(np.max(action)-np.min(action)))+low
            
            batchActions.append(action)
            batchFrames.append(frames)

            current_size +=1

        batchFrames = np.asarray(batchFrames)
        batchNoise  = np.asarray(batchNoise)
        batchActions = np.asarray(batchActions)

        for itr in range(num_D):
            sess.run(D_train_op, feed_dict={frames_p:batchFrames, noise_p:batchNoise, real_actions:batchActions})
        
        sess.run(G_train_op, feed_dict={frames_p:batchFrames, noise_p:batchNoise})
        
        D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={frames_p:batchFrames, noise_p:batchNoise, real_actions:batchActions})

        gen_action, real_action = sess.run([gen_actions,real_actions], feed_dict={frames_p:batchFrames, noise_p:batchNoise, real_actions:batchActions})
        #print(chr(27) + "[2J")
        print(gen_action[0])#.astype('int32'))
        print(real_action[0].astype('int32'))

        summary_writer.add_summary(summary, step)
        print('epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G loss:',G_loss,'\n\n')
        step += 1

        if step%500 == 0:
            print('Saving model...')
            saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
            saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
            print('Model saved\n')
