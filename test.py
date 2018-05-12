'''
    This isn't working currently. The model during testing only is input invariant
    Problems I've ruled out:
        - different shapes in the lastish layer of the network (due to squeezing and batch size 1 etc) - not the problem
        - image is read in differently like BGR instead of RGB - nope this is fine, I printed out the values in train and test and they match up
        - model not being loaded correctly - it is
        - model looking at just noise and predicting - nope I removed noise and it still does the same crap
        - here in the loop when looking at 4 frames, all frames are the same - now trainging on only 1 frame so this isn't the problem

    Seems like replacing the frames with training data works. Don't see a difference in types or ranges though

'''

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode,mean
import numpy as np
#from motion import motion_detection
import tensorflow as tf
from random import randint

from data_ops import *
from nets import *

WIDTH = 256
HEIGHT = 256

#choices = deque([], maxlen=5)
#hl_hist = 250
#choice_hist = deque([], maxlen=hl_hist)

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(W)
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    #ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def right():
    PressKey(W)
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    #ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    
def reverse():
    PressKey(W)
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():

    #if random.randrange(0,3) == 1:
    #    PressKey(W)
    #else:
    #    ReleaseKey(W)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


if __name__ == '__main__':

    BATCH_SIZE = 16
    num_actions = 9

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=False,default='driving',type=str,help='Dataset to use for training')
    parser.add_argument('--ganloss',required=False,default='gan',type=str,help='Type of GAN loss to use')
    parser.add_argument('--numFrames',required=False,default=1,type=int,help='Number of frames to send in at once')
    parser.add_argument('--useNoise',required=False,default=1,type=int,help='Whether or not to use noise')

    a = parser.parse_args()

    dataset = a.dataset
    ganloss = a.ganloss
    numFrames = a.numFrames
    useNoise  = bool(a.useNoise)

    checkpoint_dir = 'checkpoints/dataset_'+dataset+'/ganloss_'+ganloss+'/numFrames_'+str(numFrames)+'/useNoise_'+str(useNoise)+'/'
    #checkpoint_dir = 'checkpoints/dataset_'+dataset+'/loss_classification/numFrames_'+str(numFrames)+'/useNoise_'+str(useNoise)+'/'

    frames_p = tf.placeholder(tf.float32, shape=(BATCH_SIZE,256,256,3*numFrames), name='frames_p')
    noise_p  = tf.placeholder(tf.float32, shape=(BATCH_SIZE,256,256,1), name='noise_p')

    # generate an set of actions
    gen_actions = netG(frames_p, noise_p, num_actions, useNoise)

    saver = tf.train.Saver(max_to_keep=1)
    init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess = tf.Session()
    sess.run(init)

    checkpoint_dir = checkpoint_dir.replace('/','\\')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print('\n\ncheckpoint_dir:',checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('\n\nRestoring previous model...\n\n')
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored')
        except:
            print('Could not restore model')
            raise

    #print(tf.trainable_variables())
    #var = [v for v in tf.trainable_variables() if v.name == "g_conv1/weights:0"][0]
    #print(sess.run(var))
    #exit()

    paused = False
    counter = 0

    train_paths_, info_dict = readDataset(dataset)
    num_train = len(train_paths_)
    train_paths = []
    for t in train_paths_:
        train_paths.append(t.replace('\\','/'))

    while(True):
        if not paused:
            current_size = 0
            batchFrames = []
            batchNoise  = []

            frames = grab_screen(region=(0,40,800,625))
            frames = cv2.resize(frames, (256,256))
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

            # grab BATCH_SIZE random training images
            while current_size < BATCH_SIZE:

                #start_idx = random.randint(0,num_train)
                #end_idx   = start_idx + (numFrames-1)

                # end case check
                #if end_idx >= num_train:
                #    end_idx = num_train-1
                #    start_idx = num_train-numFrames

                #frames = cv2.imread(train_paths[start_idx])
                frames = preprocess(frames)

                noise = np.random.normal(-1.0, 1.0, size=[256,256,1]).astype(np.float32)
                batchNoise.append(noise)
                batchFrames.append(frames)
                current_size += 1

            #cv2.imwrite('frames.png', frames)
            #exit()
            #batchFrames.append(frames)
            #batchNoise.append(noise)
            batchFrames = np.asarray(batchFrames)
            batchNoise  = np.asarray(batchNoise)

            prediction = np.asarray(sess.run(gen_actions, feed_dict={frames_p:batchFrames, noise_p:batchNoise}))
            iii = randint(0, BATCH_SIZE-1)
            prediction = prediction[iii]
            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            if counter % 10 == 0:
                print(prediction)
                print()
                #exit()
                #print(choice_picked)
                #print()
                #print(frames)
            counter += 1
        keys = key_check()