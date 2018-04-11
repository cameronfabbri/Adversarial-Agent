import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import sys
from data_ops import *
import _pickle as pickle

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

if len(sys.argv) < 2:
    print('\nUsage: python collect_data.py [dataset]')
    exit()

dataset = sys.argv[1]

try: os.mkdir(dataset+'/')
except: pass

'''
    TODO:
        - Have continue option that just loads the file and appends
        - Save periodically so don't run outta RAM
'''
if os.path.isfile(dataset+'.npy'):
    pass
    #a = input('File '+dataset+' already exists, overwrite? (y/n)\n')
    #if a == 'y': pass
    #else: exit()

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output

import ipdb

def main(dataset):
    training_data = []
    # sleeps for 4 seconds
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    print('STARTING!!!')
    
    # 42 fps
    frame_num = 0
    info_dict = {}
    while(True):
        
        if not paused:
            # get screen from top left
            screen = grab_screen(region=(0,40,800,625))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (256,256))

            keys = key_check()
            output = keys_to_output(keys)

            cv2.imwrite(dataset+'/'+str(frame_num)+'.png',screen)
            info_dict[dataset+'/'+str(frame_num)+'.png'] = output

            frame_num += 1

            if frame_num % 100 == 0:
                print('Recorded '+str(frame_num)+' frames')

        keys = key_check()
        if 'T' in keys:
            print('Saving and quitting...')
            pkl = open(dataset+'/actions.pkl', 'wb')
            data = pickle.dumps(info_dict)
            pkl.write(data)
            pkl.close()
            break

main(dataset)