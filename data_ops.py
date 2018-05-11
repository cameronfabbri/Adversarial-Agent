'''

    Data Operations

'''

from tqdm import tqdm
import numpy as np
import math
import skimage.io
import argparse
from glob import glob
import sys
import cv2
import os
import io

#try: import _pickle as pickle
#except: import cPickle as pickle
import _pickle as pickle

'''
   Converts a single image to range [low,high]
'''
def preprocess(x, high=1, low=-1):
    #return ((high-low)*(x-np.min(x))/(np.max(x)-np.min(x)))+low
    return (x/127.5)-1.

'''
   Converts a single image from [-1,1] range to [0,255]
'''
def deprocess(image):
    image = (image+1.)
    image *= 127.5
    #image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def readDataset(dataset):

    # this isn't in correct order but that's okay because we just pick one, then get the rest from the dictionary
    image_paths = sorted(glob(dataset+'/*.png'))

    pkl_file = open(dataset+'/actions.pkl', 'rb')
    a = pickle.load(pkl_file)
    return image_paths, a
    '''
    try:
        pkl_file = open(dataset+'/actions_p3.pkl', 'rb')
        a = pickle.load(pkl_file)
        return image_paths, a
    except:
        pkl_file = open(dataset+'/actions_p2.pkl', 'rb')
        a = pickle.load(pkl_file)

        for key in a:
           key_ = key.encode('ascii', 'ignore')
           new_dict[key_] = a[key]
           return image_paths, new_dict
    '''