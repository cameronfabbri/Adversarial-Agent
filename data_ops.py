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
import _pickle as pickle

'''
   Converts a single image to range [low,high]
'''
def preprocess(x, high=1, low=-1):
    #return ((high-low)*(x-np.min(x))/(np.max(x)-np.min(x)))+low
    return (x/127.5)-1.0

'''
   Converts a single image from [-1,1] range to [0,255]
'''
def deprocess(image):
    #return ((image+1.0)/2.0)
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



def readData_old(dataset):

    datafile = np.load(dataset+'.npy')

    images   = []
    commands = []
    for e in datafile:
        image   = e[0]
        command = e[1]
        images.append(image)
        commands.append(command)

    # this astype was causing D to explode
    #images   = np.asarray(images)#.astype('float32')
    #commands = np.asarray(commands)#.astype('float32')
    
    return images,commands