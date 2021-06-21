# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:10:45 2020

@author: kdutta01
"""


import os 
import natsort
import numpy as np
from skimage.io import imread
    
def load_train_data():
    images_train = np.load('images_train.npy')
    images_train_T1 = np.load('images_train_T1.npy')
    mask_train = np.load('mask_train.npy')
    print('====== Loading of Training Images and Masks ===================')
    return images_train,images_train_T1,mask_train

def load_test_data():
    images_test = np.load('images_test.npy')
    print('======Loading of Test Data=======')
    return images_test
    



    
    




                     


