# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:10:45 2020

@author: kdutta01
"""


import os 
import natsort
import numpy as np
from skimage.io import imread

################################ TRAINING ##############################################
train_datapath = './data/train_image_T2/'
train_datapath_T1 = './data/train_image_T1/'
dirr = os.listdir(train_datapath)

dirr_image = os.path.join(train_datapath, 'image/')
dirr_mask = os.path.join(train_datapath, 'label/')
dirr_image_T1 = os.path.join(train_datapath_T1, 'image/')

no_of_images = len(os.listdir(dirr_image))
image_row_size = 128
image_col_size = 128

image = np.ndarray((no_of_images, image_row_size , image_col_size), dtype = np.uint8)
image_T1 = np.ndarray((no_of_images, image_row_size , image_col_size), dtype = np.uint8)
mask = np.ndarray((no_of_images, image_row_size, image_col_size), dtype = np.uint8)

image_iter = natsort.natsorted(os.listdir(dirr_image))
mask_iter = natsort.natsorted(os.listdir(dirr_mask))

count = 0
for image_name in image_iter:
    img = imread(os.path.join(dirr_image, image_name), as_gray = True)
    img = np.array([img])
    image[count] = img
    
    img_T1 = imread(os.path.join(dirr_image_T1, image_name), as_gray = True)
    img_T1 = np.array([img_T1])
    image_T1[count] = img_T1
    count = count + 1
    
count = 0
for mask_name in image_iter:
    msk = imread(os.path.join(dirr_mask, mask_name), as_gray = True)
    msk = np.array([msk])
    mask[count] = msk
    count = count + 1
 

image = np.expand_dims(image,3)
image_T1 = np.expand_dims(image_T1,3)
mask = np.expand_dims(mask,3)
#image = np.append(image, image_T1, axis = 3)

np.save('images_train.npy', image)
np.save('images_train_T1.npy', image_T1)
np.save('mask_train.npy', mask)
    
def load_train_data():
    images_train = np.load('images_train.npy')
    images_train_T1 = np.load('images_train_T1.npy')
    mask_train = np.load('mask_train.npy')
    print('====== Loading of Training Images and Masks ===================')
    return images_train,images_train_T1,mask_train


########################   Validation   ###################
validation_datapath = './data/validation_T2/'
validation_datapath_T1 = './data/validation_T1/'
dirr_validation = os.listdir(validation_datapath)
dirr_image = os.path.join(validation_datapath, 'image/')
dirr_mask = os.path.join(validation_datapath, 'label/')
dirr_image_T1 = os.path.join(validation_datapath_T1, 'image/')

no_of_images = len(os.listdir(dirr_image))
image_row_size = 128
image_col_size = 128

image = np.ndarray((no_of_images, image_row_size , image_col_size), dtype = np.uint8)
mask = np.ndarray((no_of_images, image_row_size, image_col_size), dtype = np.uint8)
image_T1 = np.ndarray((no_of_images, image_row_size, image_col_size), dtype = np.uint8)

image_iter = natsort.natsorted(os.listdir(dirr_image))
mask_iter = natsort.natsorted(os.listdir(dirr_mask))

count = 0
for image_name in image_iter:
    img = imread(os.path.join(dirr_image, image_name), as_gray = True)
    img = img.astype(np.uint8)
    img = np.array([img])
    image[count] = img
    
    img_T1 = imread(os.path.join(dirr_image_T1, image_name), as_gray = True)
    img_T1 = img_T1.astype(np.uint8)
    img_T1 = np.array([img_T1])
    image_T1[count] = img_T1
    count = count + 1
    
count = 0
for mask_name in image_iter:
    msk = imread(os.path.join(dirr_mask, mask_name), as_gray = True)
    msk = np.array([msk])
    mask[count] = msk
    count = count + 1
 

image = np.expand_dims(image,3)
mask = np.expand_dims(mask,3)
image_T1 = np.expand_dims(image_T1,3)
#image_final_validation = np.append(image, image_T1, axis = 3)


np.save('images_validation.npy', image)
np.save('mask_validation.npy', mask)
np.save('images_validation_T1.npy', image_T1)
    
def load_validation_data():
    images_train = np.load('images_validation.npy')
    mask_train = np.load('mask_validation.npy')
    images_train_T1 = np.load('images_validation_T1.npy')
    print('====== Loading of Validation Images and Masks ===================')
    return images_train,images_train_T1, mask_train


    
################################ Testing ###############

test_datapath = './data/test_image_T2/set1/'
test_datapath_T1 = './data/test_image_T1/set1/'
dirr_test_image = os.listdir(test_datapath)
dirr_test_image_T1 = os.listdir(test_datapath_T1)

no_of_test_images = len(dirr_test_image)
test_image = np.ndarray((no_of_test_images, image_row_size, image_col_size), dtype = np.uint8)
test_image_T1 = np.ndarray((no_of_test_images, image_row_size, image_col_size), dtype = np.uint8)

image_test_iter = natsort.natsorted(dirr_test_image)

count = 0
for image_name in image_test_iter:
    img_test = imread(os.path.join(test_datapath, image_name), as_gray = True)
    img_test = img_test.astype(np.uint8)
    img_test = np.array([img_test])
    test_image[count] = img_test
    
    img_test_T1 = imread(os.path.join(test_datapath_T1, image_name), as_gray = True)
    img_test_T1 = img_test_T1.astype(np.uint8)
    img_test_T1 = np.array([img_test_T1])
    test_image_T1[count] = img_test_T1
    count+=1
    
test_image = np.expand_dims(test_image, axis = 3)
test_image_T1 = np.expand_dims(test_image_T1, axis = 3)
test_image_final = np.append(test_image, test_image_T1, axis = 3)
np.save('images_test.npy', test_image_final)


def load_test_data():
    images_test = np.load('images_test.npy')
    print('======Loading of Test Data=======')
    return images_test
    



    
    




                     


