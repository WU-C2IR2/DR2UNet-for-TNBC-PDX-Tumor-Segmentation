# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:24:29 2020

@author: kdutta01
"""


from dense_unet import denseunet
from res_unet import resunet
from new_r2udensenet import r2udensenet
from new_r2unet import r2unet
from unet import unet_model
from data2D import load_train_data, load_test_data
import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time
from sklearn.model_selection import KFold

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dice = []
precision = []
recall = []
auc = []
accuracy = []

#################### Training of the Network by Five Fold Cross Validation #################################

def train():
    kfold = KFold(n_splits = 5, shuffle = True)
    images_train, images_train_T1, mask_train = load_train_data()
    images_train = images_train.astype('float32')
    images_train_T1 = images_train_T1.astype('float32')
    mask_train = mask_train.astype('float32')

    images_train_mean = np.mean(images_train)
    images_train_std = np.std(images_train)
    images_train = (images_train - images_train_mean)/images_train_std

    images_train_mean = np.mean(images_train_T1)
    images_train_std = np.std(images_train_T1)
    images_train_T1 = (images_train_T1 - images_train_mean)/images_train_std
    mask_train /= 255.
    image_datagen = ImageDataGenerator(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    mask_datagen = ImageDataGenerator(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    def generate_generator(generator1, generator2, images_train, mask_train, images_train_T1):
        train_gen = generator1.flow(images_train, y= None, batch_size = 2, shuffle = True, seed = 100, 
                                              sample_weight = None)
    
        label_gen = generator2.flow(mask_train, y =None, batch_size = 2, shuffle = True, seed = 100, 
                                              sample_weight = None)
    
        train_gen_T1 = generator1.flow(images_train_T1, y= None, batch_size = 2, shuffle = True, seed = 100, 
                                              sample_weight = None)
        
        while True:
            image_T2 = train_gen.next()
            mask = label_gen.next()
            image_T1 = train_gen_T1.next()
            image_final = np.append(image_T2, image_T1, axis = 3)
            yield image_final,mask

#train_generator = generate_generator(image_datagen, mask_datagen, images_train, mask_train, images_train_T1)
    def get_model_name(k):
        return 'model_r2udensenet'+str(k)+'.hdf5'
    
    def get_log_name(k):
        return 'log_r2udensenet'+str(k)+'.csv'

    fold_no = 1
    for train_index, validation_index in kfold.split(images_train):
        trainData_T2 = images_train[train_index]
        trainData_T1 = images_train_T1[train_index]
        trainMask = mask_train[train_index]
    
        validationData_T2 = images_train[validation_index]
        validationData_T1 = images_train_T1[validation_index]
        validationMask = mask_train[validation_index]
    
        train_generator = generate_generator(image_datagen, mask_datagen, trainData_T2, trainMask, trainData_T1)
        validation_generator = generate_generator(image_datagen, mask_datagen, validationData_T2, validationMask, validationData_T1)
    
        model = r2udensenet()
        weight_directory = 'weights'
        if not os.path.exists(weight_directory):
            os.mkdir(weight_directory)
        model_checkpoint = ModelCheckpoint(os.path.join(weight_directory,get_model_name(fold_no)), monitor = 'loss', verbose = 1, save_best_only=True)
    
        log_directory = 'logs'
        if not os.path.exists(log_directory):
            os.mkdir(log_directory)
        logger = CSVLogger(os.path.join(log_directory,get_log_name(fold_no)), separator = ',', append = False)
    
        start = time()
        history = model.fit_generator(train_generator, steps_per_epoch = len(trainData_T2)/2, epochs = 250, verbose = 1, 
                                  validation_data = validation_generator,
              validation_steps = 26, callbacks = [model_checkpoint, logger])
        print("===== K Fold Validation Step ======", fold_no)
        fold_no = fold_no + 1;
    
        val_punch = np.append(validationData_T2,validationData_T1, axis = 3)
        scores = model.evaluate(val_punch, validationMask, verbose=0)
        dice.append(scores[1])
        precision.append(scores[2])
        recall.append(scores[3])
        auc.append(scores[5])
        accuracy.append(scores[5])
        

    print('DICE Coefficient == ', dice , 'Mean Dice == ', np.mean(dice))
    print('Precision==', precision, 'Mean Precision ==', np.mean(precision))
    print('Recall==', recall, 'Mean Recall ==', np.mean(recall))
    print('AUC==', auc, 'Mean AUC==', np.mean(auc))
    print('Accuracy==', accuracy, 'Mean Accuracy ==',np.mean(accuracy))
           
if __name__ == '__main__':
    train()
    


