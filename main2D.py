# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:24:29 2020

@author: kdutta01
"""


#from model import unet
#from model2D import unet
#from model_bcdu_net import BCDU_net_D3
#from model_multi_resnet import MultiResUnet
#from res_unet import resunet
#from r2unet import r2unet

from r2udensenet import r2udensenet
from data2D import load_train_data, load_test_data, load_validation_data
import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#################### Training of the Network #################################
def train():
    print('=============Loading of Training Data and Preprocessing==============')
    
    images_train, images_train_T1, mask_train = load_train_data()
    images_validation, images_validation_T1, mask_validation = load_validation_data()
    
    images_train = images_train.astype('float32')
    images_train_T1 = images_train_T1.astype('float32')
    mask_train = mask_train.astype('float32')
    
    images_validation = images_validation.astype('float32')
    images_validation_T1 = images_train_T1.astype('float32')
    mask_validation = mask_validation.astype('float32')
    
    images_train_mean = np.mean(images_train)
    images_train_std = np.std(images_train)
    images_train = (images_train - images_train_mean)/images_train_std
    
    images_train_mean = np.mean(images_train_T1)
    images_train_std = np.std(images_train_T1)
    images_train_T1 = (images_train_T1 - images_train_mean)/images_train_std
    mask_train /= 255.
    
    images_validation_mean = np.mean(images_validation)
    images_validation_std = np.std(images_validation)
    images_validation = (images_validation - images_validation_mean)/images_validation_std
    
    images_validation_mean = np.mean(images_validation_T1)
    images_validation_std = np.std(images_validation_T1)
    images_validation_T1 = (images_validation_T1 - images_validation_mean)/images_validation_std
    mask_validation /= 255.
    
    
    ###########Image_Generator##############
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
    
    train_generator = generate_generator(image_datagen, mask_datagen, images_train, mask_train, images_train_T1)
    validation_generator = generate_generator(image_datagen, mask_datagen, images_validation, mask_validation, images_validation_T1)
    
    # validation_gen = image_datagen.flow(images_validation, y= None, batch_size = 2, shuffle = True, seed = 100, 
    #                                           sample_weight = None)
    # label_validation_gen = mask_datagen.flow(mask_validation, y= None, batch_size = 2, shuffle = True, seed = 100, 
    #                                           sample_weight = None)
    # validation_generator = zip(validation_gen, label_validation_gen)
    #validation_generator = zip(images_validation,mask_validation)
    
                                              
    model = r2udensenet()
    weight_directory = 'weights'
    if not os.path.exists(weight_directory):
        os.mkdir(weight_directory)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_directory,'2dUnet.hdf5'), monitor = 'loss', verbose = 1, save_best_only=True)
    
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    logger = CSVLogger(os.path.join(log_directory,'log.csv'), separator = ',', append = False)
    
    start = time()
    history = model.fit_generator(train_generator, steps_per_epoch = 119, epochs = 250, verbose = 1, 
                                  validation_data = validation_generator,
              validation_steps = 9, callbacks = [model_checkpoint, logger])
                                   
        
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    #plt.title('Accuracy Curve',fontweight='bold',fontsize = 20)
    plt.ylabel('Accuracy',fontweight='bold',fontsize = 20)
    plt.xlabel('Epochs',fontweight='bold',fontsize = 20)
    plt.legend(['train','validation'], loc='lower right',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('Model Loss',fontweight='bold',fontsize = 18)
    plt.ylabel('Loss',fontweight='bold',fontsize = 20)
    plt.xlabel('Epochs',fontweight='bold',fontsize = 20)
    plt.legend(['train','validation'], loc='upper right',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    print('===============Training Done==========================')
    print('Time taken for Training to complete =',time()-start,' ms')
    

################################### PREDICTION OF THE NETWORK ###################################      

def predict():
    print('============= Beginning of Prediction ================')
    images_test = load_test_data()
    images_test = images_test.astype('float32')
    
    images_test_mean = np.mean(images_test)
    images_test_std = np.std(images_test)
    images_test = (images_test - images_test_mean)/images_test_std
    
    model = r2udensenet()
    weight_directory = 'weights'
    model.load_weights(os.path.join(weight_directory,'2dUnet.hdf5'))
    masks_test = model.predict(images_test, batch_size=1, verbose =1)    
    masks_test = np.squeeze(masks_test, axis = 3)
    masks_test = np.around(masks_test, decimals = 0)
    masks_test = (masks_test*255.).astype(np.uint8)
    
    pred_directory = './data/test_image_T2/prediction/r2udensenet2/'
    if not os.path.exists(pred_directory):
        os.mkdir(pred_directory)
    
    count = 0
    for i in range(0, masks_test.shape[0]):
        imsave(os.path.join(pred_directory,  str(count) + '_pred' + '.png' ), masks_test[i])
        count = count + 1
    
    print('===========Prediction Done ==============')
    
        
if __name__ == '__main__':
    train()
    predict()