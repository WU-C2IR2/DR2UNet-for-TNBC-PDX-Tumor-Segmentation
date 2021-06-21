# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:24:29 2020

@author: kdutta01
"""


#from model import unet
from model2D import denseunet
from unet import unet_model
from res_unet import resunet
#from model_bcdu_net import BCDU_net_D3
#from model_multi_resnet import MultiResUnet
#from res_unet import resunet
#from r2unet import r2unet

from new_r2udensenet import r2udensenet
from new_r2unet import r2unet
import scipy.io as sio
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import binarize

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

accuracy = []

#################### Training of the Network by Five Fold Cross Validation #################################

def train():
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

    def get_model_name():
        return 'model_r2udensenet.hdf5'
    
    def get_log_name():
        return 'log_r2udensenet.csv'

    
    trainData_T2 = images_train
    trainData_T1 = images_train_T1
    trainMask = mask_train
    
        
    train_generator = generate_generator(image_datagen, mask_datagen, trainData_T2, trainMask, trainData_T1)
    
    model = r2udensenet()
    weight_directory = 'weights_new'
    if not os.path.exists(weight_directory):
        os.mkdir(weight_directory)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_directory,get_model_name()), monitor = 'loss', verbose = 1, save_best_only=True)
    
    log_directory = 'logs_new'
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    logger = CSVLogger(os.path.join(log_directory,get_log_name()), separator = ',', append = False)
    
    start = time()
    history = model.fit_generator(train_generator, steps_per_epoch = len(trainData_T2)/2, epochs = 250, verbose = 1,
                                   callbacks = [model_checkpoint, logger])
    
    
    plt.plot(history.history['recall'], history.history['precision'])
    plt.ylabel('Precision',fontweight='bold',fontsize = 20)
    plt.xlabel('Recall',fontweight='bold',fontsize = 20)
    # #plt.legend(['train','validation'], loc='lower right',fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('precision_recall_train1.png')
    plt.plot(history.history['auc'])
    plt.plot(history.history['pr_auc'])
    plt.legend(['auc','pr_auc'], loc='lower right',fontsize=16)
    plt.savefig('auc.png')
   

# def gen_precision_recall():
#     print('====== Generate Precision Recall Curve ========')
    
#     images_train, images_train_T1, mask_train = load_train_data()
#     images_train = images_train.astype('float32')
#     images_train_T1 = images_train_T1.astype('float32')
#     mask_train = mask_train.astype('float32')

#     images_train_mean = np.mean(images_train)
#     images_train_std = np.std(images_train)
#     images_train = (images_train - images_train_mean)/images_train_std

#     images_train_mean = np.mean(images_train_T1)
#     images_train_std = np.std(images_train_T1)
#     images_train_T1 = (images_train_T1 - images_train_mean)/images_train_std
#     mask_train /= 255.
#     mask_train = mask_train.ravel().reshape(-1,1)
#     mask_train = binarize(mask_train, threshold = 0.5)
    
    
#     images_train_final = np.append(images_train, images_train_T1, axis = 3)
    
#     model = r2udensenet()
#     weight_directory = 'weights_new'
#     model.load_weights(os.path.join(weight_directory,'model_r2udensenet_retest.hdf5'))
#     y_pred = model.predict(images_train_final, batch_size = 1, verbose =1)  
#     y_pred = np.squeeze(y_pred, axis = 3)
    
#     y_test = mask_train
#     nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = precision_recall_curve(y_test.ravel(), y_pred.ravel())
#     diff = np.abs(nn_fpr_keras-nn_tpr_keras)
#     place = np.argmin(diff)
#     print('Optimal Threshold = ', str(nn_thresholds_keras[place]))
#     print('Recall = ',nn_fpr_keras[place])
#     print('Precision = ',nn_tpr_keras[place])
#     plt.plot(nn_tpr_keras, nn_fpr_keras)
#     plt.ylabel('Precision',fontweight='bold',fontsize = 20)
#     plt.xlabel('Recall',fontweight='bold',fontsize = 20)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.savefig('precision_recall.png')
#     opt_thresh = nn_thresholds_keras[place]
#     return opt_thresh
    
    
    

# ################################### PREDICTION OF THE NETWORK ###################################      

# def predict(opt_thresh = 0.5):
#     print('============= Beginning of Prediction ================')
#     images_test = load_test_data()
#     images_test = images_test.astype('float32')
    
#     images_test_T2 = images_test[:,:,:,0]
#     images_test_T1 = images_test[:,:,:,1]
    
#     images_test_mean = np.mean(images_test_T2)
#     images_test_std = np.std(images_test_T2)
#     images_test_T2 = (images_test_T2 - images_test_mean)/images_test_std
#     images_test_T2 = np.expand_dims(images_test_T2, axis = 3)
#     #print(images_test_T2.shape)
    
#     images_test_mean = np.mean(images_test_T1)
#     images_test_std = np.std(images_test_T1)
#     images_test_T1 = (images_test_T1 - images_test_mean)/images_test_std
#     images_test_T1 = np.expand_dims(images_test_T1, axis = 3)
#     #print(images_test_T1.shape)
    
#     images_test = np.append(images_test_T2 ,images_test_T1, axis = 3)
    
#     masks_gt = sio.loadmat('staple_test_gt.mat', matlab_compatible = True)['staple_gt']
#     masks_gt = np.swapaxes(masks_gt,0,2)
#     masks_gt = masks_gt
    
#     model = r2udensenet()
#     weight_directory = 'weights_new'
#     model.load_weights(os.path.join(weight_directory,'model_r2udensenet_retest.hdf5'))
#     masks_test = model.predict(images_test, batch_size=1, verbose =1)    
#     masks_test = np.squeeze(masks_test, axis = 3)
#     print(masks_test.shape)
#     masks_test = np.around(masks_test, decimals = 0)
#     masks_pred  = masks_test
    
#     masks_test = (masks_test*255.).astype(np.uint8)
    
#     pred_directory = './output/prediction/r2udensenet_retest/'
#     if not os.path.exists(pred_directory):
#         os.mkdir(pred_directory)
    
#     count = 0
    
#     dice = []
#     precision = []
#     recall = []
#     auc = []
#     print('Threshold', opt_thresh)
#     for i in range(0, masks_test.shape[0]):
#         imsave(os.path.join(pred_directory,  str(count) + '_pred' + '.png' ), masks_test[i])
#         count = count + 1
#         gt = masks_gt[i,:,:].T
#         gt = binarize(gt, threshold = 0.5)
#         # plt.imshow(gt)
#         # plt.show()
        
#         test = masks_pred[i,:,:]
#         # plt.imshow(test)
#         # plt.show()
        
#         test[test >= opt_thresh] = 1
#         #print(test.shape)
        
#         dice1 = f1_score(gt.flatten(), test.flatten(), average = 'binary')
#         dice.append(dice1)
        
#         precision1 = precision_score(gt.flatten(), test.flatten(), average = 'binary')
#         precision.append(precision1)
        
#         recall1 = recall_score(gt.flatten(), test.flatten(), average = 'binary')
#         recall.append(recall1)
        
#         auc1 = roc_auc_score(gt.flatten(), test.flatten())
#         auc.append(auc1)
        
#     dice_mean = np.mean(dice)
#     dice_std = np.std(dice)
    
#     prec_mean = np.mean(precision)
#     prec_std = np.std(precision)
    
#     recall_mean = np.mean(recall)
#     recall_std = np.std(recall)
    
#     auc_mean = np.mean(auc)
#     auc_std = np.std(auc)
    
    
#     print('Dice Mean = ', str(dice_mean))
#     print('Standard Dev Dice = ', str(dice_std))
    
#     print('Precision Mean = ', str(prec_mean))
#     print('Standard Dev Precision = ', str(prec_std))
    
#     print('Recall Mean = ', str(recall_mean))
#     print('Standard Dev Recall = ', str(recall_std))
    
#     print('AUC Mean = ', str(auc_mean))
#     print('Standard Dev AUC = ', str(auc_std))
        
#     #print('Dice  = ', str(np.mean(dice))
#     #print('Dice Std = ', str(np.std(dice))
#     print('============= End of Prediction ================')
    
         
    
        
if __name__ == '__main__':
    train()
    # opt_thresh = gen_precision_recall()
    # predict(opt_thresh)