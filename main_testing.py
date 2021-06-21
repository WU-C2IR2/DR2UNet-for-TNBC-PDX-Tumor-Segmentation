# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:24:29 2020

@author: kdutta01
"""


from dense_unet import denseunet
from unet import unet_model
from res_unet import resunet
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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import binarize

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gen_precision_recall():
    print('====== Generate Precision Recall Curve ========')
    
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
    mask_train = mask_train.ravel().reshape(-1,1)
    mask_train = binarize(mask_train, threshold = 0.5)
    
    
    images_train_final = np.append(images_train, images_train_T1, axis = 3)
    
    model = r2udensenet()
    weight_directory = 'weights_new'
    model.load_weights(os.path.join(weight_directory,'model_r2udensenet1.hdf5'))
    y_pred = model.predict(images_train_final, batch_size = 1, verbose =1)  
    y_pred = np.squeeze(y_pred, axis = 3)
    
    y_test = mask_train
    nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = precision_recall_curve(y_test.ravel(), y_pred.ravel())
    diff = np.abs(nn_fpr_keras-nn_tpr_keras)
    place = np.argmin(diff)
    print('Optimal Threshold = ', str(nn_thresholds_keras[place]))
    print('Recall = ',nn_fpr_keras[place])
    print('Precision = ',nn_tpr_keras[place])
    plt.plot(nn_tpr_keras, nn_fpr_keras)
    plt.ylabel('Precision',fontweight='bold',fontsize = 20)
    plt.xlabel('Recall',fontweight='bold',fontsize = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('precision_recall.png')
    opt_thresh = nn_thresholds_keras[place]
    return opt_thresh
    
    
    

################################### PREDICTION OF THE NETWORK ###################################      

def predict(opt_thresh = 0.5):
    print('============= Beginning of Prediction ================')
    images_test = load_test_data()
    images_test = images_test.astype('float32')
    
    images_test_T2 = images_test[:,:,:,0]
    images_test_T1 = images_test[:,:,:,1]
    
    images_test_mean = np.mean(images_test_T2)
    images_test_std = np.std(images_test_T2)
    images_test_T2 = (images_test_T2 - images_test_mean)/images_test_std
    images_test_T2 = np.expand_dims(images_test_T2, axis = 3)
    #print(images_test_T2.shape)
    
    images_test_mean = np.mean(images_test_T1)
    images_test_std = np.std(images_test_T1)
    images_test_T1 = (images_test_T1 - images_test_mean)/images_test_std
    images_test_T1 = np.expand_dims(images_test_T1, axis = 3)
    #print(images_test_T1.shape)
    
    images_test = np.append(images_test_T2 ,images_test_T1, axis = 3)
    
    masks_gt = sio.loadmat('staple_test_gt.mat', matlab_compatible = True)['staple_gt']
    masks_gt = np.swapaxes(masks_gt,0,2)
    masks_gt = masks_gt
    
    model = r2udensenet()
    weight_directory = 'weights_new'
    model.load_weights(os.path.join(weight_directory,'model_r2udensenet1.hdf5'))
    masks_test = model.predict(images_test, batch_size=1, verbose =1)    
    masks_test = np.squeeze(masks_test, axis = 3)
    print(masks_test.shape)
    masks_test = np.around(masks_test, decimals = 0)
    masks_pred  = masks_test
    
    masks_test = (masks_test*255.).astype(np.uint8)
    
    pred_directory = './output/r2udensenet/'
    if not os.path.exists(pred_directory):
        os.mkdir(pred_directory)
    
    count = 0
    
    dice = []
    precision = []
    recall = []
    auc = []
    print('Threshold', opt_thresh)
    for i in range(0, masks_test.shape[0]):
        imsave(os.path.join(pred_directory,  str(count) + '_pred' + '.png' ), masks_test[i])
        count = count + 1
        gt = masks_gt[i,:,:].T
        gt = binarize(gt, threshold = 0.5)
        # plt.imshow(gt)
        # plt.show()
        
        test = masks_pred[i,:,:]
        test[test >= opt_thresh] = 1
        
        dice1 = f1_score(gt.flatten(), test.flatten(), average = 'binary')
        dice.append(dice1)
        
        precision1 = precision_score(gt.flatten(), test.flatten(), average = 'binary')
        precision.append(precision1)
        
        recall1 = recall_score(gt.flatten(), test.flatten(), average = 'binary')
        recall.append(recall1)
        
        auc1 = roc_auc_score(gt.flatten(), test.flatten())
        auc.append(auc1)
        
        
    dice_mean = np.mean(dice)
    dice_std = np.std(dice)
    
    prec_mean = np.mean(precision)
    prec_std = np.std(precision)
    
    recall_mean = np.mean(recall)
    recall_std = np.std(recall)
    
    auc_mean = np.mean(auc)
    auc_std = np.std(auc)
    
    print('Dice Mean = ', str(dice_mean))
    print('Standard Dev Dice = ', str(dice_std))
    
    print('Precision Mean = ', str(prec_mean))
    print('Standard Dev Precision = ', str(prec_std))
    
    print('Recall Mean = ', str(recall_mean))
    print('Standard Dev Recall = ', str(recall_std))
    
    print('AUC Mean = ', str(auc_mean))
    print('Standard Dev AUC = ', str(auc_std))

    print('============= End of Prediction ================')
    
    return dice


if __name__ == '__main__':
    opt_thresh = gen_precision_recall()
    dice = predict(opt_thresh)