# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:37:19 2020

@author: kdutta01
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy

K.set_image_data_format('channels_last')
smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1 -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


pr_metric = AUC(curve='PR', num_thresholds=10, name = 'pr_auc') 
roc_metric = AUC(name = 'auc')
METRICS = [dice_coef,
      Precision(name='precision'),
      Recall(name='recall'),
      pr_metric, roc_metric
]    

########## Initialization of Parameters #######################
image_row = 128
image_col = 128
image_depth = 2

def unet_model():
    inputs = Input((image_row, image_col, image_depth))
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)

    conv21 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)

    conv31 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv32 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv31)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv32)

    conv41 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv42 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv41)
    drop4 = Dropout(0.5)(conv42)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    

    conv51 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv52 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv51)
    drop5 = Dropout(0.5)(conv52)
    #drop5 = conv52

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5), conv42], axis=3)
    conv61 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv62 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv61)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv62), conv32], axis=3)
    conv71 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv72 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv71)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv72), conv22], axis=3)
    conv81 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv82 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv81)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv82), conv12], axis=3)
    conv91 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv92 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv91)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv92)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.summary()

    model.compile(optimizer = Adam(lr=1e-5),
                  loss = dice_loss, metrics=METRICS)
    
    pretrained_weights = None

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)



    return model




