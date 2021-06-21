# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:25:07 2020

@author: kdutta01
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Dropout, Add
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

def resunet():
    inputs = Input((image_row, image_col, image_depth))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    input_new = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    conc1 = Add()([input_new, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    input_new = Conv2D(64, (1, 1), activation='relu', padding='same')(pool1)
    conc2 = Add()([input_new, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    input_new = Conv2D(128, (1, 1), activation='relu', padding='same')(pool2)
    conc3 = Add()([input_new, conv3])
    drop3 = Dropout(0.5)(conc3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    input_new = Conv2D(256, (1, 1), activation='relu', padding='same')(pool3)
    conc4 = Add()([input_new, conv4])
    drop4 = Dropout(0.5)(conc4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    input_new = Conv2D(512, (1, 1), activation='relu', padding='same')(pool4)
    conc5 = Add()([input_new, conv5])

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    input_new = Conv2D(256, (1, 1), activation='relu', padding='same')(up6)
    conc6 = Add()([input_new, conv6])

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    input_new = Conv2D(128, (1, 1), activation='relu', padding='same')(up7)
    conc7 = Add()([input_new, conv7])

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    input_new = Conv2D(64, (1, 1), activation='relu', padding='same')(up8)
    conc8 = Add()([input_new, conv8])

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    input_new = Conv2D(32, (1, 1), activation='relu', padding='same')(up9)
    conc9 = Add()([input_new, conv9])

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.summary()
    

    model.compile(optimizer=Adam(lr=1e-5), loss= dice_loss, metrics=METRICS)
    
    pretrained_weights = None

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
