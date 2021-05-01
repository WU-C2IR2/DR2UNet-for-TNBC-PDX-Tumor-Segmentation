# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:49:39 2020

@author: kdutta01
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K


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


# def tversky_loss(beta):
#   def loss(y_true, y_pred):
#     numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
#     denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

#     return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

#   return loss

"""Recurrent Layer"""
def rec_layer(layer, filters):
    reconv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(layer)
    reconc1 = concatenate([layer, reconv1], axis=3)
    drop_inter = Dropout(0.3)(reconc1)
    reconv2 = Conv2D(filters, (3, 3), activation='relu', padding='same')(drop_inter)
    reconc2 = concatenate([layer, reconv2], axis=3)
    return reconc2
    

########## Initialization of Parameters #######################
image_row = 128
image_col = 128
image_depth = 2

def r2udensenet():
    inputs = Input((image_row, image_col, image_depth))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    reconv1 = rec_layer(conv1, 32)
    concinter1 = concatenate([inputs,reconv1], axis=3)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(concinter1)
    reconv1 = rec_layer(conv1, 32)
    conc1 = concatenate([inputs, reconv1], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    reconv2 = rec_layer(conv2, 64)
    concinter2 = concatenate([conv2,reconv2], axis=3)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(concinter2)
    reconv2 = rec_layer(conv2, 64)
    conc2 = concatenate([pool1, reconv2], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    reconv3 = rec_layer(conv3, 128)
    concinter3 = concatenate([conv3,reconv3], axis=3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(concinter3)
    reconv3 = rec_layer(conv3, 128)
    conc3 = concatenate([pool2, reconv3], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    reconv4 = rec_layer(conv4, 256)
    concinter3 = concatenate([conv4,reconv4], axis=3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(concinter3)
    reconv4 = rec_layer(conv4, 256)
    conc4 = concatenate([pool3, reconv4], axis=3)
    drop4 = Dropout(0.5)(conc4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    reconv5 = rec_layer(conv5, 512)
    concinter3 = concatenate([conv5,reconv5], axis=3)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(concinter3)
    reconv5 = rec_layer(conv5, 512)
    conc5 = concatenate([pool4, reconv5], axis=3)
    drop5 = Dropout(0.5)(conc5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    reconv6 = rec_layer(conv6, 256)
    concinter6 = concatenate([conv6,reconv6], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(concinter6)
    reconv6 = rec_layer(conv6,256)
    conc6 = concatenate([up6, reconv6], axis=3)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    reconv7 = rec_layer(conv7, 128)
    concinter7 = concatenate([conv7,reconv7], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(concinter7)
    reconv7 = rec_layer(conv7, 128)
    conc7 = concatenate([up7, reconv7], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    reconv8 = rec_layer(conv8, 64)
    concinter8 = concatenate([conv8,reconv8], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(concinter8)
    reconv8 = rec_layer(conv8, 64)
    conc8 = concatenate([up8, reconv8], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    reconv9 = rec_layer(conv9, 32)
    concinter9 = concatenate([conv9,reconv9], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(concinter9)
    reconv9 = rec_layer(conv9, 32)
    conc9 = concatenate([up9, reconv9], axis=3)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc9)
    #reconv10 = rec_layer(conv10, 1)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.summary()

    model.compile(optimizer=Adam(lr=1e-5), loss= dice_loss, metrics=[dice_coef])
    
    pretrained_weights = None

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model