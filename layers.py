from keras.models import Sequential, Model
from keras import layers
from keras import backend as K
from keras.layers import Layer
from keras.layers import Dense, Conv2D, MaxPooling2D , Flatten, add, concatenate
from keras.layers import Dropout, BatchNormalization, Activation, ZeroPadding2D, Concatenate, Input 
from keras.layers import SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D, UpSampling2D, LeakyReLU
from keras.layers.core import Activation, Reshape
import tensorflow as tf
import numpy as np
from tensorflow import keras

def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',batch_normalization=True,conv_first=True):
  
    conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def fire_module(x, fire_id, squeeze=16, expand=64):
    x = Conv2D(squeeze, (1, 1), padding='valid')(x)
    x = Activation('relu')(x)
 
    left = Conv2D(expand, (1, 1), padding='valid')(x)
    left = Activation('relu')(left)
 
    right = Conv2D(expand, (3, 3), padding='same')(x)
    right = Activation('relu')(right)
 
    x = concatenate([left, right], axis=3)
    return x