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

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 

  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer

def conv_block_nf(x, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
               kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)
    return x