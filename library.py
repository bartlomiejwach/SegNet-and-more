#Libraries used
from keras.models import Sequential, Model
from keras import layers
from keras import backend as K
from keras.layers import Layer
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, add, concatenate, merge, Convolution2D
from keras.layers import Dropout, BatchNormalization, Activation, ZeroPadding2D, Concatenate, Input 
from keras.layers import SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D, UpSampling2D, LeakyReLU, GlobalMaxPooling2D
from keras.layers.core import Activation, Reshape
from layers import resnet_layer, fire_module, Inception_block, conv_block_nf, create_wide_residual_network
import tensorflow as tf
from keras.regularizers import l2
from keras.optimizers import SGD
import numpy as np
import cv2


#Models
#Models will be saved as Model_Name.model

#model_name(x_train, y_train, input_shape, classes, batch_size, epochs, depth/optional)

#AlexNet Model
def AlexNet(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3):

  #Model-Build
  inputs = Input(shape=input_shape)

  x = Conv2D(96, (11, 11), activation='relu', padding='same')(inputs)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  y = Flatten()(x)
  y = Dense(4096, activation='relu')(y)
  y = Dropout(0.4)(y)
  y = Dense(4096, activation='relu')(y)
  y = Dropout(0.4)(y)
  y = Dense(1000, activation='relu')(y)
  y = Dropout(0.4)(y)
  outputs = Dense(classes, activation='softmax')(y)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('AlexNet.model')

#VGG19 Model
def VGG19(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3):

  inputs = Input(shape=input_shape)

  x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  y = Flatten()(x)
  y = Dense(4096, activation='relu')(y)
  y = Dense(4096, activation='relu')(y)
  outputs = Dense(classes, activation='softmax')(y)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('VGG19.model')

#VGG16 Model
def VGG16(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3):

  inputs = Input(shape=input_shape)

  x = Conv2D(64, (3, 3),activation='relu',padding='same')(inputs)
  x = Conv2D(64, (3, 3),activation='relu',padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(128, (3, 3),activation='relu',padding='same')(x)
  x = Conv2D(128, (3, 3),activation='relu',padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(256, (3, 3),activation='relu',padding='same')(x)
  x = Conv2D(256, (3, 3),activation='relu',padding='same')(x)
  x = Conv2D(256, (3, 3),activation='relu',padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(512, (3, 3),activation='relu',padding='same')(x)
  x = Conv2D(512, (3, 3),activation='relu',padding='same')(x)
  x = Conv2D(512, (3, 3),activation='relu',padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(512, (3, 3),activation='relu',padding='same')(x)
  x = Conv2D(512, (3, 3),activation='relu',padding='same')(x)
  x = Conv2D(512, (3, 3),activation='relu',padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  y = Flatten()(x)
  y = Dense(4096, activation='relu')(y)
  y = Dense(4096, activation='relu')(y)
  outputs = Dense(classes, activation='softmax')(y)

  # Instantiate model.
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('VGG16.model')

#ResNet_1 Model
def ResNet_1(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3, depth=20):

  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  inputs = Input(shape=input_shape)
  x = resnet_layer(inputs=inputs)

  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1

      if stack > 0 and res_block == 0:  
        strides = 2
      y = resnet_layer(inputs=x,num_filters=num_filters,strides=strides)
      y = resnet_layer(inputs=y,num_filters=num_filters,activation=None)
      x = resnet_layer(inputs=x,num_filters=num_filters,kernel_size=1,strides=strides,activation=None,batch_normalization=False)
      x = add([x, y])
      x = Activation('relu')(x)
    num_filters *= 2

  x = AveragePooling2D(pool_size=8)(x)
  y = Flatten()(x)
  outputs = Dense(classes,
                  activation='softmax',
                  kernel_initializer='he_normal')(y)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('ResNet_1.model')

#ResNet_2 Model
def ResNet_2(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3, depth=20):

  if (depth - 2) % 9 != 0:
    raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
  num_filters_in = 16
  num_res_blocks = int((depth - 2) / 9)

  inputs = Input(shape=input_shape)
  x = resnet_layer(inputs=inputs,
                    num_filters=num_filters_in,
                    conv_first=True)

  for stage in range(3):
      for res_block in range(num_res_blocks):
          activation = 'relu'
          batch_normalization = True
          strides = 1
          if stage == 0:
              num_filters_out = num_filters_in * 4
              if res_block == 0:
                  activation = None
                  batch_normalization = False
          else:
              num_filters_out = num_filters_in * 2
              if res_block == 0:
                  strides = 2

          y = resnet_layer(inputs=x,
                            num_filters=num_filters_in,
                            kernel_size=1,
                            strides=strides,
                            activation=activation,
                            batch_normalization=batch_normalization,
                            conv_first=False)
          y = resnet_layer(inputs=y,
                            num_filters=num_filters_in,
                            conv_first=False)
          y = resnet_layer(inputs=y,
                            num_filters=num_filters_out,
                            kernel_size=1,
                            conv_first=False)
          if res_block == 0:

              x = resnet_layer(inputs=x,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                strides=strides,
                                activation=None,
                                batch_normalization=False)
          x = add([x, y])

      num_filters_in = num_filters_out

  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = AveragePooling2D(pool_size=8)(x)
  y = Flatten()(x)
  outputs = Dense(classes,activation='softmax',kernel_initializer='he_normal')(y)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('ResNet_2.model')

#SqueezeNet Model
def SqueezeNet(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3):

  inputs = Input(shape=input_shape)
  x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
  x = Activation('relu', name='relu_conv1')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
  
  x = fire_module(x, fire_id=2, squeeze=16, expand=64)
  x = fire_module(x, fire_id=3, squeeze=16, expand=64)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
  
  x = fire_module(x, fire_id=4, squeeze=32, expand=128)
  x = fire_module(x, fire_id=5, squeeze=32, expand=128)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
  
  x = fire_module(x, fire_id=6, squeeze=48, expand=192)
  x = fire_module(x, fire_id=7, squeeze=48, expand=192)
  x = fire_module(x, fire_id=8, squeeze=64, expand=256)
  x = fire_module(x, fire_id=9, squeeze=64, expand=256)
  x = Dropout(0.5, name='drop9')(x)
  
  x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
  x = Activation('relu', name='relu_conv10')(x)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Activation('softmax', name='loss')(x)
  x = Flatten()(x)
  
  model = Model(inputs, x)

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('SqueezeNet.model')

#GoogleNet Model
def GoogleNet(x_train, y_train, input_shape=[244,244,3], classes=10, batch_size=32, epochs=3):

  # input layer 
  input_layer = Input(shape=input_shape)

  # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
  X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(input_layer)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # convolutional layer: filters = 64, strides = 1
  X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)

  # convolutional layer: filters = 192, kernel_size = (3,3)
  X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 1st Inception block
  X = Inception_block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)

  # 2nd Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 3rd Inception block
  X = Inception_block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)

  # Extra network 1:
  X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
  X1 = Flatten()(X1)
  X1 = Dense(1024, activation = 'relu')(X1)
  X1 = Dropout(0.7)(X1)
  X1 = Dense(5, activation = 'softmax')(X1)
  X1 = Flatten()(X1)

  # 4th Inception block
  X = Inception_block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 5th Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 6th Inception block
  X = Inception_block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)

  # Extra network 2:
  X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
  X2 = Flatten()(X2)
  X2 = Dense(1024, activation = 'relu')(X2)
  X2 = Dropout(0.7)(X2)
  X2 = Dense(1000, activation = 'softmax')(X2)
  X2 = Flatten()(X2)
  
  # 7th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, 
                      f3_conv5 = 128, f4 = 128)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # 8th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)

  # 9th Inception block
  X = Inception_block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)

  # Global Average pooling layer 
  X = GlobalAveragePooling2D(name = 'GAPL')(X)

  # Dropoutlayer 
  X = Dropout(0.4)(X)

  # output layer 
  X = Dense(classes, activation = 'softmax')(X)
  X = Flatten()(X)

  # model
  model = Model(input_layer, [X, X1, X2])

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('GoogleNet.model')

#ZFNet
def ZFNet(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3):

  #Model-Build
  inputs = Input(shape=input_shape)

  x = Conv2D(96, (7, 7), strides=(2, 2), name='conv1')(inputs)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)
  x = BatchNormalization(axis=3, name='bn_conv1')(x)

  x = Conv2D(256, (5, 5), strides=(4, 4), name='conv2')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2')(x)
  x = BatchNormalization(axis=3, name='bn_conv2')(x)

  x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv3')(x)
  x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv4')(x)
  x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv5')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)

  y = Dense(4096)(x)
  y = Dense(4096)(x)
  y = Dense(classes)(x)
  outputs = Activation('softmax')(x)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('ZFNet.model')

#NFNet_F2
def NFNet_F2(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3):

  #Model-Build
  inputs = Input(shape=input_shape)

  for _ in range(2):
    x = conv_block_nf(inputs, 32)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  for _ in range(2):
    x = conv_block_nf(inputs, 64)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = GlobalAveragePooling2D()(x)
  outputs = Dense(classes, activation="softmax")(x)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('NFNet_F2.model')

#ColorNet
def ColorNet(x_train, y_train, input_shape=[125,125,3], classes=10, batch_size=32, epochs=3):

  #Model-Build

  inputs = Input(shape=input_shape)

  x = Conv2D(16, (3, 3),activation="relu")(inputs)
  x = Conv2D(16, (5, 5),activation="relu")(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(32, (3, 3),activation="relu")(x)
  x = Conv2D(32, (5, 5),activation="relu")(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Conv2D(64, (3, 3),activation="relu",kernel_initializer="he_uniform")(x)
  x = Conv2D(64, (5, 5),activation="relu")(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Conv2D(128, (3, 3),activation="relu",kernel_initializer="he_uniform")(x)
  x = Conv2D(128, (5, 5),activation="relu")(x)
  x = MaxPooling2D(pool_size=(3, 3))(x)

  y = Flatten()(x)
  y = Dense(256,activation="relu",kernel_regularizer=l2(0.001),activity_regularizer=l2(0.001))(y)
  y = Dropout(0.5)(y)
  y = Dense(256,activation="relu",kernel_regularizer=l2(0.001),activity_regularizer=l2(0.001))(y)
  y = Dropout(0.5)(y)
  outputs = Dense(classes, activation="softmax")(y)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('ColorNet.model')

#WideResNet
def WideResNet(x_train, y_train, input_shape=[125,125,3], classes=10, batch_size=32, epochs=3, depth=28):

  #Model-Build

  inputs = Input(shape=input_shape)

  x = create_wide_residual_network(classes, inputs, depth)

  model = Model(inputs=inputs, outputs=x)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('WideResNet.model')

