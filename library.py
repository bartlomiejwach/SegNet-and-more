#Libraries used
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D , Flatten, add
from keras.layers import Dropout, BatchNormalization, Activation, ZeroPadding2D, Concatenate, Input 
from keras.layers import SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D
import tensorflow as tf


#Models
#Models will be saved as Model_Name.model

#model_name(x_train, y_train, input_shape, classes, batch_size, epochs, depth/optional)

#AlexNet Model
def AlexNet(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3, depth=20):

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
def VGG19(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3, depth=20):

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
def VGG16(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3, depth=20):

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