#Libraries used
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, add, Convolution2D, Dropout, LSTM
from keras.layers import Dropout, BatchNormalization, Activation, Input
from keras.layers import GlobalAveragePooling2D, AveragePooling2D
from keras.layers.core import Activation, Reshape
from layers import resnet_layer, fire_module, Inception_block, conv_block_nf, create_wide_residual_network
import pandas as pd
from keras.regularizers import l2
import numpy
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv


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
def WideResNet(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3, depth=28):

  #Model-Build

  inputs = Input(shape=input_shape)

  x = create_wide_residual_network(classes, inputs, depth)

  model = Model(inputs=inputs, outputs=x)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('WideResNet.model')

#LSTM_Net_text
def LSTM_Net_text(filepath, batch_size=128, epochs=3):

  filename = filepath
  raw_text = open(filename, 'r', encoding='utf-8').read()
  raw_text = raw_text.lower()
  chars = sorted(list(set(raw_text)))
  char_to_int = dict((c, i) for i, c in enumerate(chars))
  n_chars = len(raw_text)
  n_vocab = len(chars)
  seq_length = 100
  dataX = []
  dataY = []
  for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
  n_patterns = len(dataX)
  X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
  # normalize
  x_train = X / float(n_vocab)
  # one hot encode the output variable
  y_train = np_utils.to_categorical(dataY)

  model = Sequential()
  model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(y_train.shape[1], activation='softmax'))

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('LSTM_Net_text.model')

#LSTM_big_Net_text
def LSTM_big_Net_text(filepath, batch_size=128, epochs=3):

  filename = filepath
  raw_text = open(filename, 'r', encoding='utf-8').read()
  raw_text = raw_text.lower()
  chars = sorted(list(set(raw_text)))
  char_to_int = dict((c, i) for i, c in enumerate(chars))
  n_chars = len(raw_text)
  n_vocab = len(chars)
  seq_length = 100
  dataX = []
  dataY = []
  for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
  n_patterns = len(dataX)
  X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
  # normalize
  x_train = X / float(n_vocab)
  # one hot encode the output variable
  y_train = np_utils.to_categorical(dataY)

  model = Sequential()
  model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(256))
  model.add(Dropout(0.2))
  model.add(Dense(y_train.shape[1], activation='softmax'))

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('LSTM_big_Net_text.model')

#LSTM_Net_time_series
def LSTM_Net_time_series(filepath, time_steps=1, batch_size=1, epochs=3):

  def create_dataset(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_steps-1):
      a = dataset[i:(i+time_steps), 0]
      dataX.append(a)
      dataY.append(dataset[i + time_steps, 0])
    return numpy.array(dataX), numpy.array(dataY)
  # fix random seed for reproducibility
  numpy.random.seed(7)
  # load the dataset
  dataframe = read_csv(filepath, usecols=[1], engine='python')
  dataset = dataframe.values
  dataset = dataset.astype('float32')
  # normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  # split into train and test sets
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  # reshape into X=t and Y=t+1
  time_steps = 1
  x_train, y_train = create_dataset(train, time_steps)
  testX, testY = create_dataset(test, time_steps)
  # reshape input to be [samples, time steps, features]
  x_train = numpy.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
  testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  model = Sequential()
  model.add(LSTM(4, input_shape=(1, time_steps)))
  model.add(Dense(1))

  model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('LSTM_Net_time_series.model')

def Stock_Net(filepath, batch_size=16, epochs=3):

  def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset)-step_size-1):
      a = dataset[i:(i+step_size), 0]
      data_X.append(a)
      data_Y.append(dataset[i + step_size, 0])
    return numpy.array(data_X), numpy.array(data_Y)

  numpy.random.seed(7)

  # IMPORTING DATASET 
  dataset = pd.read_csv(filepath, usecols=[1,2,3,4])
  dataset = dataset.reindex(index = dataset.index[::-1])

  # CREATING OWN INDEX FOR FLEXIBILITY
  obs = numpy.arange(1, len(dataset) + 1, 1)

  # TAKING DIFFERENT INDICATORS FOR PREDICTION
  OHLC_avg = dataset.mean(axis = 1)
  HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
  close_val = dataset[['Close']]

  # PREPARATION OF TIME SERIES DATASE
  OHLC_avg = numpy.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
  scaler = MinMaxScaler(feature_range=(0, 1))
  OHLC_avg = scaler.fit_transform(OHLC_avg)

  # TRAIN-TEST SPLIT
  train_OHLC = int(len(OHLC_avg) * 0.75)
  test_OHLC = len(OHLC_avg) - train_OHLC
  train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

  # TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
  trainX, trainY = new_dataset(train_OHLC, 1)
  testX, testY = new_dataset(test_OHLC, 1)

  # RESHAPING TRAIN AND TEST DATA
  trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  step_size = 1

  # LSTM MODEL
  model = Sequential()
  model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
  model.add(LSTM(16))
  model.add(Dense(1))
  model.add(Activation('linear'))

  # MODEL COMPILING AND TRAINING
  model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
  model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

  model.save('Stock_Net.model')
