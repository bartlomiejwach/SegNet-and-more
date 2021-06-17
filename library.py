###Libraries used
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, add, Convolution2D, Dropout, LSTM, Permute
from keras.layers import Dropout, BatchNormalization, Activation, Input, Lambda, Dot, Softmax
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Bidirectional
from keras.layers.core import Activation, Reshape
from layers import resnet_layer, fire_module, conv_block_nf, create_wide_residual_network
import pandas as pd
from keras.regularizers import l2
import numpy
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from tensorflow import squeeze


###Models
###Models will be saved as Model_Name.model

def AlexNet(train_ds, classes=10, batch_size=32, epochs=3): #227x227

  model = Sequential([
  Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
  BatchNormalization(),
  MaxPooling2D(pool_size=(3,3), strides=(2,2)),
  Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
  BatchNormalization(),
  MaxPooling2D(pool_size=(3,3), strides=(2,2)),
  Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
  BatchNormalization(),
  Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
  BatchNormalization(),
  Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
  BatchNormalization(),
  MaxPooling2D(pool_size=(3,3), strides=(2,2)),
  Flatten(),
  Dense(4096, activation='relu'),
  Dropout(0.5),
  Dense(4096, activation='relu'),
  Dropout(0.5),
  Dense(classes, activation='softmax')
  ])
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('AlexNet.model')

def VGG19(train_ds, classes=10, batch_size=32, epochs=3): #224x224

  inputs = Input(shape=[112,112,3])

  x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)

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
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('VGG19.model')

def VGG16(train_ds, classes=10, batch_size=32, epochs=3): #224x224

  inputs = Input(shape=[224,224,3])

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

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('VGG16.model')

def ResNet_1(train_ds, classes=10, batch_size=32, epochs=3, depth=20): #32x32

  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  inputs = Input(shape=[32,32,3])
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
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('ResNet_1.model')

def ResNet_2(train_ds, classes=10, batch_size=32, epochs=3, depth=20): #32x32

  if (depth - 2) % 9 != 0:
    raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
  num_filters_in = 16
  num_res_blocks = int((depth - 2) / 9)

  inputs = Input(shape=[32,32,3])
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
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('ResNet_2.model')

def SqueezeNet(train_ds, classes=10, batch_size=32, epochs=3): #227x227

  inputs = Input(shape=[48,48,3])
  x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid')(inputs)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  
  x = fire_module(x, fire_id=2, squeeze=16, expand=64)
  x = fire_module(x, fire_id=3, squeeze=16, expand=64)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  
  x = fire_module(x, fire_id=4, squeeze=32, expand=128)
  x = fire_module(x, fire_id=5, squeeze=32, expand=128)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  
  x = fire_module(x, fire_id=6, squeeze=48, expand=192)
  x = fire_module(x, fire_id=7, squeeze=48, expand=192)
  x = fire_module(x, fire_id=8, squeeze=64, expand=256)
  x = fire_module(x, fire_id=9, squeeze=64, expand=256)
  x = Dropout(0.5)(x)
  
  x = Convolution2D(classes, (1, 1), padding='valid')(x)
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Activation('softmax')(x)
  x = Flatten()(x)
  
  model = Model(inputs, x)

  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('SqueezeNet.model')

def LeNet(train_ds, classes=10, batch_size=32, epochs=3): #32x32

  model = Sequential()

  model.add(Conv2D(input_shape=(32,32,3), kernel_size=(5, 5), filters=20, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

  model.add(Conv2D(kernel_size=(5, 5), filters=50,  activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dense(classes, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('LeNet.model')

def ZFNet(train_ds, classes=10, batch_size=32, epochs=3): #32x32

  model = Sequential()

  model.add(Conv2D(input_shape=(32,32,3), kernel_size=(7,7), filters=96, activation='relu',strides= [1,1],padding= 'valid'))# Convolution layer
  model.add(MaxPooling2D(pool_size=(3,3), strides=[2,2]))# Pooling layer
  model.add(BatchNormalization(axis= 1))

  model.add(Conv2D(kernel_size=(5,5), filters=256, activation='relu',strides= [2,2],padding= 'same'))# Convolution layer
  model.add(MaxPooling2D(pool_size=(3,3), strides=[1,1]))# Pooling layer
  model.add(BatchNormalization(axis= 1))

  model.add(Conv2D(kernel_size=(3,3), filters=384,activation='relu',strides= [1,1],padding= 'same'))# Convolution layer
  model.add(BatchNormalization(axis= 1))

  model.add(Conv2D(kernel_size=(3,3), filters=384, activation='relu',strides= [1,1],padding= 'same'))# Convolution layer
  model.add(BatchNormalization(axis= 1))

  model.add(Conv2D(kernel_size=(3,3), filters=256, activation='relu',strides= [1,1],padding= 'same'))# Convolution layer
  model.add(MaxPooling2D(pool_size=(3,3), strides=[1,1]))# Pooling layer
  model.add(BatchNormalization(axis= 1))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5)) 
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5)) 
  model.add(Dense(classes, activation='softmax'))
  
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('ZFNet.model')

def NFNet_F2(train_ds, classes=10, batch_size=32, epochs=3): #32x32

  inputs = Input(shape=[32,32,3])

  for _ in range(2):
    x = conv_block_nf(inputs, 32)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  for _ in range(2):
    x = conv_block_nf(inputs, 64)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = GlobalAveragePooling2D()(x)
  outputs = Dense(classes, activation="softmax")(x)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('NFNet_F2.model')

def ColorNet(train_ds, classes=10, batch_size=32, epochs=3): #125x125

  inputs = Input(shape=[125,125,3])

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
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('ColorNet.model')

def WideResNet(train_ds, classes=10, batch_size=32, epochs=3, depth=28): #32x32

  inputs = Input(shape=[32,32,3])

  x = create_wide_residual_network(classes, inputs, depth)

  model = Model(inputs=inputs, outputs=x)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
  model.fit(train_ds, batch_size=batch_size, epochs=epochs)
  model.save('WideResNet.model')

def LSTM_text(filepath, batch_size=128, epochs=3):

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
  x_train = X / float(n_vocab)
  y_train = np_utils.to_categorical(dataY)

  inputs = Input((x_train.shape[1], x_train.shape[2]))

  x = LSTM(256, return_sequences=True)(inputs)
  x = Dropout(0.2)(x)
  x = LSTM(256)(x)
  x = Dropout(0.2)(x)
  x = Dense(y_train.shape[1], activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)

  model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('LSTM_text.model')

def LSTM_time_series(filepath, time_steps=1, batch_size=1, epochs=3):

  def create_dataset(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_steps-1):
      a = dataset[i:(i+time_steps), 0]
      dataX.append(a)
      dataY.append(dataset[i + time_steps, 0])
    return numpy.array(dataX), numpy.array(dataY)
  numpy.random.seed(7)
  dataframe = read_csv(filepath, usecols=[1], engine='python')
  dataset = dataframe.values
  dataset = dataset.astype('float32')
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  time_steps = 1
  x_train, y_train = create_dataset(train, time_steps)
  testX, testY = create_dataset(test, time_steps)
  x_train = numpy.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
  testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  inputs = Input((1, time_steps))

  x = LSTM(30, return_sequences=True, stateful=False)(inputs)
  x = Activation('relu')(x)
  x = LSTM(30, return_sequences=True, stateful=False)(x)
  x = Activation('relu')(x)
  x = Dense(30)(x)
  x = Activation('relu')(x)
  x = Dense(1)(x)

  model = Model(inputs=inputs, outputs=x)

  model.compile(loss='mean_squared_error',optimizer='adam')
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
  model.save('LSTM_Net_time_series.model')

def LSTM_Stock(filepath, batch_size=16, epochs=3):

  def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset)-step_size-1):
      a = dataset[i:(i+step_size), 0]
      data_X.append(a)
      data_Y.append(dataset[i + step_size, 0])
    return numpy.array(data_X), numpy.array(data_Y)

  numpy.random.seed(7)

  dataset = pd.read_csv(filepath, usecols=[1,2,3,4])
  dataset = dataset.reindex(index = dataset.index[::-1])

  obs = numpy.arange(1, len(dataset) + 1, 1)

  OHLC_avg = dataset.mean(axis = 1)
  HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
  close_val = dataset[['Close']]

  OHLC_avg = numpy.reshape(OHLC_avg.values, (len(OHLC_avg),1))
  scaler = MinMaxScaler(feature_range=(0, 1))
  OHLC_avg = scaler.fit_transform(OHLC_avg)

  train_OHLC = int(len(OHLC_avg) * 0.75)
  test_OHLC = len(OHLC_avg) - train_OHLC
  train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

  trainX, trainY = new_dataset(train_OHLC, 1)
  testX, testY = new_dataset(test_OHLC, 1)

  trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  step_size = 1

  inputs = Input((1, step_size))

  x = LSTM(30, return_sequences=True, stateful=False)(inputs)
  x = Activation('relu')(x)
  x = LSTM(30, return_sequences=True, stateful=False)(x)
  x = Activation('relu')(x)
  x = Dense(30)(x)
  x = Activation('relu')(x)
  x = Dense(1)(x)

  model = Model(inputs=inputs, outputs=x)

  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

  model.save('Stock_Net.model')

def RNN_Speech(x_train, y_train, classes, sampling_rate=16000, input_length=16000, batch_size=32, epochs=3):

  inputs = Input((input_length,))

  x = Reshape((1, -1))(inputs)

  x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, input_length),
                      padding='same', sr=sampling_rate, n_mels=80,
                      fmin=40.0, fmax=sampling_rate / 2, power_melgram=1.0,
                      return_decibel_melgram=True, trainable_fb=False,
                      trainable_kernel=False)(x)

  x = Normalization2D(int_axis=0)(x)

  x = Permute((2, 1, 3))(x)

  x = Conv2D(10, (5, 1), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(1, (5, 1), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)

  x = Lambda(lambda q: squeeze(q, -1))(x)

  x = Bidirectional(LSTM(64, return_sequences=True))(x)
  x = Bidirectional(LSTM(64))(x)

  x = Dense(64, activation='relu')(x)
  x = Dense(32, activation='relu')(x)

  output = Dense(classes, activation='softmax')(x)

  model = Model(inputs=[inputs], outputs=[output])

  model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
  model.fit(x_train, validation_data=y_train, epochs=epochs, batch_size=batch_size,  use_multiprocessing=False, workers=4, verbose=2)

  model.save('RNN_Speech.model')

def Att_RNN_Speech(x_train, y_train, classes, sampling_rate=16000, input_length=16000, batch_size=32, epochs=3):

  inputs = Input((input_length,))

  x = Reshape((1, -1))(inputs)

  m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, input_length),
                      padding='same', sr=sampling_rate, n_mels=80,
                      fmin=40.0, fmax=sampling_rate / 2, power_melgram=1.0,
                      return_decibel_melgram=True, trainable_fb=False,
                      trainable_kernel=False)
  m.trainable = False

  x = m(x)

  x = Normalization2D(int_axis=0)(x)

  x = Permute((2, 1, 3))(x)

  x = Conv2D(10, (5, 1), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(1, (5, 1), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)

  x = Lambda(lambda q: squeeze(q, -1))(x)

  x = Bidirectional(LSTM(64, return_sequences=True))(x)
  x = Bidirectional(LSTM(64, return_sequences=True))(x)

  xFirst = Lambda(lambda q: q[:, -1])(x)
  query = Dense(128)(xFirst)

  attScores = Dot(axes=[1, 2])([query, x])
  attScores = Softmax()(attScores)

  attVector = Dot(axes=[1, 1])([attScores, x])

  x = Dense(64, activation='relu')(attVector)
  x = Dense(32)(x)

  output = Dense(classes, activation='softmax')(x)

  model = Model(inputs=[inputs], outputs=[output])

  model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
  model.fit(x_train, validation_data=y_train, epochs=epochs, batch_size=batch_size,  use_multiprocessing=False, workers=4, verbose=2)

  model.save('Att_RNN_Speech.model')