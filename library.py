#Libraries used
from keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, Activation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

#GPU Init
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#AlexNet Model
def AlexNet(train_data, epochs, validation_data, validation_freq):

    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, input_shape=(32,32,3,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    #model.fit(train_ds, epochs=1, validation_data=validation_ds, validation_freq=1)
    model.fit(train_data, epochs=epochs, validation_data=validation_data, validation_freq=validation_freq)

    model.save('new.model')