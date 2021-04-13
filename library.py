#Libraries used
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, Activation
import tensorflow as tf

#GPU Init

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
CUDA_VISIBLE_DEVICES=1


#Models
#Models will be saved as Model_Name.model

#AlexNet Model
def AlexNet(train_data, validation_data, epochs, validation_freq, batch_size):

  #Image Processing
  def process_images(image, label):
    #Resize
    image = tf.image.per_image_standardization(image)
    #Resize
    image = tf.image.resize(image, (227,227))
    return image, label

  train_data_size = tf.data.experimental.cardinality(train_data).numpy()
  validation_data_size = tf.data.experimental.cardinality(validation_data).numpy()

  train_data = (train_data.map(process_images).shuffle(buffer_size=train_data_size).batch(batch_size=32, drop_remainder=True))
  validation_data = (validation_data.map(process_images).shuffle(buffer_size=train_data_size).batch(batch_size=32, drop_remainder=True))

  #End of Image Processing

  #Model-Build
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

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  model.fit(train_data, epochs=epochs, validation_data=validation_data, validation_freq=validation_freq, batch_size=batch_size)

  model.save('AlexNet.model')

#VGG16 Model
def VGG16(train_data, validation_data, epochs, validation_freq, batch_size):

  #Image Processing
  def process_images(image, label):
    #Resize
    image = tf.image.per_image_standardization(image)
    #Resize
    image = tf.image.resize(image, (227,227))
    return image, label

  train_data_size = tf.data.experimental.cardinality(train_data).numpy()
  validation_data_size = tf.data.experimental.cardinality(validation_data).numpy()

  train_data = (train_data.map(process_images).shuffle(buffer_size=train_data_size).batch(batch_size=32, drop_remainder=True))
  validation_data = (validation_data.map(process_images).shuffle(buffer_size=train_data_size).batch(batch_size=32, drop_remainder=True))

  #End of Image Processing

  #Model-Build
  model = Sequential()
  model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Flatten())
  model.add(Dense(units=4096))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(units=4096))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(units=2, activation='softmax'))
  model.add(BatchNormalization())
  model.add(Activation('softmax'))

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  model.fit(train_data, epochs=epochs, validation_data=validation_data, validation_freq=validation_freq, batch_size=batch_size)

  model.save('VGG16.model')