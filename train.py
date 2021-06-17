#-------------------------------------Datasets-and-Models------------------------------------------------
#
#
#
#-------------------------------------Datasets-----------------------------------------------------------
#
#-------------------------------------DataSet-for-CNNs---------------------------------------------------
###CIFAR-10 dataset with resizing option
'''
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
validation_images, validation_labels = x_train[:5000], y_train[:5000]
train_images, train_labels = x_train[5000:], y_train[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    # Resize to wanted size
    image = tf.image.resize(image, (32,32))
    return image, label

train_ds = (train_ds
                  .map(process_images)
                  .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds
                  .map(process_images)
                  .batch(batch_size=32, drop_remainder=True))
validation_ds = (validation_ds
                  .map(process_images)
                  .batch(batch_size=32, drop_remainder=True))
'''
#---------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------Dataset-for-Speech-to-Text-recognition-----------------------------
###Big dataset to download (around 5 GB)
'''
from SpeechToText import SpeechGenerator, SpeechDownloader
gscInfo, classes = SpeechDownloader.PrepareGoogleSpeechCmd(version=2, task='35word')
x_train = SpeechGenerator.SpeechGen(gscInfo['train']['files'], gscInfo['train']['labels'], shuffle=True)
y_train = SpeechGenerator.SpeechGen(gscInfo['val']['files'], gscInfo['val']['labels'], shuffle=True)
'''
#---------------------------------------------------------------------------------------------------------
#
#
#
#-------------------------------------Models-Structures---------------------------------------------------
#
#-------------------------------------General-Structure---------------------------------------------------
### model_name(train_ds, classes=10, batch_size=32, epochs=3)

from library import AlexNet, VGG16, VGG19, SqueezeNet, LeNet, ZFNet, NFNet_F2, ColorNet
#AlexNet(train_ds, epochs=10)
#VGG16(train_ds, epochs=10)
#VGG19(train_ds, epochs=10, batch_size=64)
#SqueezeNet(train_ds, epochs=60)
#LeNet(train_ds, epochs=10)
#ZFNet(train_ds, epochs=10)
#NFNet_F2(train_ds, epochs=10)
#ColorNet(train_ds, epochs=10)
#---------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------ResNet-Structures---------------------------------------------------
### model_name(train_ds, classes=10, batch_size=32, epochs=3, depth=20)

from library import ResNet_1, ResNet_2, WideResNet
#WideResNet(train_ds, epochs=10)
#ResNet_1(train_ds, epochs=10, depth=152)
#ResNet_2(train_ds, epochs=10)
#---------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------LSTM-text-prediction-Structure--------------------------------------
### model_name(filepath, batch_size=128, epochs=3)

from library import LSTM_text
LSTM_text("datasets/wonderland.txt", epochs=30)
#---------------------------------------------------------------------------------------------------------
#
#
#--------------------------------------LSTM-time-series-Structure-----------------------------------------
### model_name(filepath, time_steps=1, batch_size=1, epochs=3)

from library import LSTM_time_series
#LSTM_time_series("datasets/airline.csv", epochs=30)
#---------------------------------------------------------------------------------------------------------
#
#
#--------------------------------------Stock-Prediction-Structure-----------------------------------------
### model_name(filepath, batch_size=16, epochs=3)

from library import LSTM_Stock
#LSTM_Stock('datasets/apple_share_price.csv', epochs=30)
#---------------------------------------------------------------------------------------------------------
#
#
#--------------------------------------Speech-To-Text-Structure-------------------------------------------
### model_name(x_train, y_train, classes, samplingrate=16000, inputLength=16000, epochs=3)

from library import RNN_Speech, Att_RNN_Speech
#RNN_Speech(x_train, y_train, classes, epochs=10)
#Att_RNN_Speech(x_train, y_train, classes, epochs=10)
#---------------------------------------------------------------------------------------------------------