#Libraries used
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy
from keras.utils import np_utils
from library import AlexNet, VGG19, VGG16, ResNet_1, ResNet_2, SqueezeNet, GoogleNet, ZFNet, NFNet_F2, ColorNet, WideResNet
from library import LSTM_Net_text, LSTM_big_Net_text, LSTM_Net_time_series
import cv2



#DataSet for CNNs
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train = np.array([cv2.resize(img, (244,244)) for img in x_train[:50000,:,:,:]])

#model_name(x_train, y_train, input_shape, classes, batch_size, epochs, depth/optional)

#AlexNet(x_train, y_train)

#VGG16(x_train, y_train)

#VGG19(x_train, y_train)

#ResNet_1(x_train, y_train)

#ResNet_2(x_train, y_train)

#SqueezeNet(x_train, y_train)

#GoogleNet(x_train, y_train)

#ZFNet(x_train, y_train)

#NFNet_F2(x_train, y_train)

#ColorNet(x_train, y_train)

#WideResNet(x_train, y_train)

#Text Data for LSTM
"""
filename = "datasets/wonderland.txt"
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
"""

#LSTM_Net_text(x_train, y_train)

#LSTM_big_Net_text(x_train, y_train)

#LSTM_Net_time_series("datasets/airline.csv", 1)