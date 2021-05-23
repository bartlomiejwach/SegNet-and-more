#Libraries used
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from library import AlexNet, VGG19, VGG16, ResNet_1

#DataSet
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x = x_train.astype("uint8")

#model_name(x_train, y_train, input_shape, classes, batch_size, epochs, depth/optional)

#AlexNet(x_train, y_train)

#VGG16(x_train, y_train)

#VGG19(x_train, y_train)

#ResNet_1(x_train, y_train)