#Libraries used
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from library import AlexNet, VGG19, VGG16, ResNet_1, ResNet_2, SqueezeNet, GoogleNet, ZFNet, NFNet_F2, ColorNet, WideResNet
import cv2

#DataSet
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
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