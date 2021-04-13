#Libraries used
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from library import AlexNet

#GPU Init
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#test

#DataSet
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
validation_data = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

AlexNet(train_data, validation_data, 1, 1)