#Libraries used
from tensorflow.keras.datasets import cifar10
from keras.utils import to_categorical
from library import AlexNet, VGG19, VGG16, ResNet_1, ResNet_2, SqueezeNet, GoogleNet, ZFNet, NFNet_F2, ColorNet, WideResNet
from library import LSTM_Net_text, LSTM_big_Net_text, LSTM_Net_time_series, Stock_Net, RNN_Speech, Att_RNN_Speech
#from SpeechToText import SpeechGenerator, SpeechDownloader


#DataSet for CNNs
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train = np.array([cv2.resize(img, (128,128)) for img in x_train[:50000,:,:,:]])

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

#LSTM_Net_text("datasets/wonderland.txt")

#LSTM_big_Net_text("datasets/wonderland.txt")

#LSTM_Net_time_series("datasets/airline.csv", 2)

#Stock_Net('datasets/apple_share_price.csv')

"""
gscInfo, classes = SpeechDownloader.PrepareGoogleSpeechCmd(version=2, task='35word')
x_train = SpeechGenerator.SpeechGen(gscInfo['train']['files'], gscInfo['train']['labels'], shuffle=True)
# handle the fact that number of samples in validation may not be multiple of batch_size with shuffle=True
y_train = SpeechGenerator.SpeechGen(gscInfo['val']['files'], gscInfo['val']['labels'], shuffle=True)
"""

#RNN_Speech(x_train, y_train, classes)

#Att_RNN_Speech(x_train, y_train, classes)