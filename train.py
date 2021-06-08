#-------------------------------------Datasets-and-Models------------------------------------------------
#
#
#
#-------------------------------------Datasets-----------------------------------------------------------
#
#-------------------------------------DataSet-for-CNNs---------------------------------------------------
#from tensorflow.keras.datasets import cifar10
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

###Optional resizing - not very convinient to use whith cifar but it is something
#x_train = np.array([cv2.resize(img, (128,128)) for img in x_train[:50000,:,:,:]])
#---------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------Dataset-for-Speech-to-Text-recognition-----------------------------
###Big dataset to download (around 5 GB)
#from SpeechToText import SpeechGenerator, SpeechDownloader
#gscInfo, classes = SpeechDownloader.PrepareGoogleSpeechCmd(version=2, task='35word')
#x_train = SpeechGenerator.SpeechGen(gscInfo['train']['files'], gscInfo['train']['labels'], shuffle=True)
#y_train = SpeechGenerator.SpeechGen(gscInfo['val']['files'], gscInfo['val']['labels'], shuffle=True)
#---------------------------------------------------------------------------------------------------------
#
#
#
#-------------------------------------Models-Structures---------------------------------------------------
#
#-------------------------------------General-Structure---------------------------------------------------
### model_name(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3)

#from library import AlexNet, VGG16, VGG19, SqueezeNet, GoogleNet, ZFNet, NFNet_F2, ColorNet
#AlexNet(x_train, y_train)
#VGG16(x_train, y_train)
#VGG19(x_train, y_train)
#SqueezeNet(x_train, y_train)
#GoogleNet(x_train, y_train)
#ZFNet(x_train, y_train)
#NFNet_F2(x_train, y_train)
#ColorNet(x_train, y_train)
#---------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------ResNet-Structures---------------------------------------------------
### model_name(x_train, y_train, input_shape=[32,32,3], classes=10, batch_size=32, epochs=3, depth=20)

#from library import ResNet_1, ResNet_2, WideResNet
#WideResNet(x_train, y_train)
#ResNet_1(x_train, y_train)
#ResNet_2(x_train, y_train)
#---------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------LSTM-text-prediction-Structure--------------------------------------
### model_name(filepath, batch_size=128, epochs=3)

#from library import LSTM_text
#LSTM_text("datasets/wonderland.txt")
#---------------------------------------------------------------------------------------------------------
#
#
#--------------------------------------LSTM-time-series-Structure-----------------------------------------
### model_name(filepath, time_steps=1, batch_size=1, epochs=3)

#from library import LSTM_time_series
#LSTM_time_series("datasets/airline.csv")
#---------------------------------------------------------------------------------------------------------
#
#
#--------------------------------------Stock-Prediction-Structure-----------------------------------------
### model_name(filepath, batch_size=16, epochs=3)

#from library import LSTM_Stock
#LSTM_Stock('datasets/apple_share_price.csv')
#---------------------------------------------------------------------------------------------------------
#
#
#--------------------------------------Speech-To-Text-Structure-------------------------------------------
### model_name(x_train, y_train, classes, samplingrate=16000, inputLength=16000, epochs=3)

#from library import RNN_Speech, Att_RNN_Speech
#RNN_Speech(x_train, y_train, classes)
#Att_RNN_Speech(x_train, y_train, classes)
#---------------------------------------------------------------------------------------------------------