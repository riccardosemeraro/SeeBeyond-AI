# Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Import Warnings 
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
# Import tensorflow as the backend for Keras
from keras import backend as K
K.image_data_format() # sostituito a --> K.set_image_dim_ordering()
from keras.utils import to_categorical # sostituito a --> from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten #sostituito a --> from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D as Convolution2D, MaxPooling2D #sostituito a --> from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam as adam #sostituto a --> from keras.optimizers import SGD,RMSprop,adam
from keras.callbacks import TensorBoard
# Import required libraries for cnfusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

from keras.models import load_model

import pickle

#-------------------------------------------------------------- 1. Loading the previously variable saved


with open('./model/variabili.pkl', 'rb') as file:
    X_test, y_test = pickle.load(file)


#-------------------------------------------------------------- 4. Evaluating the model & Predicting the output class of a test image
# ora devo caricare il modello che ho salvato in precedenza
# Load the pre-trained models
cnn_model = load_model('./model/SeeBeyond.h5')

score = cnn_model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

#--------------------------------------------------------------


test_image = X_test[0:1]
print (test_image.shape)
print(cnn_model.predict(test_image))
print(cnn_model.predict_step(test_image))
#print(cnn_model.predict_classes(test_image))
print(y_test[0:1])

#--------------------------------------------------------------

test_img = cv2.imread('./dataset/gatto/00000001_000.jpg')

test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_img = cv2.resize(test_img, (128, 128))
test_img = np.array(test_img)
test_img = test_img.astype('float32')
test_img /= 255
print(test_img.shape)


#--------------------------------------------------------------

image = test_img.reshape((128,128))
plt.imshow(image)
plt.show()