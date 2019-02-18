from keras.datasets import mnist
import matplotlib.pyplot as plt
import math
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from sklearn.metrics import accuracy_score
from sklearn import metrics

from collections import Counter
from time import time

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)
np.seterr(divide='ignore', invalid='ignore')

inds = y_train.argsort()
y_train = y_train[inds]
X_train = X_train[inds]
num_per_class = Counter(y_train)
counter=0
X_train_divided = []
for classNum in range(0,10):
    X_train_divided.append(X_train[counter:counter+num_per_class[classNum]])
    counter+=num_per_class[classNum]

inds = y_test.argsort()
y_test = y_test[inds]
X_test = X_test[inds]
num_per_class = Counter(y_test)
counter=0
X_test_divided = []
for classNum in range(0,10):
    X_test_divided.append(X_test[counter:counter+num_per_class[classNum]])
    counter+=num_per_class[classNum]

X_train, X_test = None, None

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
for classNum in range(0,10):
    X_train_divided[classNum] = X_train_divided[classNum] / 255
    X_test_divided[classNum] = X_test_divided[classNum] / 255


for classNum in range(0,10):
    X_train_divided[classNum] = X_train_divided[classNum].reshape(X_train_divided[classNum].shape[0], 28, 28, 1).astype('float32')
    X_test_divided[classNum] = X_test_divided[classNum].reshape(X_test_divided[classNum].shape[0], 28, 28, 1).astype('float32')


input_img = Input((28, 28, 1))
#latent_vec = Input((294, 1))
encoders, decoders = [], []
autoencoders = []
for classNum in range(0,10):
    encoders.append(encoder1())
    decoders.append(decoder1())
    encoded_repr = encoders[classNum](input_img)
    reconstructed_img = decoders[classNum](encoded_repr)
    autoencoders.append(Model(input_img, reconstructed_img))
    autoencoders[classNum].compile(loss='mean_squared_error', optimizer='RMSprop')
    #print(encoders[classNum].summary())
    #print(decoders[classNum].summary())

for classNum in range(0,10):
    autoencoders[classNum].load_weights("weights/weights-ae"+str(classNum)+".hdf5")

per_class_average = []
for classNum in range(0,10):
    per_class_average.append(np.average(np.array(X_train_divided[classNum]).T, axis=1))
global_average = np.average(per_class_average, axis=0)

inter_group = []
for classNum in range(0,10):
    inter_group.append(np.sum(np.square(np.subtract(per_class_average[classNum],global_average))*num_per_class[classNum]))
print('variance inter-class:',np.sum(inter_group))

intra_group = []
for classNum in range(0,10):
    intra_group.append(np.sum(np.sum(np.square(np.subtract(X_train_divided[classNum], per_class_average[classNum])), axis=0)))
print('variance intra-class:',np.sum(intra_group))
