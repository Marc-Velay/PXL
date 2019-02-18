from keras.datasets import mnist
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

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
    X_train_divided[classNum] = np.reshape(X_train_divided[classNum], (len(X_train_divided[classNum]), 784))
    X_test_divided[classNum] = X_test_divided[classNum] / 255
    X_test_divided[classNum] = np.reshape(X_test_divided[classNum], (len(X_test_divided[classNum]), 784))

#Using 100% of the data
per_class_average = []
for classNum in range(0,10):
    per_class_average.append(np.average(np.array(X_train_divided[classNum]).T, axis=1))
global_average = np.average(per_class_average, axis=0)

inter_group = []
for classNum in range(0,10):
    inter_group.append(np.sum(np.square(np.subtract(per_class_average[classNum],global_average))*num_per_class[classNum]))
print('variance inter-class with 100 % data:',np.sum(inter_group))

intra_group = []
for classNum in range(0,10):
    intra_group.append(np.sum(np.sum(np.square(np.subtract(X_train_divided[classNum], per_class_average[classNum])), axis=0)))
print('variance intra-class with 100 % data:',np.sum(intra_group))



#Using 10% of the data
per_class_average = []
used_indices = []
for classNum in range(0,10):
    indices = random.sample(range(len(X_train_divided[classNum])), int(len(X_train_divided[classNum])/10))
    used_indices.append(indices)
    per_class_average.append(np.average(np.array(X_train_divided[classNum][indices]).T, axis=1))
global_average = np.average(per_class_average, axis=0)

inter_group = []
for classNum in range(0,10):
    inter_group.append(np.sum(np.square(np.subtract(per_class_average[classNum],global_average))*[len(indi) for indi in used_indices][classNum]))
print('variance inter-class with 10 % data:',np.sum(inter_group))

intra_group = []
for classNum in range(0,10):
    intra_group.append(np.sum(np.sum(np.square(np.subtract(X_train_divided[classNum][used_indices[classNum]], per_class_average[classNum])), axis=0)))
print('variance intra-class with 10 % data:',np.sum(intra_group))
