from keras.datasets import mnist
import matplotlib.pyplot as plt
import math
import numpy as np
from keras import backend as K

from collections import Counter
from time import time

def class_summary(vect_arr):
    res = []
    for attribute in np.array(vect_arr).T:
        start = time()
        res.append((np.mean(attribute), np.std(attribute)))
    return res

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)


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

X_train, X_test, y_train, y_test = None, None, None, None

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
for classNum in range(0,10):
    X_train_divided[classNum] = X_train_divided[classNum] / 255
    X_train_divided[classNum] = np.reshape(X_train_divided[classNum], (len(X_train_divided[classNum]), 784))
    X_test_divided[classNum] = X_test_divided[classNum] / 255
    X_test_divided[classNum] = np.reshape(X_test_divided[classNum], (len(X_test_divided[classNum]), 784))

summaries = []
for index, classSet in enumerate(X_train_divided):
    start = time()
    summaries.append(class_summary(classSet))
    print('time per class:', time()-start)
