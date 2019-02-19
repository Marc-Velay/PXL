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


def divide_by_class(X, y):
    inds = y.argsort()
    y = y[inds]
    X = X[inds]
    num_per_class = Counter(y)
    counter=0
    X_divided = []
    for classNum in range(0,10):
        X_divided.append(X[counter:counter+num_per_class[classNum]])
        counter+=num_per_class[classNum]
    return X_divided, num_per_class

def calc_inter_intra_variance(X_divided, num_per_class):
    per_class_average = []
    for classNum in range(0,10):
        per_class_average.append(np.average(np.array(X_divided[classNum]).T, axis=1))
    global_average = np.average(per_class_average, axis=0)

    inter_group = []
    for classNum in range(0,10):
        inter_group.append(np.sum(np.square(np.subtract(per_class_average[classNum],global_average))*num_per_class[classNum]))

    intra_group = []
    for classNum in range(0,10):
        intra_group.append(np.sum(np.sum(np.square(np.subtract(X_divided[classNum], per_class_average[classNum])), axis=0)))

    return np.sum(inter_group), np.sum(intra_group)

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)
np.seterr(divide='ignore', invalid='ignore')

X_train_divided, num_per_class = divide_by_class(X_train, y_train)
X_test_divided, num_per_class_test = divide_by_class(X_test, y_test)

X_train, X_test = None, None

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
for classNum in range(0,10):
    X_train_divided[classNum] = X_train_divided[classNum] / 255
    X_train_divided[classNum] = np.reshape(X_train_divided[classNum], (len(X_train_divided[classNum]), 784))
    X_test_divided[classNum] = X_test_divided[classNum] / 255
    X_test_divided[classNum] = np.reshape(X_test_divided[classNum], (len(X_test_divided[classNum]), 784))

#Using 100% of the data
inter_variance, intra_variance = calc_inter_intra_variance(X_train_divided, num_per_class)
print('variance inter-class with 100 % data:',inter_variance/np.sum([num_per_class[numInClass] for numInClass in num_per_class]))
print('variance intra-class with 100 % data:',intra_variance/np.sum([num_per_class[numInClass] for numInClass in num_per_class]))


#Using 10% of the data
used_indices = []
X_train_divided_reduced = []
for classNum in range(0,10):
    indices = random.sample(range(len(X_train_divided[classNum])), int(len(X_train_divided[classNum])/10))
    used_indices.append(indices)
    X_train_divided_reduced.append(X_train_divided[classNum][indices])
inter_variance_10, intra_variance_10 = calc_inter_intra_variance(X_train_divided_reduced, [len(indi) for indi in used_indices])
print('variance inter-class with 10 % data:',inter_variance_10/np.sum([numInClass for numInClass in [len(indi) for indi in used_indices]]))
print('variance intra-class with 10 % data:',intra_variance_10/np.sum([numInClass for numInClass in [len(indi) for indi in used_indices]]))
