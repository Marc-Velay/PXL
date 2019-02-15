from keras.datasets import mnist
import matplotlib.pyplot as plt
import math
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
'''
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
'''
'''
summaries = []
for index, classSet in enumerate(X_train_divided):
    start = time()
    summaries.append(class_summary(classSet))
    print('time per class:', time()-start)

preds = []
grnd_truth = []
for classIndex, classSet in enumerate(X_test_divided):
    for test_img in classSet:
        preds.append(predict(summaries, test_img))
        grnd_truth.append(classIndex)
        #print(preds)
        #print(classIndex)
        #input()
acc = accuracy_score(preds, grnd_truth)
print('accuracy:', acc)
'''

X_train = X_train / 255
X_train = np.reshape(X_train, (len(X_train), 784))

X_test = X_test / 255
X_test = np.reshape(X_test, (len(X_test), 784))

model = GaussianNB()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds))
print(metrics.confusion_matrix(y_test, preds))

preds = model.predict(X_train)
print(accuracy_score(y_train, preds))
print(metrics.classification_report(y_train, preds))
print(metrics.confusion_matrix(y_train, preds))
