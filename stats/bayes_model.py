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
