from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from autoencoder_utilities import compare_images, divide_by_class, shuffle_lists
from AE_layers import autoencoder, encoder2, decoder2
import keras_callbacks

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def calc_accuracy(encoders, bayes_model, test_set):
    preds = []
    for encoder in encoders:
        encoded_test = []
        y_test = []
        for classNum in range(0,10):
            encoded_test.extend(encoder.predict(X_test_divided[classNum]))
            y_test.extend([classNum]*len(X_test_divided[classNum]))
        preds.append(bayes_model.predict(encoded_test))
    print(np.array(preds).shape)
    preds = np.array(preds).T
    preds = [Counter(pred).most_common()[0][0] for pred in preds]
    print(np.array(preds).shape)
    input()

    print("Model accuracy:", accuracy_score(y_test, preds))
    #print(metrics.classification_report(y_test, preds))
    print(metrics.confusion_matrix(y_test, preds))


#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)
weights_base_name = "weights/sigm_weights-ae"

X_train_divided, num_per_class = divide_by_class(X_train, y_train)
X_test_divided, num_per_class_test = divide_by_class(X_test, y_test)
X_train, X_test, y_train, y_test = None, None, None, None

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
for classNum in range(0,10):
    X_train_divided[classNum] = X_train_divided[classNum] / 255
    X_test_divided[classNum] = X_test_divided[classNum] / 255

input_img = Input((28, 28, 1))
#latent_vec = Input((294, 1))
encoders, decoders = [], []
autoencoders = []
for classNum in range(0,10):
    encoders.append(encoder2())
    decoders.append(decoder2())
    encoded_repr = encoders[classNum](input_img)
    reconstructed_img = decoders[classNum](encoded_repr)
    autoencoders.append(Model(input_img, reconstructed_img))
    autoencoders[classNum].compile(loss='mean_squared_error', optimizer='RMSprop')
    #print(encoders[classNum].summary())
    #print(decoders[classNum].summary())

for classNum in range(0,10):
    autoencoders[classNum].load_weights(weights_base_name+str(classNum)+".hdf5")

X_train_divided_base = []
X_train_divided_left = []
y_train_left = []
for classNum in range(0,10):
    X_train, X_val, dump_, dump_ = train_test_split(X_train_divided[classNum], X_train_divided[classNum], test_size=0.2)
    base_indices = random.sample(range(len(X_val)), 800)
    left_indices = [index for index in range(0, len(X_val)) if index not in base_indices]
    X_train_divided_base.append(X_val[base_indices])
    X_train_divided_left.extend(X_val[left_indices])
    y_train_left.extend([classNum]*len(left_indices))
X_train_divided = None
X_train_divided_left, y_train_left = shuffle_lists(X_train_divided_left, y_train_left)

#Shape images so they fit the architecture of a CNN
for classNum in range(0,10):
    X_train_divided_base[classNum] = X_train_divided_base[classNum].reshape(X_train_divided_base[classNum].shape[0], 28, 28, 1).astype('float32')
    X_test_divided[classNum] = X_test_divided[classNum].reshape(X_test_divided[classNum].shape[0], 28, 28, 1).astype('float32')
X_train_divided_left = np.array(X_train_divided_left).reshape(np.array(X_train_divided_left).shape[0], 28, 28, 1).astype('float32')

print("Creating training base for naive bayes model.")
encoded_train = []
y_train = []
for classNum in range(0,10):
    encoded_train.extend(encoders[classNum].predict(X_train_divided_base[classNum]))
    y_train.extend([classNum]*len(X_train_divided_base[classNum]))

print("y_train shape:", np.array(y_train).shape)

bayes_model = GaussianNB()
bayes_model.partial_fit(encoded_train, y_train, classes=np.arange(0,10))

print("Testing model prediction score.")
calc_accuracy(encoders, bayes_model, X_test_divided)


for (new_sample, classNum) in zip(X_train_divided_left, y_train_left):
    new_sample = new_sample.reshape(1, 28, 28, 1).astype('float32')
    new_encoded = encoders[classNum].predict(new_sample)
    bayes_model.partial_fit(new_encoded, [classNum])

    calc_accuracy(encoders, bayes_model, X_test_divided)
