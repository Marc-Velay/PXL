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
import matplotlib.pyplot as plt

from autoencoder_utilities import compare_images, divide_by_class, shuffle_lists
from AE_layers import autoencoder, encoder2, decoder2
import keras_callbacks

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

accuracy = []
plt.ion()
fig = plt.figure()
plt.plot(accuracy)
plt.show()

def calc_accuracy(encoders, bayes_model, X_test, y_test):
    encoded_test = []
    encoded_test.extend(encoders.predict(X_test))
    preds = bayes_model.predict(encoded_test)
    acc = accuracy_score(y_test, preds)
    accuracy.append(acc)
    print("Model accuracy:", acc)
    print(metrics.confusion_matrix(y_test, preds))

    plt.plot(accuracy, '-b')
    plt.draw()
    plt.pause(0.001)


#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)
weights_base_name = "weights/sigm_weights-ae"
'''
X_train_divided, num_per_class = divide_by_class(X_train, y_train)
X_test_divided, num_per_class_test = divide_by_class(X_test, y_test)
X_train, X_test, y_train, y_test = None, None, None, None
'''
# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
X_train = X_train / 255
X_test = X_test / 255

print(np.array(X_train).shape)

input_img = Input((28, 28, 1))
#latent_vec = Input((294, 1))
encoders, decoders = [], []
autoencoders = []
encoders = encoder2()
decoders = decoder2()
encoded_repr = encoders(input_img)
reconstructed_img = decoders(encoded_repr)
autoencoders = Model(input_img, reconstructed_img)
autoencoders.compile(loss='mean_squared_error', optimizer='RMSprop')
#print(encoders.summary())
#print(decoders.summary())

autoencoders.load_weights(weights_base_name+".hdf5")

X_train_base = []
y_train_base = []
X_train_left = []
y_train_left = []

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
base_indices = random.sample(range(len(X_val)), 25)
left_indices = [index for index in range(0, len(X_val)) if index not in base_indices]
X_train_base = X_val[base_indices]
y_train_base = y_val[base_indices]
X_train_left = X_val[left_indices]
y_train_left = y_val[left_indices]

print(np.array(X_train_base).shape)
print(np.array(y_train_base).shape)

X_train_base, y_train_base = shuffle_lists(X_train_base, y_train_base)
X_train_left, y_train_left = shuffle_lists(X_train_left, y_train_left)

print(np.array(X_train_base).shape)

#Shape images so they fit the architecture of a CNN
X_train_base = np.array(X_train_base).reshape(np.array(X_train_base).shape[0], 28, 28, 1).astype('float32')
X_train_left = np.array(X_train_left).reshape(np.array(X_train_left).shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

print("Creating training base for naive bayes model.")
encoded_train = []
encoded_train.extend(encoders.predict(X_train_base))

print("y_train shape:", np.array(y_train_base).shape)

bayes_model = GaussianNB()
bayes_model.partial_fit(encoded_train, y_train_base, classes=np.arange(0,10))

print("Testing model prediction score.")
calc_accuracy(encoders, bayes_model, X_test, y_test)


for (new_sample, classNum) in zip(X_train_left, y_train_left):
    new_sample = new_sample.reshape(1, 28, 28, 1).astype('float32')
    new_encoded = encoders.predict(new_sample)
    bayes_model.partial_fit(new_encoded, [classNum])

    calc_accuracy(encoders, bayes_model, X_test, y_test)
