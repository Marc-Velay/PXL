from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from AE_layers import autoencoder, encoder0, decoder0, encoder1, decoder1
import keras_callbacks
from keras.callbacks import ModelCheckpoint

from autoencoder_utilities import compare_images, divide_by_class

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)


X_train_divided, num_per_class = divide_by_class(X_train, y_train)
X_test_divided, num_per_class_test = divide_by_class(X_test, y_test)
X_train, X_test, y_train, y_test = None, None, None, None

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
for classNum in range(0,10):
    X_train_divided[classNum] = X_train_divided[classNum] / 255
    X_test_divided[classNum] = X_test_divided[classNum] / 255

#Shape images so they fit the architecture of a CNN
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

print("Creating training base for naive bayes model.")
encoded_train = []
y_train = []
for classNum in range(0,10):
    encoded_train.extend(encoders[classNum].predict(X_train_divided[classNum]))
    y_train.extend([classNum]*len(X_train_divided[classNum]))

bayes_model = GaussianNB()
bayes_model.fit(encoded_train, y_train)

print("Testing model prediction score.")
encoded_test = []
y_test = []
for classNum in range(0,10):
    encoded_test.extend(encoders[classNum].predict(X_test_divided[classNum]))
    y_test.extend([classNum]*len(X_test_divided[classNum]))


preds = bayes_model.predict(encoded_test)
print("Model accuracy:", accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds))
print(metrics.confusion_matrix(y_test, preds))
