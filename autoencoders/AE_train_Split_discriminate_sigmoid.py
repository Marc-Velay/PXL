from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from sklearn.metrics import accuracy_score

from AE_layers import autoencoder, encoder2, decoder2
import keras_callbacks
from keras.callbacks import ModelCheckpoint

from collections import Counter

from autoencoder_utilities import compare_images, divide_by_class

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

#Shape images so they fit the architecture of a CNN
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = len(X_train_divided)#y_test.shape[1]
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
print(encoders.summary())
print(decoders.summary())

histories = keras_callbacks.Histories()
checkpoint = ModelCheckpoint(weights_base_name+".hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [histories, checkpoint]

epochs=25

#Fit model
train_history = autoencoders.fit(X_train, X_train, epochs=epochs, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.2)

autoencoders.load_weights(weights_base_name+".hdf5")

'''
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test_divided[4][i, ..., 0], cmap='gray')
#plt.show()
for classNum in range(0,10):
    pred = autoencoders[classNum].predict(X_test_divided[4])
    plt.figure(figsize=(20, 4))
    print("Reconstruction of Test Images")
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(pred[i, ..., 0], cmap='gray')
plt.show()
'''
'''
preds = []
g_truth = []
for index, classSet in enumerate(X_test):
    for img_test in classSet:
        pred = [compare_images(img_test, autoencoder.predict(np.reshape(np.array(img_test),(1,28,28,1)))) for autoencoder in autoencoders]
        preds.append(np.argmin(pred))
        g_truth.append(index)

acc = accuracy_score(preds, g_truth)
print(acc)'''
