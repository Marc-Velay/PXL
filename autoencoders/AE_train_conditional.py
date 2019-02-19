from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Input

from AE_layers import autoencoder
import keras_callbacks
from keras.callbacks import ModelCheckpoint

from collections import Counter

from autoencoder_utilities import divide_by_class


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

'''
plt.subplot(221)
plt.imshow(X_train_divided[0][0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train_divided[1][0], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train_divided[2][0], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train_divided[3][0], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
'''

#Shape images so they fit the architecture of a CNN
for classNum in range(0,10):
    X_train_divided[classNum] = X_train_divided[classNum].reshape(X_train_divided[classNum].shape[0], 28, 28, 1).astype('float32')
    X_test_divided[classNum] = X_test_divided[classNum].reshape(X_test_divided[classNum].shape[0], 28, 28, 1).astype('float32')



input_img = Input((28, 28, 1))
autoencoders = []
for classNum in range(0,10):
    autoencoders.append(Model(input_img, autoencoder(input_img)))
    autoencoders[classNum].compile(loss='mean_squared_error', optimizer='RMSprop')
    print(autoencoders[classNum].summary())

    histories = keras_callbacks.Histories()
    checkpoint = ModelCheckpoint("weights/weights-ae"+str(classNum)+".hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [histories, checkpoint]

    epochs=25

    #Fit model
    train_history = autoencoders[classNum].fit(X_train_divided[classNum], X_train_divided[classNum], epochs=epochs, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.2)

    '''
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    '''
for classNum in range(0,10):
    autoencoders[classNum].load_weights("weights/weights-ae"+str(classNum)+".hdf5")


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
