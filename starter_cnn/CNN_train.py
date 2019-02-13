from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras import backend as K

from CNN_layers import baseline_model
import keras_callbacks
from keras.callbacks import ModelCheckpoint

weightsFilepath="weights/weights-cnn.hdf5"

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
X_train = X_train / 255
X_test = X_test / 255

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


#Shape images so they fit the architecture of a CNN
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Define model using keras code in CNN_layers
model = baseline_model(num_classes)

histories = keras_callbacks.Histories()
checkpoint = ModelCheckpoint(weightsFilepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [histories]

#Fit model
model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=2, callbacks=callbacks_list, validation_split=0.2)
# Basic accuracy score
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN accuracy on test set: %.2f%%" % (scores[1]*100))
