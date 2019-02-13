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

weightsFilepath="weights/weights-ae.hdf5"

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
X_train = X_train / 255
X_test = X_test / 255

'''
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
'''

#Shape images so they fit the architecture of a CNN
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = len(X_train_divided)#y_test.shape[1]
input_img = Input((28, 28, 1))
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer='RMSprop')
print(autoencoder.summary())

histories = keras_callbacks.Histories()
checkpoint = ModelCheckpoint("weights/weights-ae.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [histories, checkpoint]

epochs=25

#Fit model
train_history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.2)

# Basic accuracy score
#scores = autoencoder.evaluate(X_test, X_test, verbose=0)
#print("CNN accuracy on test set: %.2f%%" % (scores[1]*100))
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
autoencoder.load_weights("weights/weights-ae.hdf5")


plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[i, ..., 0], cmap='gray')
#plt.show()
pred = autoencoder.predict(X_test)
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')
plt.show()
