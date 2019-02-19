from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from sklearn.metrics import accuracy_score

from AE_layers import autoencoder, encoder0, decoder0, encoder1, decoder1
import keras_callbacks
from keras.callbacks import ModelCheckpoint

from collections import Counter

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



# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = len(X_train_divided)#y_test.shape[1]
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
    print(encoders[classNum].summary())
    print(decoders[classNum].summary())

    histories = keras_callbacks.Histories()
    checkpoint = ModelCheckpoint("weights/weights-ae"+str(classNum)+".hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [histories, checkpoint]

    epochs=25

    #Fit model
    train_history = autoencoders[classNum].fit(X_train_divided[classNum], X_train_divided[classNum], epochs=epochs, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.2)

for classNum in range(0,10):
    autoencoders[classNum].load_weights("weights/weights-ae"+str(classNum)+".hdf5")

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
preds = []
g_truth = []
for index, classSet in enumerate(X_test_divided):
    for img_test in classSet:
        pred = [compare_images(img_test, autoencoder.predict(np.reshape(np.array(img_test),(1,28,28,1)))) for autoencoder in autoencoders]
        preds.append(np.argmin(pred))
        g_truth.append(index)

acc = accuracy_score(preds, g_truth)
print(acc)
