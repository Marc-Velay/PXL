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

from collections import Counter

weightsFilepath="weights/weights-ae.hdf5"

def compare_images(img1, img2):
    # calculate the difference and its norms
    diff = img1 - img2
    err = np.sum((diff) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])
    return err

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)

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

'''
encoded_imgs = encoders[0].predict(X_test_divided[0][:4])
encoded_imgs = np.reshape(encoded_imgs, (4, 294, 1))
decoded_imgs = decoders[0].predict(encoded_imgs)

plt.figure(figsize=(20, 4))
n=4
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test_divided[0][i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

encoded_train = []
y_train = []
for classNum in range(0,10):
    encoded_train.extend(encoders[classNum].predict(X_train_divided[classNum]))
    y_train.extend([classNum]*len(X_train_divided[classNum]))

bayes_model = GaussianNB()
bayes_model.fit(encoded_train, y_train)

encoded_test = []
y_test = []
for classNum in range(0,10):
    encoded_test.extend(encoders[classNum].predict(X_test_divided[classNum]))
    y_test.extend([classNum]*len(X_test_divided[classNum]))


preds = bayes_model.predict(encoded_test)
print(accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds))
print(metrics.confusion_matrix(y_test, preds))
