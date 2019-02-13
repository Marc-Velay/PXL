from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

from skimage.transform import rotate
from scipy.ndimage import shift
from scipy.ndimage import median_filter
from scipy.signal import convolve2d

def rotate_img(img, angle):
    return rotate(img, angle)

def shift_img(img, height, width):
    return shift(img, [height, width])

def median_filter_img(img, window_size):
    return median_filter(img, size=(window_size,window_size))

def convolve_contours_img(img, window):
    return convolve2d(img, window, mode="same", boundary="symm")

#Load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#For reproductability
np.random.seed(7)

# normalize inputs from 0-255 to 0-1
#Here it's a simple element-wise operation due to images in grayscale, not RGB
X_train = X_train / 255
X_test = X_test / 255


plt.subplot(331)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))

rot_img = rotate_img(X_train[0], 30)
plt.subplot(332)
plt.imshow(rot_img, cmap=plt.get_cmap('gray'))

#shift by (height, width, dimension): here [28,28,1] image
shift_img1 = shift_img(X_train[0], 0, 10)
plt.subplot(333)
plt.imshow(shift_img1, cmap=plt.get_cmap('gray'))

shift_img2 = shift_img(X_train[0], 10, 0)
plt.subplot(334)
plt.imshow(shift_img2, cmap=plt.get_cmap('gray'))

filtered_img = median_filter_img(X_train[0], 3)
plt.subplot(335)
plt.imshow(filtered_img, cmap=plt.get_cmap('gray'))

filtered_rot_img = median_filter_img(rot_img, 3)
plt.subplot(336)
plt.imshow(filtered_rot_img, cmap=plt.get_cmap('gray'))

sobel_x = np.c_[
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

sobel_y = np.c_[
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
]

nop = np.c_[
    [1,2,1],
    [2,18,2],
    [1,2,1]
]

filtered_sobel_x = convolve_contours_img(X_train[0], sobel_x)
plt.subplot(337)
plt.imshow(filtered_sobel_x, cmap=plt.get_cmap('gray'))

filtered_sobel_y = convolve_contours_img(X_train[0], sobel_y)
print(X_train[0].shape)
plt.subplot(338)
plt.imshow(filtered_sobel_y, cmap=plt.get_cmap('gray'))

plt.subplot(339)
plt.imshow(convolve_contours_img(X_train[0], nop), cmap=plt.get_cmap('gray'))

plt.show()
