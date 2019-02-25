import numpy as np
from collections import Counter
from sklearn.utils import shuffle

def compare_images(img1, img2):
    # calculate the difference and its norms
    diff = img1 - img2
    err = np.sum((diff) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])
    return err


def divide_by_class(X, y):
    inds = y.argsort()
    y = y[inds]
    X = X[inds]
    num_per_class = Counter(y)
    counter=0
    X_divided = []
    for classNum in range(0,10):
        X_divided.append(X[counter:counter+num_per_class[classNum]])
        counter+=num_per_class[classNum]
    return X_divided, num_per_class

def shuffle_lists(list1, list2):
    return shuffle(list1, list2)
