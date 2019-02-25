import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, Flatten, Reshape, Dense
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop

def autoencoder(input_img):
    #input_img = Input(shape = (28, 28, 1))

    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    batch4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2,2))(batch4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    batch5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2,2))(batch5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

def encoder0():
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    img = Input((28, 28, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img) #28 x 28 x 32
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    #batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #7 x 7 x 64
    flat = Flatten()(pool3)
    # 258 latent dim size
    latent = Dense(294)(flat)

    return Model(img, latent)

def decoder0():
    #decoder
    #input = Reshape((7,7,6))(latentVect)
    latent = Input((294, 1))
    reshaped = Reshape((7,7,6))(latent)
    batch4 = BatchNormalization()(reshaped)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch4) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    batch5 = BatchNormalization()(up1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch5) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return Model(latent, decoded)


def encoder1():
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    img = Input((28, 28, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    flat = Flatten()(pool4)
    # 294 latent dim size
    latent = Dense(294)(flat)

    return Model(img, latent)

def decoder1():
    #decoder
    #input = Reshape((7,7,6))(latentVect)
    latent = Input((294, 1))
    reshaped = Reshape((7,7,6))(latent)
    batch4 = BatchNormalization()(reshaped)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch4) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv5) # 14 x 14 x 128
    batch5 = BatchNormalization()(up1)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch5) #7 x 7 x 128
    up2 = UpSampling2D((2,2))(conv6) # 14 x 14 x 128
    batch6 = BatchNormalization()(up2)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch6) #7 x 7 x 128
    #up3 = UpSampling2D((2,2))(conv7) # 14 x 14 x 128
    batch7 = BatchNormalization()(conv7)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(batch7) # 14 x 14 x 64
    #up4 = UpSampling2D((2,2))(conv8) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv8) # 28 x 28 x 1
    return Model(latent, decoded)


def encoder2():
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    img = Input((28, 28, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    flat = Flatten()(pool4)
    # 294 latent dim size
    latent = Dense(294, activation='sigmoid')(flat)

    return Model(img, latent)

def decoder2():
    #decoder
    #input = Reshape((7,7,6))(latentVect)
    latent = Input((294, 1))
    reshaped = Reshape((7,7,6))(latent)
    batch4 = BatchNormalization()(reshaped)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch4) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv5) # 14 x 14 x 128
    batch5 = BatchNormalization()(up1)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch5) #7 x 7 x 128
    up2 = UpSampling2D((2,2))(conv6) # 14 x 14 x 128
    batch6 = BatchNormalization()(up2)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch6) #7 x 7 x 128
    #up3 = UpSampling2D((2,2))(conv7) # 14 x 14 x 128
    batch7 = BatchNormalization()(conv7)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(batch7) # 14 x 14 x 64
    #up4 = UpSampling2D((2,2))(conv8) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv8) # 28 x 28 x 1
    return Model(latent, decoded)
