from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Input, Concatenate, Flatten
import numpy as np
import keras.backend as K


def cnn_model():
    input_img1 = Input(shape=(256, 256, 1))  # channel first
    X = Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_img1)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Convolution2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    # X = Convolution2D(filters=512, kernel_size=(7, 7), padding='same', activation='relu')(X)
    # X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    # X = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = Flatten()(X)
    # X = Dense(128, activation='relu')(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    # X = Dense(128, activation='relu')(X)
    # X = Dense(64, activation='relu')(X)
    # print('before ' +str(X.shape))
    X = Dense(1, activation='sigmoid')(X)
    # print('after ' + str(X.shape))
    model = Model(inputs=input_img1,  outputs=X)

    return model

# model = cnn_model()
# print(model.summary())
#
# import keras
#
# model = keras.models.load_model('Models/Model-03.h5')
# print(model.summary())
