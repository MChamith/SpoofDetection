from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Input, Concatenate, Flatten
import numpy as np
import keras.backend as K


def cnn_model():
    input_img1 = Input(shape=(128, 128, 1))  # channel first
    input_img2 = Input(shape=(128, 128, 1))
    input_img3 = Input(shape=(128, 128, 1))

    X1 = Convolution2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(input_img1)
    X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    X1 = Convolution2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(X1)
    X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    X1 = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X1)
    X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    # X1 = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X1)
    # X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    # X1 = Convolution2D(filters=512, kernel_size=(7, 7), padding='same', activation='relu')(X1)
    # X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X1)
    # X1 = Convolution2D(filters=1024, kernel_size=(7, 7), padding='same', activation='relu')(X1)
    # X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X1)

    X2 = Convolution2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(input_img2)
    X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    X2 = Convolution2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    X2 = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    # X2 = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    # X2 = Convolution2D(filters=512, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X2)
    # X2 = Convolution2D(filters=1024, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X2)

    X3 = Convolution2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(input_img3)
    X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)
    X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)
    X3 = Convolution2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(X3)
    X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)
    X3 = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X3)
    X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)
    # X3 = Convolution2D(filters=512, kernel_size=(7, 7), padding='same', activation='relu')(X3)
    # X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X3)
    # X3 = Convolution2D(filters=1024, kernel_size=(7, 7), padding='same', activation='relu')(X3)
    # X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X3)
    concat = Concatenate()([X1, X2])
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(concat)
    X = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Convolution2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(X)
    # X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    # X = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    # X = Dense(128, activation='relu')(X)
    # X = Dense(64, activation='relu')(X)
    # print('before ' +str(X.shape))
    X = Dense(1, activation='sigmoid')(X)
    # print('after ' + str(X.shape))
    model = Model(inputs=[input_img1, input_img2, input_img3],  outputs=X)

    return model

# model = cnn_model()
# print(model.summary())
#
# import keras
#
# model = keras.models.load_model('Models/Model-03.h5')
# print(model.summary())
