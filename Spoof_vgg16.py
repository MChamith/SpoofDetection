from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Input, Concatenate, Flatten
import numpy as np
import keras.backend as K
from keras_applications.vgg19 import VGG19


def vgg16_feature_fusion():
    input_img1 = Input(shape=(224, 224, 3))  # channel first
    input_img2 = Input(shape=(224, 224, 3))
    input_img3 = Input(shape=(224, 224, 3))

    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    X1 = model.predict(input_img1)
    X2 = model.predict(input_img2)
    X3 = model.predict(input_img3)
    concat = Concatenate()([X1, X2, X3])
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(concat)
    X = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Convolution2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(X)
    # X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    # X = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    X = Dense(1, activation='sigmoid')(X)
    final_model = Model(inputs=[input_img1, input_img2, input_img3], outputs=X)
    return final_model


finalModel = vgg16_feature_fusion()
print(finalModel.summary())
