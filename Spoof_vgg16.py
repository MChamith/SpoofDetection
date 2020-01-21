from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Input, Concatenate, Flatten, concatenate
import numpy as np
import keras.backend as K
from keras.applications.vgg19 import VGG19


def vgg16_feature_fusion():
    base_model1 = VGG19(weights='imagenet', include_top=False)
    base_model2 = VGG19(weights='imagenet', include_top=False)
    base_model3 = VGG19(weights='imagenet', include_top=False)
    for i, layer in enumerate(base_model2.layers):
        layer.name = str(layer.name) + '_2'
    for i, layer in enumerate(base_model3.layers):
        layer.name = str(layer.name) + '_3'
    concatenated = concatenate(
        [base_model1.get_layer('block4_pool').output, base_model2.get_layer('block4_pool_2').output,
         base_model3.get_layer('block4_pool_3').output])

    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(concatenated)
    X = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Convolution2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(X)
    # X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    # X = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X)
    # X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    # X = Dense(128, activation='relu')(X)
    # X = Dense(64, activation='relu')(X)
    # print('before ' +str(X.shape))
    X = Dense(1, activation='sigmoid')(X)
    # print('after ' + str(X.shape))
    model = Model(inputs=[base_model1.input, base_model2.input, base_model3.input], outputs=X)
    return model


finalModel = vgg16_feature_fusion()
print(finalModel.summary())
