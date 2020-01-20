from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Input, Concatenate, Flatten
import numpy as np
import keras.backend as K

def vgg16_feature_fusion():
    base_model = VGG19(weights='imagenet')
    print(base_model.summary())
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    #
    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    #
    # block4_pool_features = model.predict(x

vgg16_feature_fusion()