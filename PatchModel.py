import cv2
import random
import numpy as np
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Input, Concatenate, Flatten, BatchNormalization
import numpy as np
import keras.backend as K

def cnn_model():
    input_img1 = Input(shape=(96, 96, 16))  # channel first
    input_img2 = Input(shape=(96, 96, 16))
    input_img3 = Input(shape=(96, 96, 16))

    X1 = Convolution2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu')(input_img1)
    X1 = BatchNormalization()(X1)
    X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    X1 = Convolution2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(X1)
    X1 = BatchNormalization()(X1)
    X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    X1 = Convolution2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(X1)
    X1 = BatchNormalization()(X1)
    X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)
    # X1 = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(X1)
    # X1 = BatchNormalization()(X1)
    # X1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X1)

    X2 = Convolution2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu')(input_img2)
    X2 = BatchNormalization()(X2)
    X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    X2 = Convolution2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(X2)
    X2 = BatchNormalization()(X2)
    X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    X2 = Convolution2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(X2)
    X2 = BatchNormalization()(X2)
    X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    # X2 = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(X2)
    # X2 = BatchNormalization()(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    # X2 = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    # X2 = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X2)
    # X2 = Convolution2D(filters=512, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X2)
    # X2 = Convolution2D(filters=1024, kernel_size=(7, 7), padding='same', activation='relu')(X2)
    # X2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X2)

    X3 = Convolution2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu')(input_img3)
    X3 = BatchNormalization()(X3)
    X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)
    X3 = Convolution2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(X3)
    X3 = BatchNormalization()(X3)
    X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)
    X3 = Convolution2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(X3)
    X3 = BatchNormalization()(X3)
    X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)
    # X3 = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(X3)
    # X3 = BatchNormalization()(X3)
    # X3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X3)

    concat = Concatenate()([X1, X2, X3])
    X = MaxPool2D(pool_size=(3, 3), strides=(3, 3))(concat)
    X = Convolution2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    # X = Convolution2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(X)
    # X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    # X = Convolution2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    # X = Dense(128, activation='relu')(X)
    # X = Dense(64, activation='relu')(X)
    # print('before ' +str(X.shape))
    X = Dense(1, activation='sigmoid')(X)
    print('after ' + str(X.shape))
    model = Model(inputs=[input_img1, input_img2, input_img3],  outputs=X)

    return model




model = cnn_model()
print(model.summary())
# X_gray = np.empty((32, 96, 96, 16))
#
# for i in range(32):
#     img = cv2.imread('036_spoof/036-1-2-1-12.jpg')
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print('gray shape ' + str(img_gray.shape))
#     height, width = img.shape[:2]
#     print('height ' + str(height))
#     print('width' + str(width))
#     images = []
#     i = 0
#     while i < 16:
#         rH = random.uniform(0, height - 96)
#         rW = random.uniform(0, width - 96)
#         print('rand h ' + str(rH) + 'rand w ' + str(rW))
#         x, y = int(rH), int(rW)
#         roi = img[x:x + 96, y:y + 96, :]
#         roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         images.append(roi_gray)
#         # cv2.imwrite('data/patches/roi_spoof' + str(i) + '.jpg', roi)
#         i += 1
#     image_arr = np.moveaxis(np.array(images), 0, -1).astype('float32') / 255
#     print('shape ' + str((np.moveaxis(np.array(images), 0, -1).astype('float32') / 255).shape))
#     X_gray[i] = np.moveaxis(np.array(images), 0, -1).astype('float32') / 255
img = cv2.imread('036_spoof/036-1-2-1-12.jpg')
height, width = img.shape[:2]
x = height - 96
y = height - 96
print(img[0:height, 0:width, :])
cv2.imwrite('testimage.jpg', img[x:x+99, y:y+99, :])
