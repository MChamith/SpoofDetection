import cv2
import numpy as np
import keras
from difference_of_gaussian import calc_dog
from lbp_extraction import calc_lbp
from sklearn.utils import shuffle
from utils import random_crop


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # print(index)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print('indexes ' + str(indexes))
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # print(list_IDs_temp)

        # Generate data

        # [X_g, y = self.__data_generation(list_IDs_temp)
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        with open('log.txt', 'a') as lf:
            lf.write('epoch end\n shuffling data\n')
        if self.shuffle:
            shuffle(self.list_IDs, self.labels)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X_gray = np.empty((self.batch_size, 256, 256, 1))
        # X_dog = np.empty((self.batch_size, 256, 256, 1))
        # X_lbp = np.empty((self.batch_size, 256, 256, 1))

        # for resnet
        X = np.empty((self.batch_size, 224, 224, 3))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            img = cv2.imread(ID)
            print('image ' +str(img))
            img = random_crop(img, (224, 224))
            X[i] = img
            # img = cv2.resize(img, (256, 256))
            # img = cv2.resize(img, (256, 256))   # resnet
            # img = np.expand_dims(img, axis=-1)`
            idx = self.list_IDs.index(ID)
            y[i] = self.labels[idx]
            # print('id' + str(ID) +'label ' + str(self.labels[idx]))
            # try:
            #     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            #     # cb = ycrcb[0]
            #     # # print('converted to gray')
            #     # dog = calc_dog(gray)
            #     # # print('dog')
            #     # lbp = calc_lbp(gray)
            #     # # print('lbped')
            #     # # gray_img = cv2.resize(gray, (256, 256))
            #     # gray_img = cv2.resize(cb, (256, 256))
            #     # gray_img = np.expand_dims(gray_img, axis=-1)
            #     # dog = np.expand_dims(dog, axis=-1)
            #     # lbp = np.expand_dims(lbp, axis=-1)
            #     # # gray_img = np.concatenate((gray_img, gray_img, gray_img), axis=-1)
            #     # # dog = np.concatenate((dog, dog, dog), axis=-1)
            #     # # lbp = np.concatenate((lbp, lbp, lbp), axis=-1)
            #     # X_gray[i] = gray_img.astype('float32') / 255
            #     # X_dog[i] = dog.astype('float32') / 255
            #     # X_lbp[i] = lbp.astype('float32') / 255
            #     # # Store class
            #     # # X[i] = img.astype('float32')/255
            #     #
            #     # y[i] = self.labels[idx]
            #
            #     # with open('log.txt', 'a') as lf:
            #     #     lf.write('ID ' + str(ID) + ' label ' + str(y[i])+'\n')
            #     # print('y[' + str(i) + ']= ' + str(y[i]))
            # except cv2.error as e:
            #     print(e)
            #     # print('skipping id')
            #     continue
        # print('yyy = ' +str(keras.utils.to_categorical(y, num_classes=self.n_classes)))
        # print('X_gray = ' + str(X_gray))
        # print('X_dog = ' + str(X_dog))
        # print('X_lbp = ' + str(X_lbp))
        # return [X_gray, X_dog, X_lbp], keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y