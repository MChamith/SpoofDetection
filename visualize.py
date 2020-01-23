import os
import cv2
import numpy as np
import keras
from keras import Model

from difference_of_gaussian import calc_dog
from lbp_extraction import calc_lbp

model = keras.models.load_model('Models/Model-03.h5')
X_nua = []
y_nua = []
X_gray = np.empty((1, 256, 256, 1))
X_dog = np.empty((1, 256, 256, 1))
X_lbp = np.empty((1, 256, 256, 1))
for root, dirs, filenames in os.walk('NUA'):
    for file in filenames:
        if file.endswith('.jpg'):
            print('file ' + str(file))
            path = os.path.join(root, file)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            # cb = ycrcb[0]
            # print('converted to gray')
            dog = calc_dog(gray)
            # print('dog')
            lbp = calc_lbp(gray)
            # print('lbped')
            # gray_img = cv2.resize(gray, (256, 256))
            gray_img = cv2.resize(gray, (256, 256))
            gray_img = np.expand_dims(gray_img, axis=-1)
            dog = np.expand_dims(dog, axis=-1)
            lbp = np.expand_dims(lbp, axis=-1)
            # gray_img = np.concatenate((gray_img, gray_img, gray_img), axis=-1)
            # dog = np.concatenate((dog, dog, dog), axis=-1)
            # lbp = np.concatenate((lbp, lbp, lbp), axis=-1)
            X_gray[0] = gray_img.astype('float32') / 255
            X_dog[0] = dog.astype('float32') / 255
            X_lbp[0] = lbp.astype('float32') / 255

            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer('concatenate_1').output)
            intermediate_output = intermediate_layer_model.predict([X_gray, X_dog, X_lbp])
            print('appending output')
            X_nua.append(intermediate_output)

            if path.strip('.jpg').split('_')[-1] == 'live':
                print('live image')
                y_nua.append(1)
            else:
                print('spoof image')
                y_nua.append(0)

print('saving numpy arrays')
np.save('X_nua.npy', np.array(X_nua))
np.save('y_nua.npy', np.array(y_nua))
