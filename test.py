import os

import cv2
import keras
import tensorflow as tf
from keras import models

from difference_of_gaussian import calc_dog
from lbp_extraction import calc_lbp
import numpy as np
# import face_recognition
import matplotlib.pyplot as plt

filename = '/home/ubuntu/volume/data/IMG_1759.MOV'
cap = cv2.VideoCapture(filename)
face_cascade = cv2.CascadeClassifier('/home/ubuntu/volume/AntiSpoofing/haarcascade_frontalface_default.xml')
X_gray = np.empty((1, 224, 224, 3))
X_dog = np.empty((1, 224, 224, 3))
X_lbp = np.empty((1, 224, 224, 3))
# X = np.empty((1, 224, 224, 3))
result = np.zeros(2)
model = keras.models.load_model('/home/ubuntu/volume/SpoofDetection/Checkpoint/VGG_fusion/Model-03.h5')
# print(model.summary())
# while True:
# for file in os.listdir('036_spoof'):
live_count = 0
spoof_count = 0
count = 0
while True:
    # file = 'data/Skype_Picture_2019_11_21T09_58_26_954Z.jpeg'
    ret, frame = cap.read()
    # cv2.imshow('frame' , frame)
    # print(ret)
    # print(frame)
    # print(file)
    # frame = cv2.imread(file)
    # print(frame)
    # print('height ' + str(frame.shape[0]) + 'width ' + str(frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces = face_recognition.face_locations(frame.copy(), number_of_times_to_upsample=0, model="cnn")
    # print(faces)
    if faces is not None:
    # if True:
        for face in faces:
        # if True:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            roi = frame[y:y+h, x:x+w]
            # roi = frame
            if count == 0:
                cv2.imwrite('roi.jpg', roi)
            count +=1
            # top, right, bottom, left = face
            # roi = frame[top:bottom, left:right]
            # print(roi)
            # roi = cv2.imread(file)
            # roi = cv2.resize(roi, (224, 224))
            # roi = roi.astype('float32')/255
            # X[0] = roi.astype('float32')/255
            # roi = cv2.resize(roi, (224, 224))
            # cv2.imshow('frame', roi)

            # cv2.imshow('roi', roi)
            # cv2.imwrite('roi3.jpg', roi)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dog = calc_dog(roi_gray)

            lbp = calc_lbp(roi_gray)
            # print(dog)

            gray_img = cv2.resize(roi_gray, (224, 224))

            gray_img = np.expand_dims(gray_img, axis=-1)
            dog = np.expand_dims(dog, axis=-1)
            lbp = np.expand_dims(lbp, axis=-1)
            gray_img = np.concatenate((gray_img, gray_img, gray_img), axis=-1)
            dog = np.concatenate((dog, dog, dog), axis=-1)
            lbp = np.concatenate((lbp, lbp, lbp), axis=-1)
            cv2.imwrite('gray.jpg', gray_img)
            cv2.imwrite('lbp.jpg', lbp)
            cv2.imwrite('dog.jpg', dog)
            # cv2.imshow('lbp', lbp)
            # cv2.imshow('roi', gray_img)
            # cv2.imshow('dog', dog)
            X_gray[0] = gray_img.astype('float32') / 255
            X_dog[0] = dog.astype('float32') / 255
            X_lbp[0] = lbp.astype('float32') / 255
            prediction = model.predict([X_gray, X_dog, X_lbp])
            # prediction = model.predict(X_lbp)
            #
            # if np.argmax(prediction) == 1:
            #     print('live ' + str(prediction))
            #     live_count += 1
            # else:
            #     print('spoofing ' + str(prediction))
            #     spoof_count +=1

            if prediction > 0.5:
                print('live ' + str(prediction))
                live_count += 1
            else:
                print('spoofing ' + str(prediction))
                spoof_count += 1

            # print(prediction)
            # print(np.argmax(prediction))
            # result[np.argmax(prediction)] = result[np.argmax(prediction)] + 1
            # print(result)
            # print('argmax ' + str(np.argmax(result)))
            # if np.argmax(prediction) == 0:
            #     print('spoofing')
            # else:
            #     print('live')
            # print(model.layers)
            # layer_outputs = [layer.output for layer in model.layers[3:-4]] # Extracts the outputs of the top 12 layers
            # activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model inpu
            # activations = activation_model.predict([X_gray, X_dog, X_lbp])
            # layer_names = []
            # for layer in model.layers:
            #     layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot
            # images_per_row = 16
            # for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            #     n_features = layer_activation.shape[-1]  # Number of features in the feature map
            #     size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
            #     n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
            #     display_grid = np.zeros((size * n_cols, images_per_row * size))
            #     for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            #         for row in range(images_per_row):
            #             channel_image = layer_activation[0,
            #                             :, :,
            #                             col * images_per_row + row]
            #             channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            #             channel_image /= channel_image.std()
            #             channel_image *= 64
            #             channel_image += 224
            #             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            #             display_grid[col * size: (col + 1) * size,  # Displays the grid
            #             row * size: (row + 1) * size] = channel_image
            #     scale = 1. / size
            #     plt.figure(figsize=(scale * display_grid.shape[1],
            #                         scale * display_grid.shape[0]))
            #     plt.title(layer_name)
            #     plt.grid(False)
            #     plt.imshow(display_grid, aspect='auto', cmap='viridis')
            #     plt.savefig('intermediate_features/live' +str(layer_name)+'.jpg')
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        # cv2.waitKey(0)
    else:
        print('no face found ')
print('live ' + str(live_count) + ' spoof ' + str(spoof_count) )
cap.release()
cv2.destroyAllWindows()

# y = np.empty((8), dtype=int)
#
# y = [0, 1 ,1, 0, 0, 0, 1, 1]
# result = keras.utils.to_categorical(y, num_classes=2)
# print(result)
