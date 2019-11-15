import cv2
import keras
import tensorflow as tf
from difference_of_gaussian import calc_dog
from lbp_extraction import calc_lbp
import numpy as np

filename = 'IMG_1754.MOV'
cap = cv2.VideoCapture(filename)
face_cascade = cv2.CascadeClassifier(
                '/home/chamith/Documents/Project/msid_server/venv/lib/python3.6/site-packages/cv2/data'
            '/haarcascade_frontalface_default.xml')
X_gray = np.empty((1, 256, 256, 1))
X_dog = np.empty((1, 256, 256, 1))
X_lbp = np.empty((1, 256, 256, 1))
result = np.zeros(2)
while True:
    # ret, frame = cap.read()
    frame = cv2.imread('passport (2).png')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    if faces is not None:
        for face in faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            roi = frame[y:y + h, x:x + w]
            # roi = cv2.imread('098D97B7-0209-4683-8598-CCE9DACE5EEA.jpg')
            cv2.imshow('roi', roi)
            cv2.imwrite('roi.jpg', roi)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dog = calc_dog(roi_gray)
            cv2.imwrite('dog.jpg', dog)
            lbp = calc_lbp(roi)
            cv2.imwrite('lbp.jpg', lbp)
            gray_img = cv2.resize(roi_gray, (256, 256))
            gray_img = np.expand_dims(gray_img, axis=-1)
            dog = np.expand_dims(dog, axis=-1)
            lbp = np.expand_dims(lbp, axis=-1)
            X_gray[0] = gray_img.astype('float32') / 255
            X_dog[0] = dog.astype('float32') / 255
            X_lbp[0] = lbp.astype('float32') / 255
            model = keras.models.load_model('Models/Model-03.h5')

            prediction = model.predict([X_gray, X_dog, X_lbp])
            print(prediction)
            print(np.argmax(prediction))
            result[np.argmax(prediction)] = result[np.argmax(prediction)] +1
            print(result)
            print('argmax ' + str(np.argmax(result)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('no face found ')
cap.release()
cv2.destroyAllWindows()

# y = np.empty((8), dtype=int)
#
# y = [0, 1 ,1, 0, 0, 0, 1, 1]
# result = keras.utils.to_categorical(y, num_classes=2)
# print(result)