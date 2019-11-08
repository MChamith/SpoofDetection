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
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces is not None:
        for face in faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            roi = frame[y:y + h, x:x + w]
            cv2.imshow('roi', roi)

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dog = calc_dog(roi_gray)
        lbp = calc_lbp(roi)
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# y = np.empty((8), dtype=int)
#
# y = [0, 1 ,1, 0, 0, 0, 1, 1]
# result = keras.utils.to_categorical(y, num_classes=2)
# print(result)