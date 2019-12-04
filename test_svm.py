# import cv2
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import LinearSVC
# import pickle
#
# from LocalBinaryPatterns import LocalBinaryPatterns
#
# filename = 'IMG_1754.MOV'
# cap = cv2.VideoCapture(filename)
# face_cascade = cv2.CascadeClassifier(
#     '/home/chamith/Documents/Project/msid_server/venv/lib/python3.6/site-packages/cv2/data'
#     '/haarcascade_frontalface_default.xml')
#
# desc = LocalBinaryPatterns(24, 8)
#
# live_count = 0
# spoof_count = 0
# while True:
#     # file = '036_spoof/'+ str(file)
#     ret, frame = cap.read()
#     # print(ret)
#     # print(frame)
#     # print(file)
#     # frame = cv2.imread(file)
#     # print('height ' + str(frame.shape[0]) + 'width ' + str(frame.shape[1]))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     # faces = face_recognition.face_locations(frame.copy(), number_of_times_to_upsample=0, model="cnn")
#     # print(faces)
#     if faces is not None:
#         for face in faces:
#             # if True:
#             x = face[0]
#             y = face[1]
#             w = face[2]
#             h = face[3]
#             roi = frame[y:y+h, x:x+w]
#             cv2.imshow('roi', roi)
#             img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#             img_gray = cv2.resize(img_gray, (256, 256))
#             # print(img_gray)
#             img_gray = img_gray.astype('uint8')
#             hist, lbp = desc.describe(img_gray)
#             cv2.imshow('lbp', lbp.astype('uint8'))
#             scaler = pickle.load(open('scaler.pickle', 'rb'))
#             model = pickle.load(open('model.pickle', 'rb'))
#             hist = scaler.transform(hist.reshape(1, -1))
#             prediction = model.predict(hist)
#             print(prediction)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

import pickle

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

with open('test_data.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
    test_label = pickle.load(handle)
    X_val = pickle.load(handle)
    val_label = pickle.load(handle)
model = pickle.load(open('model2.pickle', 'rb'))
predictions = model.predict(X_test)
print(accuracy_score(predictions, test_label))
