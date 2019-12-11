import pickle

import cv2
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# with open("train_data.pickle", "rb") as myfile:
#     X_train = pickle.load(myfile)
#     label = pickle.load(myfile)
#
# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(X_train)
# principalDf = pd.DataFrame(data=principalComponents
#                            , columns=['x1', 'x2', 'x3',
#                                       'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
# # print(principalDf.head())
#
# scatter_matrix(principalDf, alpha=0.2, figsize=(10, 10), diagonal='kde', c = ["r" if y == 0 else "b" for y in label])
# plt.show()

import numpy as np
import cv2 as cv
# import argparse
# # parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
# #                                               The example file can be downloaded from: \
# #                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# # parser.add_argument('image', type=str, help='path to image file')
# # args = parser.parse_args()
# cap = cv.VideoCapture('IMG_1834.MOV')
# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# # Create some random colors
# color = np.random.randint(0,255,(100,3))
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
# while(1):
#     ret,frame = cap.read()
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # Select good points
#     good_new = p1[st==1]
#     good_old = p0[st==1]
#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new, good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
#     img = cv.add(frame,mask)
#     cv.imshow('frame',img)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)

cap = cv.VideoCapture('IMG_1800.MOV')
ret, frame1 = cap.read()
print(frame1.shape)
frame1 = cv.resize(frame1, (260,260))
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
prvs = cv.resize(prvs, (260, 260))
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
face_cascade = cv2.CascadeClassifier(
    '/home/chamith/Documents/Project/msid_server/venv/lib/python3.6/site-packages/cv2/data'
    '/haarcascade_frontalface_default.xml')
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    # # faces = face_recognition.face_locations(frame.copy(), number_of_times_to_upsample=0, model="cnn")
    # # print(faces)
    # if faces is not None:
    #     for face in faces:
    #         # if True:
    #         x = face[0]
    #         y = face[1]
    #         w = face[2]
    #         h = face[3]
    # next = img_gray[y:y + h, x:x + w]
    next = cv.resize(next, (260, 260))
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    print(hsv.shape)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    bgr = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next