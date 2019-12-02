import fnmatch
import pickle
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from LocalBinaryPatterns import LocalBinaryPatterns
import os

TRAIN_DIR = '/home/ubuntu/volume/SiW_release/Train/'
TEST_DIR = '/home/ubuntu/volume/SiW_release/Test/'
# train_data = []
# label = []
desc = LocalBinaryPatterns(24, 8)
# count = 0
# for root, dirnames, filenames in os.walk(TRAIN_DIR):
#     for filename in fnmatch.filter(filenames, "*.jpg"):
#         if count % 40 == 0:
#             print(filename)
#             path = os.path.join(root, filename)
#             # print(path)
#             img = cv2.imread(path)
#             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img_gray = cv2.resize(img_gray, (256, 256))
#             hist = desc.describe(img_gray)
#             train_data.append(hist)
#             if path.split('/')[-3] == 'live':
#                 label.append(1)
#             elif path.split('/')[-3] == 'spoof':
#                 label.append(0)
#         count +=1
#         print(label)
#
# print('dumping pickle')
# with open('train_data.pickle', 'wb') as handle:
#     pickle.dump(train_data, handle)
#     pickle.dump(label, handle)
count = 0
test_data = []
test_label = []
val_label = []
val_data = []
for root, dirnames, filenames in os.walk(TEST_DIR):
    for filename in fnmatch.filter(filenames, "*.jpg"):
        path = os.path.join(root, filename)
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (256, 256))
        hist = desc.describe(img_gray)
        if count % 100 == 0 and count % 110 != 0:
            print('val file ' + str(filename))
            val_data.append(hist)
            if path.split('/')[-3] == 'live':
                val_label.append(1)
            elif path.split('/')[-3] == 'spoof':
                val_label.append(0)
        elif count %110 == 0 and count %100 !=0:
            print('test file ' + str(filename))
            test_data.append(hist)
            if path.split('/')[-3] == 'live':
                test_label.append(1)
            elif path.split('/')[-3] == 'spoof':
                test_label.append(0)
        count +=1

# Fit on training set only.
print('dumping pickle')
with open('test_data.pickle', 'wb') as handle:
    pickle.dump(test_data, handle)
    pickle.dump(test_label, handle)
    pickle.dump(val_data, handle)
    pickle.dump(val_label, handle)

print('pickle dumped')