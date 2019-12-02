import fnmatch
import pickle
import cv2
import numpy as np

from LocalBinaryPatterns import LocalBinaryPatterns
import os

TRAIN_DIR = '/home/ubuntu/volume/SiW_release/Train/'
TEST_DIR = '/home/ubuntu/volume/SiW_release/Test/'
train_data = []
label = []
desc = LocalBinaryPatterns(24, 8)
count = 0
for root, dirnames, filenames in os.walk(TRAIN_DIR):
    for filename in fnmatch.filter(filenames, "*.jpg"):
        if count % 40 == 0:
            print(filename)
            path = os.path.join(root, filename)
            # print(path)
            img = cv2.imread(path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (256, 256))
            hist = desc.describe(img_gray)
            train_data.append(hist)
            if path.split('/')[-3] == 'live':
                label.append(1)
            elif path.split('/')[-3] == 'spoof':
                label.append(0)
        count +=1

print('dumping pickle')
with open('train_data.pickle', 'wb') as handle:
    pickle.dump(train_data, handle)
    pickle.dump(label, handle)