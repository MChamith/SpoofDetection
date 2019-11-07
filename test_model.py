import fnmatch

import keras
from DataGenerator import DataGenerator
from SpoofModel import cnn_model
import os
import numpy as np
from sklearn.utils import shuffle
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

test_data = {'X_test': [], 'label': []}

TEST_DIR = '/home/ubuntu/SpoofDetection/TestData/'

print('collecting training data')
for root, dirnames, filenames in os.walk(TEST_DIR):
    for filename in fnmatch.filter(filenames, "*.jpg"):
        path = os.path.join(root, filename)
        print(path.split('/')[-3])
        test_data['X_test'].append(path)
        if path.split('/')[-3] == 'live':
            test_data['label'].append(1)
        elif path.split('/')[-3] == 'spoof':
            test_data['label'].append(0)

test_data['X_test'], test_data['label'] = shuffle(test_data['X_test'], test_data['label'])
params = {'dim': (256, 256),
          'batch_size': 8,
          'n_channels': 3,
          'shuffle': True}

test_gen = DataGenerator(**params, list_IDs=test_data['X_test'], labels=test_data['label'])

model = keras.models.load_model('Models/Model-03.h5')
metrics = model.evaluate_generator(generator=test_gen)
print('test loss, test accuracy ' + str(metrics))