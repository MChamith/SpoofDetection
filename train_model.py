import fnmatch

import keras
from keras import optimizers

from DataGenerator import DataGenerator
from SpoofModel import cnn_model
import os
import numpy as np
from sklearn.utils import shuffle
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

data = {'X_train': [], 'label': []}
test_data = {'X_val': [], 'val_label': [], 'X_test': [], 'test_label': []}
# labels = {'spoof':[]}

TRAIN_DIR = '/home/ubuntu/volume/SiW_release/Train/'
TEST_DIR = '/home/ubuntu/volume/SiW_release/Test/'

# create training data in
print('collecting training data')
for root, dirnames, filenames in os.walk(TRAIN_DIR):
    for filename in fnmatch.filter(filenames, "*.jpg"):
        path = os.path.join(root, filename)
        # print(path)
        data['X_train'].append(path)
        if path.split('/')[-3] == 'live':
            data['label'].append(1)
        elif path.split('/')[-3] == 'spoof':
            data['label'].append(0)

count = 0
print('collecting validation and test data')
for root, dirnames, filenames in os.walk(TEST_DIR):
    for filename in fnmatch.filter(filenames, "*.jpg"):
        path = os.path.join(root, filename)
        if count % 10 == 0:
            test_data['X_val'].append(path)
            if path.split('/')[-3] == 'live':
                test_data['val_label'].append(1)
            elif path.split('/')[-3] == 'spoof':
                test_data['val_label'].append(0)
        else:
            test_data['X_test'].append(path)
            if path.split('/')[-3] == 'live':
                test_data['test_label'].append(1)
            elif path.split('/')[-3] == 'spoof':
                test_data['test_label'].append(0)
        count += 1
print('shuffling data')
data['X_train'], data['label'] = shuffle(data['X_train'], data['label'])
test_data['X_test'], test_data['test_label'] = shuffle(test_data['X_test'], test_data['test_label'])
test_data['X_val'], test_data['val_label'] = shuffle(test_data['X_val'], test_data['val_label'])
print('data shuffled')
params = {'dim': (256, 256),
          'batch_size': 32,
          'n_channels': 3,
          'shuffle': True}

val_params = {'dim': (256, 256),
              'batch_size': 16,
              'n_channels': 3,
              'shuffle': False}

train_gen = DataGenerator(**params, list_IDs=data['X_train'], labels=data['label'])
val_generator = DataGenerator(**params, list_IDs=test_data['X_val'], labels=test_data['val_label'])
test_generator = DataGenerator(**params, list_IDs=test_data['X_test'], labels=test_data['test_label'])

model = cnn_model()
print('compiling model')
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
print('model compiled')
file_path = 'Checkpoint/SGDModel1/Model-{epoch:02d}.h5'
check_pointer = ModelCheckpoint(filepath=file_path)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.00001)
early_stop = EarlyStopping(patience=2)
tensorboard_keras = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                                write_graph=True, write_images=True)

model_history = model.fit_generator(generator=train_gen,
                                    epochs=100,
                                    validation_data=val_generator,
                                    callbacks=[check_pointer,
                                               reduce_lr, tensorboard_keras, early_stop],
                                    shuffle=True,
                                    steps_per_epoch=1000, validation_steps=40
                                    )
