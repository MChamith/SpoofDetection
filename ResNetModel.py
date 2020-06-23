import keras
from keras import Model, optimizers
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, Dense, MaxPooling1D, GlobalAveragePooling1D
import cv2
from DataGenerator import DataGenerator
import fnmatch
from sklearn.utils import shuffle
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

res_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                              pooling=None, classes=1000)
x = res_model.output

x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
x = Dense(256, activation='relu')(x)

x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=res_model.input, outputs=x)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
# print(model.summary())
data = {'X_train': [], 'label': []}
test_data = {'X_val': [], 'val_label': [], 'X_test': [], 'test_label': []}
# labels = {'spoof':[]}

TRAIN_DIR = '/home/ubuntu/Spoofing/dataset/LCC_FASD/LCC_FASD_training'
VAL_DIR = '/home/ubuntu/Spoofing/dataset/LCC_FASD/LCC_FASD_development'

# create training data in
print('collecting training data')
for root, dirnames, filenames in os.walk(TRAIN_DIR):
    for filename in fnmatch.filter(filenames, "*.png"):
        path = os.path.join(root, filename)
        img = cv2.imread(path)
        try:
            height, width = img.shape[:-1]
            if height > 224 and width > 224:
                data['X_train'].append(path)
                print(path.split('/')[-2])
                if path.split('/')[-2] == 'real':

                    data['label'].append(1)
                elif path.split('/')[-2] == 'spoof':
                    data['label'].append(0)
        except :
            continue
for root, dirnames, filenames in os.walk(VAL_DIR):
    for filename in fnmatch.filter(filenames, "*.png"):
        path = os.path.join(root, filename)
        img = cv2.imread(path)
        try:
            height, width = img.shape[:-1]
            if height > 224 and width > 224:
                test_data['X_val'].append(path)
                if path.split('/')[-2] == 'real':
                    test_data['val_label'].append(1)
                elif path.split('/')[-2] == 'spoof':
                    test_data['val_label'].append(0)
        except:
            continue
# count = 0
# print('collecting validation and test data')
# for root, dirnames, filenames in os.walk(TEST_DIR):
#     for filename in fnmatch.filter(filenames, "*.jpg"):
#         path = os.path.join(root, filename)
#         if count %10 == 0:
#             test_data['X_val'].append(path)
#             if path.split('/')[-3] == 'live':
#                 test_data['val_label'].append(1)
#             elif path.split('/')[-3] == 'spoof':
#                 test_data['val_label'].append(0)
#         else:
#             test_data['X_test'].append(path)
#             if path.split('/')[-3] == 'live':
#                 test_data['test_label'].append(1)
#             elif path.split('/')[-3] == 'spoof':
#                 test_data['test_label'].append(0)
#         count += 1
print(data)
print('shuffling data')
data['X_train'], data['label'] = shuffle(data['X_train'], data['label'])
# test_data['X_test'], test_data['test_label'] = shuffle(test_data['X_test'], test_data['test_label'])
test_data['X_val'], test_data['val_label'] = shuffle(test_data['X_val'], test_data['val_label'])
print('data shuffled')
params = {'dim': (224, 224),
          'batch_size': 32,
          'n_channels': 3,
          'shuffle': True}

val_params = {'dim': (224, 224),
              'batch_size': 16,
              'n_channels': 3,
              'shuffle': False}

train_gen = DataGenerator(**params, list_IDs=data['X_train'], labels=data['label'])
val_generator = DataGenerator(**val_params, list_IDs=test_data['X_val'], labels=test_data['val_label'])

file_path = 'Checkpoint/ResModel/Model-{epoch:02d}.h5'
check_pointer = ModelCheckpoint(filepath=file_path)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.00001)
early_stop = EarlyStopping(patience=3)
tensorboard_keras = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                                write_graph=True, write_images=True)
model_history = model.fit_generator(generator=train_gen,
                                    epochs=100,
                                    validation_data=val_generator,
                                    callbacks=[check_pointer,
                                               reduce_lr, tensorboard_keras, early_stop],
                                    shuffle=True,
                                    steps_per_epoch=1000, validation_steps=20
                                    )
