import keras
from keras import Model, optimizers
from keras.applications.xception import xception
from keras.layers import GlobalAveragePooling2D, Dense, MaxPooling1D, GlobalAveragePooling1D

from DataGenerator import DataGenerator
import fnmatch
from sklearn.utils import shuffle
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os