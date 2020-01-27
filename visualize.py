import fnmatch
import os
from sklearn.utils import shuffle
import random
import cv2
import numpy as np
import keras
from keras import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from difference_of_gaussian import calc_dog
from lbp_extraction import calc_lbp
import matplotlib.patheffects as PathEffects

TEST_DIR = '/home/ubuntu/volume/SiW_release/Test/'
model = keras.models.load_model('Models/Model-03.h5')


def siw_data_create():
    X_siw_paths = []
    X_siw = []
    y_siw = []
    X_gray = np.empty((1, 256, 256, 1))
    X_dog = np.empty((1, 256, 256, 1))
    X_lbp = np.empty((1, 256, 256, 1))
    count = 0
    print('collecting validation and test data')
    for root, dirnames, filenames in os.walk(TEST_DIR):
        for filename in fnmatch.filter(filenames, "*.jpg"):
            path = os.path.join(root, filename)

            if path.split('/')[-3] == 'live':
                X_siw_paths.append(path)
                y_siw.append(1)
            elif path.split('/')[-3] == 'spoof':
                X_siw_paths.append(path)
                y_siw.append(0)
            count += 1
    X_siw_paths, y_siw = shuffle(X_siw_paths, y_siw)

    pairs = list(zip(X_siw_paths, y_siw))  # make pairs out of the two lists
    pairs = random.sample(pairs, 1500)  # pick 3 random pairs
    filepaths, labels = zip(*pairs)

    for image_path in filepaths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        # cb = ycrcb[0]
        # print('converted to gray')
        dog = calc_dog(gray)
        # print('dog')
        lbp = calc_lbp(gray)
        # print('lbped')
        # gray_img = cv2.resize(gray, (256, 256))
        gray_img = cv2.resize(gray, (256, 256))
        gray_img = np.expand_dims(gray_img, axis=-1)
        dog = np.expand_dims(dog, axis=-1)
        lbp = np.expand_dims(lbp, axis=-1)
        # gray_img = np.concatenate((gray_img, gray_img, gray_img), axis=-1)
        # dog = np.concatenate((dog, dog, dog), axis=-1)
        # lbp = np.concatenate((lbp, lbp, lbp), axis=-1)
        X_gray[0] = gray_img.astype('float32') / 255
        X_dog[0] = dog.astype('float32') / 255
        X_lbp[0] = lbp.astype('float32') / 255

        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer('concatenate_1').output)
        intermediate_output = intermediate_layer_model.predict([X_gray, X_dog, X_lbp])
        print('appending output')
        X_siw.append(intermediate_output)
        #
        # if path.strip('.jpg').split('_')[-1] == 'live':
        #     print('live image')
        #     y_siw.append(1)
        # else:
        #     print('spoof image')
        #     y_nua.append(0)

    print('saving numpy arrays')
    np.save('X_siw.npy', np.array(X_siw))
    np.save('y_siw.npy', np.array(labels))


def nua_data_create():
    count = 0
    X_nua = []
    y_nua = []
    X_gray = np.empty((1, 256, 256, 1))
    X_dog = np.empty((1, 256, 256, 1))
    X_lbp = np.empty((1, 256, 256, 1))
    for root, dirs, filenames in os.walk('NUA'):
        for file in filenames:
            if file.endswith('.jpg'):
                if count % 10 == 0:
                    print('file ' + str(file))
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
                    # cb = ycrcb[0]
                    # print('converted to gray')
                    dog = calc_dog(gray)
                    # print('dog')
                    lbp = calc_lbp(gray)
                    # print('lbped')
                    # gray_img = cv2.resize(gray, (256, 256))
                    gray_img = cv2.resize(gray, (256, 256))
                    gray_img = np.expand_dims(gray_img, axis=-1)
                    dog = np.expand_dims(dog, axis=-1)
                    lbp = np.expand_dims(lbp, axis=-1)
                    # gray_img = np.concatenate((gray_img, gray_img, gray_img), axis=-1)
                    # dog = np.concatenate((dog, dog, dog), axis=-1)
                    # lbp = np.concatenate((lbp, lbp, lbp), axis=-1)
                    X_gray[0] = gray_img.astype('float32') / 255
                    X_dog[0] = dog.astype('float32') / 255
                    X_lbp[0] = lbp.astype('float32') / 255

                    intermediate_layer_model = Model(inputs=model.input,
                                                     outputs=model.get_layer('concatenate_1').output)
                    intermediate_output = intermediate_layer_model.predict([X_gray, X_dog, X_lbp])
                    print('appending output')
                    X_nua.append(intermediate_output)

                    if path.strip('.jpg').split('_')[-1] == 'live':
                        print('live image')
                        y_nua.append(1)
                    else:
                        print('spoof image')
                        y_nua.append(0)
                count += 1
    print('saving numpy arrays')
    np.save('X_nua.npy', np.array(X_nua))
    np.save('y_nua.npy', np.array(y_nua))


def scatter(x, colors, dataset):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 4))
    if str(dataset) == 'Siw':
        pallet_color = palette[colors.astype(np.int)]
    if str(dataset) == 'Nua':
        pallet_color = palette[colors.astype(np.int)+2]
    # We create a scatter plot.
    f = plt.figure(figsize=(48, 48))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=100,
                    c=pallet_color)
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(2):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(dataset) + '_' + str(i), fontsize=64)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

#
# siw_data_create()
# nua_data_create()

X_siw = np.load('X_siw.npy')
y_siw = np.load('y_siw.npy')
X_nua = np.load('X_nua.npy')
y_nua = np.load('y_nua.npy')

# X_siw = StandardScaler().fit_transform(X_siw)

X_siw = X_siw.reshape(X_siw.shape[0], -1)
X_nua = X_nua.reshape(X_nua.shape[0], -1)

X_siw_embedded = TSNE(n_components=2).fit_transform(X_siw)
X_nua_embedded = TSNE(n_components=2).fit_transform(X_nua)

scatter(X_siw_embedded, y_siw, 'Siw')
scatter(X_nua_embedded, y_nua , 'Nua')
plt.savefig('tsne-plot.png', dpi=120)
