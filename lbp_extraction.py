from skimage import feature
import numpy as np

from skimage import data, feature, color, filters, img_as_float
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib
import cv2


def calc_lbp(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lbp = feature.local_binary_pattern(hsv[:, :, 1], 8,
                                       1, method="default")
    lbp = cv2.resize(lbp, (256, 256))
    return lbp
# matplotlib.use('TkAgg')
# filename = 'videos/live/003/003-1-1-1-1.mov'
# # original_image = img_as_float(Image.open())
# # img = color.rgb2gray(original_image)
# f = open(filename.split('.mov')[0] + '.face')
# cap = cv2.VideoCapture(filename)
# count = 0
# while (True):
#     ret, frame = cap.read()
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # img = color.rgb2gray(frame)
#     read_line = f.readline()
#     print(read_line)
#     if read_line == '':
#         break_the_loop = True
#     else:
#         print(read_line)
#         try:
#             x, y, w, h = map(int, read_line.strip().split(' '))
#         except:
#             print('exception')
#             pass
#         print('x= ' + str(x) + ' y= ' + str(y) + ' w= ' + str(w) + ' h= ' + str(h))
#         if w and h:
#             img = frame[y:h, x:w]
#             # img = cv2.imread('face (4).jpg')
#             # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             # plt.subplot(2, 2, 1)
#             # plt.imshow(img)
#             # plt.title('Original Image')
#
#             # for idx, sigma in enumerate([4.0, 8.0, 16.0, 32.0]):
#             #     s1 = filters.gaussian(img, k * sigma)
#             #     s2 = filters.gaussian(img, sigma)
#             #
#             #     # multiply by sigma to get scale invariance
#             #     dog = (s1 - s2)*img
#             #     plt.subplot(2, 3, idx + 2)
#             #     print(dog.min(), dog.max())
#             #     plt.imshow(dog, cmap='RdBu')
#             #     plt.title('DoG with sigma=' + str(sigma) + ', k=' + str(k))
#
#             # ax = plt.subplot(2, 3, 6)
#             # blobs_dog = [(x[0], x[1], x[2]) for x in
#             #              feature.blob_dog(img, min_sigma=4, max_sigma=32, threshold=0.5, overlap=1.0)]
#             # # skimage has a bug in my version where only maxima were returned by the above
#             # blobs_dog += [(x[0], x[1], x[2]) for x in
#             #               feature.blob_dog(-img, min_sigma=4, max_sigma=32, threshold=0.5, overlap=1.0)]
#             #
#             # # remove duplicates
#             # blobs_dog = set(blobs_dog)
#             #
#             # img_blobs = color.gray2rgb(img)
#             # for blob in blobs_dog:
#             #     y, x, r = blob
#             #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
#             #     ax.add_patch(c)
#
#             # plt.subplot(2, 2, 2)
#             # plt.imshow(lbp)
#             # plt.title('lbp Image')
#
#             # plt.subplot(2, 2, 3)
#             # plt.imshow(lbp)
#             # plt.title('lbps Image')
#             # (hist, bins) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
#             # hist = hist.astype("float")
#             # hist /= (hist.sum() + eps)
#             # width = 0.7 * (bins[1] - bins[0])
#             # center = (bins[:-1] + bins[1:]) / 2
#             # plt.subplot(2, 2, 3)
#             # plt.bar(center, hist, align='center', width=width)
#             # plt.imshow()
#             plt.axis('off')
#             plt.imshow(lbp, cmap='gray')
#             plt.savefig('images/live/lbp_new/result' + str(count) + '.png', bbox_inches='tight')
#             count += 1
#
# # plt.axis('off')
# # plt.imshow(result, cmap='gray')
# #
# # plt.savefig('images/live/dog_new/result' + str(suffix) + '.png', bbox_inches='tight')
# # plt.show()
#
# # plt.show()
#
#
# #
# # (hist, _) = np.histogram(lbp.ravel(),
# #                          bins=np.arange(0, self.numPoints + 3),
# #                          range=(0, self.numPoints + 2))
# #
# # # normalize the histogram
# # hist = hist.astype("float")
# # hist /= (hist.sum() + eps)
#
# # return the histogram of Local Binary Patterns
