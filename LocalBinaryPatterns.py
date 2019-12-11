# import the necessary packages
import cv2
from skimage import feature
import numpy as np
import matplotlib.pyplot as plt

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        print(lbp.astype('uint8'))
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist, lbp

# desc = LocalBinaryPatterns(24, 8)
# img = cv2.imread('001-1-2-1-1104.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.resize(img_gray, (256,256))
# hist = desc.describe(img_gray)

#
# img2 = cv2.imread('roi3.jpg')
# img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# img_gray2 = cv2.resize(img_gray2, (256,256))
# hist2 = desc.describe(img_gray2)
# print(hist2.shape)
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# axs[0].hist(hist)
# axs[1].hist(hist2)
# plt.show()