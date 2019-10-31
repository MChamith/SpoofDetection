import numpy as np
import cv2
import os
import fnmatch

TRAIN_DIR = '/home/ec2-user/dataset/SIWFaces/SiW_release/Train/'
full_count = 0
delete_count = 0
for root, dirnames, filenames in os.walk(TRAIN_DIR):
    for filename in fnmatch.filter(filenames, "*.jpg"):
        path = os.path.join(root, filename)
        print('path ' + str(path))
        img = cv2.imread(path)
        full_count += 1
        print(full_count)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(e)
            os.remove(path)
            print('removed ' + str(path))
            delete_count +=1
            print(delete_count)
            continue
# matplotlib.use('TkAgg')
# filename = 'videos/live/003/003-1-1-1-1.mov'
# # original_image = img_as_float(Image.open())
# # img = color.rgb2gray(original_image)
# f = open(filename.split('.mov')[0] + '.face')
# cap = cv2.VideoCapture(filename)
# count = 0
# while (True):
#
#     ret, frame = cap.read()
#     k = 1.6
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
#             img = img[y:h, x:w]
#             cv2.imwrite('face.png', img)
#             plt.subplot(3, 3, 1)
#             plt.imshow(img)
#             plt.title('Original Image')
#             for idx, sigma in enumerate([4.0, 8.0, 16.0, 32.0]):
#                 s1 = filters.gaussian(img, k * sigma)
#                 s2 = filters.gaussian(img, sigma)
#
#                 # multiply by sigma to get scale invariance
#                 dog = (s1 - s2)*img
#                 plt.axis('off')
#                 plt.imshow(dog, cmap='gray')
#
#                 plt.savefig('images/live/dog_new/result' + str(count) + '.png', bbox_inches='tight')
#                 # cv2.imwrite('images/live/plots/dog/result' + str(count) + '.png', dog)
#                 plt.subplot(3, 3, idx + 2)
#                 print(dog.min(), dog.max())
#                 plt.imshow(dog, cmap='RdBu')
#                 plt.title('DoG with sigma=' + str(sigma) + ', k=' + str(k))
#
#                 (hist, bins) = np.histogram((dog*255).astype('uint8').ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
#                 hist = hist.astype("float")
#                 hist /= (hist.sum() + 1e-7)
#                 width = 0.7 * (bins[1] - bins[0])
#                 center = (bins[:-1] + bins[1:]) / 2
#                 plt.subplot(3, 3, idx + 6)
#                 plt.bar(center, hist, align='center', width=width)
#                 # plt.imshow()
#                 plt.title('histogram ' + str(sigma))
#
#             # ax = plt.subplot(3, 3, 6)
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
#             plt.savefig('images/live/plots/dog/' + str(count) + '.png', bbox_inches='tight')
#             # plt.imshow(img_blobs)
#             # plt.title('Detected DoG Maxima')
#             # plt.savefig('images/spoof/003_1_3_2_2/' + str(count) + '.png', bbox_inches='tight')
#             count += 1
#
# # plt.show()
