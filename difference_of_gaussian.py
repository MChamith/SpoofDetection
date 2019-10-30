import matplotlib
import numpy as np
import cv2
from keras.preprocessing import image



def calc_dog(img):

    blur5 = cv2.GaussianBlur(img, (5, 5), 0)
    blur3 = cv2.GaussianBlur(img, (3, 3), 0)
    DoG = blur5 - blur3
    result = cv2.resize(DoG, (256, 256))
    # print('result size ' + str(result.shape) + 'result type ' + str(type(result)))
    return result

# matplotlib.use('TkAgg')
# filename = 'videos/spoofing/003/003-1-3-2-2.mov'
# # original_image = img_as_float(Image.open())
# # img = color.rgb2gray(original_image)
# f = open(filename.split('.mov')[0] + '.face')
# cap = cv2.VideoCapture(filename)
# count = 0
# while True:
#     ret, frame = cap.read()
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
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             # print(img_arry.shape)
#             result = calc_dog(img)
#             cv2.imwrite('images/spoof/dog_new/result'+str(count)+'.jpg', result)
#             count += 1
