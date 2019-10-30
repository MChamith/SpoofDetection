import fnmatch

# import DataGenerator
# from SpoofModel import cnn_model
import json
import os
import tqdm
import numpy as np

data = {'train': [], 'label': []}
# labels = {'spoof':[]}

TRAIN_DIR = '/home/ec2-user/dataset/SIWFaces/SiW_release/Train/'

for root, dirnames, filenames in os.walk(TRAIN_DIR):
    for filename in fnmatch.filter(filenames, "*.jpg"):
        path = os.path.join(root, filename)
        # print(path)
        data['train'].append(path)
        if path.split('/')[-3] == 'live':
            data['label'].append(1)
            print('path ' + str(path) + ' label 1')
        elif path.split('/')[-3] == 'spoof':
            data['label'].append(0)
            print('path ' + str(path) + ' label 0')

#       img = image.load_img(path, target_size=(128,128), grayscale = True)
#       img_array = image.img_to_array(img)
#       train_data.append(np.array(img_array))

#   np.save('training_data.npy' , train_data)

print(data['label'])
# path = 'home/ec2-user/dataset/SIWFaces/SiW_release/Train/live/003/1231.jpg'
# print(path.split('/')[-3])
# with open('data.txt', 'w') as file:
#     file.write(json.dumps(data))
# with open('labels.txt', 'w') as file:
#     file.write(json.dumps(labels))
