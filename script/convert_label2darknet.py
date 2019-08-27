# convert the dataset label to darknet format
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

dataset_path = "/home/neusoft/amy/AT-201/data/water_mechanical_190516/"
output_path = "/home/neusoft/amy/AT-201/data/water_mechanical_190516_train/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

index = 0
g = os.walk(dataset_path)
for root, dir, files in g:
    for file in files:
        if '.jpg' in file:
            filename, file_extension = os.path.splitext(file)
            filename += '.txt'
            txt = open(root + filename)
            print(txt)
            img = mpimg.imread(root + file)
            #print(type(img))
            #plt.imshow(img, interpolation='nearest')
            #plt.show()

            while(1):
                line = txt.readline()
                #print(line)
                if len(line) == 0:
                    break
                reading = str.split(line)[0]
                x = int(str.split(line)[1])
                y = int(str.split(line)[2])
                w = int(str.split(line)[3])
                h = int(str.split(line)[4])
                print("reading:%s %s %s %s %s" %(reading, x, y, w, h))
                print("%s %s %s %s" % (int(x-w/2), int(x+w/2), int(y-h/2), int(y+h/2)))
                img_figure = img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
                img_figure = rgb2gray(img_figure)
                savepath = output_path + str(reading) + '/'
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                mpimg.imsave(savepath + str(index) + '.jpg', img_figure, cmap='gray')
                index += 1
                #plt.imshow(img_figure, cmap='gray', vmin=0, vmax=255)
                #plt.show()

