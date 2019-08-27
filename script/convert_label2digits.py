# create 0~9 figures' images from darknet format dataset to digits set
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


dataset_path = "/home/neusoft/amy/AT-201/data/electric_mechanical_190516/gray/"
output_path = "/home/neusoft/amy/AT-201/data/electric_mechanical_190516/digits/"

# create folder if not exists
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i in range(10):
    if not os.path.exists(output_path + str(i)):
        os.makedirs(output_path + str(i))

g = os.walk(dataset_path)
for root, dir, files in g:
    for file in files:
        if '.jpg' in file:
            filename, file_extension = os.path.splitext(file)
            txt = open(root + filename + '.txt')
            img = mpimg.imread(root + file)
            print(filename)
            #plt.imshow(img, interpolation='nearest')
            #plt.show()
            index = 1
            while(1):
                line = txt.readline()
                #print(line)
                if len(line) == 0:
                    break
                location = []
                reading = str.split(line)[0]
                #precision = (float)(str.split(line)[5])
                #print("reading:", reading)
                location.append((int)(img.shape[1]*(float)(str.split(line)[1]) - img.shape[1]*(float)(str.split(line)[3])/2))
                location.append((int)(img.shape[0]*(float)(str.split(line)[2]) - img.shape[0]*(float)(str.split(line)[4])/2))
                location.append((int)(img.shape[1]*(float)(str.split(line)[1]) + img.shape[1]*(float)(str.split(line)[3])/2))
                location.append((int)(img.shape[0]*(float)(str.split(line)[2]) + img.shape[0]*(float)(str.split(line)[4])/2))
                #print(location)
                img_figure = img[location[1]:location[3], location[0]:location[2]]
                img_figure = rgb2gray(img_figure)
                #plt.imshow(img_figure, cmap='gray', vmin=0, vmax=255)
                #plt.show()
                #if(precision > 0.9):
                savepath = output_path + reading + '/' + filename + '_' + str(index) + '.jpg'
                #print(savepath)
                mpimg.imsave(savepath, img_figure, cmap='gray')
                index += 1


