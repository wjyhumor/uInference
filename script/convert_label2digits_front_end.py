# create 0~9 figures' images from darknet format dataset to training set, separate front and last figures
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


dataset_path = "/home/neusoft/amy/AT-201/data/water_mechanical_190625/gray/"
output_path_front = "/home/neusoft/amy/AT-201/data/water_mechanical_190625/digits_front/"
output_path_end = "/home/neusoft/amy/AT-201/data/water_mechanical_190625/digits_end/"

# create folder if not exists
if not os.path.exists(output_path_front):
    os.makedirs(output_path_front)
for i in range(10):
    if not os.path.exists(output_path_front + str(i)):
        os.makedirs(output_path_front + str(i))
# create folder if not exists
if not os.path.exists(output_path_end):
    os.makedirs(output_path_end)
for i in range(10):
    if not os.path.exists(output_path_end + str(i)):
        os.makedirs(output_path_end + str(i))

g = os.walk(dataset_path)
for root, dir, files in g:
    for file in files:
        if '.jpg' in file:
            filename, file_extension = os.path.splitext(file)
            txt = open(root + filename + '.txt')
            img = mpimg.imread(root + file)
            #print(filename)
            #plt.imshow(img, interpolation='nearest')
            #plt.show()
            index = 0
            readings = []
            locations = []
            imgs = []
            while(1):
                line = txt.readline()
                #print(line)
                if len(line) == 0:
                    break
                location = []
                reading = str.split(line)[0]
                readings.append(reading)
                #precision = (float)(str.split(line)[5])
                #print("reading:", reading)

                location.append((int)(img.shape[1]*(float)(str.split(line)[1]) - img.shape[1]*(float)(str.split(line)[3])/2))
                location.append((int)(img.shape[0]*(float)(str.split(line)[2]) - img.shape[0]*(float)(str.split(line)[4])/2))
                location.append((int)(img.shape[1]*(float)(str.split(line)[1]) + img.shape[1]*(float)(str.split(line)[3])/2))
                location.append((int)(img.shape[0]*(float)(str.split(line)[2]) + img.shape[0]*(float)(str.split(line)[4])/2))
                locations.append(location)

                img_figure = img[location[1]:location[3], location[0]:location[2]]
                #img_figure = rgb2gray(img_figure)
                imgs.append(img_figure)
                #plt.imshow(img_figure, cmap='gray', vmin=0, vmax=255)
                #plt.show()
            max = locations[-1][0]
            end_index = len(locations)-1
            for i in range(len(locations)-1):
                if locations[i][0] > max:
                    max = locations[i][0]
                    end_index = i
            if end_index != 4:
                print(end_index)
                print(filename)
            for i in range(len(locations)):
                if i != end_index:
                    savepath = output_path_front + readings[i] + '/' + filename + '_' + str(index) + '.jpg'
                    #print(savepath)
                    #print(i)
                    mpimg.imsave(savepath, imgs[i], cmap='gray')
                    index += 1
            savepath = output_path_end + readings[end_index] + '/' + filename + '_' + str(index) + '.jpg'
            mpimg.imsave(savepath, imgs[end_index], cmap='gray')
            #print(savepath)


