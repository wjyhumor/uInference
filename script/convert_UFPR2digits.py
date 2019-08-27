# create 0~9 figures' images from UFPR dataset for training
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

dataset_path = "/home/neusoft/data/UFPR-AMR/validation/"
output_path = "/home/neusoft/data/UFPR-AMR/figures/validation/"

# create folder if not exists
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i in range(10):
    if not os.path.exists(output_path + str(i)):
        os.makedirs(output_path + str(i))

index = 0
g = os.walk(dataset_path)
for root, dir, files in g:
    for file in files:
        if '.jpg' in file:
            filename, file_extension = os.path.splitext(file)
            filename += '.txt'
            txt = open(root + filename)
            img = mpimg.imread(root + file)
            print(type(img))
            #plt.imshow(img, interpolation='nearest')
            # plt.show()

            line = txt.readline()
            line = txt.readline()  # reading
            reading = str.split(line)[1]
            print("reading:", reading)

            temp = int(reading)
            figure_reading = []
            figure_reading.append(int(temp/10000))
            temp -= int(temp/10000) * 10000
            figure_reading.append(int(temp/1000))
            temp -= int(temp/1000) * 1000
            figure_reading.append(int(temp/100))
            temp -= int(temp/100) * 100
            figure_reading.append(int(temp/10))
            temp -= int(temp/10) * 10
            figure_reading.append(int(temp))
            print("figure_reading:", figure_reading)

            line = txt.readline()  # position
            position = []
            temp = str.split(line)
            position.append(int(temp[1]))
            position.append(int(temp[2]))
            position.append(int(temp[3]))
            position.append(int(temp[4]))
            print("position:", position)

            figure = []  # figures
            for i in range(5):
                line = txt.readline()
                pos = str.split(line)
                temp = []
                temp.append(int(pos[2]))
                temp.append(int(pos[3]))
                temp.append(int(pos[4]))
                temp.append(int(pos[5]))
                figure.append(temp)
            print("figure:", figure)

            # print(img.shape)
            counter = img[position[1]:position[1]+position[3] +
                          1, position[0]: position[0]+position[2]+1]
            counter = rgb2gray(counter)
            #plt.imshow(counter, cmap='gray', vmin=0, vmax=255)
            # plt.show()

            for i in range(5):
                img_figure = img[figure[i][1]:figure[i][1] +
                                 figure[i][3]+1, figure[i][0]: figure[i][0]+figure[i][2]+1]
                img_figure = rgb2gray(img_figure)
                mpimg.imsave(
                    output_path + str(figure_reading[i]) + '/' + str(index) + '.jpg', img_figure, cmap='gray')
                index += 1
                #plt.imshow(img_figure, cmap='gray', vmin=0, vmax=255)
                # plt.show()


"""
subprocess.call("rm -rf " + data_folder + output_all, shell=True)
f = open(data_folder + output_all,'w')

    for path,dir_list,file_list in g:
        for file_name in file_list:
                f.write(os.path.join(path, file_name))
                f.write('\n')

image_path = "../Linux/data/11_3.jpg"
img = rgb2gray(mpimg.imread(image_path))

#img = rgb2gray(mpimg.imread(str(line).strip('\n')))

for i in range(len(np.array(img))):
    for j in range(len(np.array(img)[i])):
        print(np.array(img)[i][j], end=",")
print(" ")

m = int(re.search('/\d/', line).group(0).strip('/'))
print(str(m))
"""
