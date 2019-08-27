# convert the dataset label(get from Yolov3) to darknet format
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

dataset_path = "/home/neusoft/amy/AT-201/data/electric_mechanical_190412_2/electric_mechanical_190412_2_result/"

g = os.walk(dataset_path)
for root, dir, files in g:
    for file in files:
        if '.jpg' in file:
            filename, file_extension = os.path.splitext(file)
            filename += '.txt'
            txt = open(root + filename)
            print(txt)
            img = mpimg.imread(root + file)
            img_height = img.shape[0]
            img_width = img.shape[1]
            outfile = open(root + "a.txt", "a")
            while(1):
                line = txt.readline()
                #print(line)
                if len(line) == 0:
                    break
                reading = int(str.split(line)[0])
                precision = float(str.split(line)[1])
                x = float(str.split(line)[2])
                y = float(str.split(line)[3])
                w = float(str.split(line)[4])
                h = float(str.split(line)[5])
                #print("reading:%d %f %f %f %f %f" % (reading, precision, x, y, w, h))
                x_out = (x + w / 2) / img_width 
                y_out = (y + h / 2) / img_height
                w_out = w / img_width
                h_out = h / img_height
                out = str(reading) + " " + str(x_out) + " " + str(y_out) \
                    + " " + str(w_out) + " " + str(h_out) + " " + str(precision) + "\r\n"
                #print(out)
                outfile.write(out)
            os.remove(root + filename)
            os.rename(root + "a.txt", root + filename)

