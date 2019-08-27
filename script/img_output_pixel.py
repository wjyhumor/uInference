
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# get data
"""
image_path = "../Linux/data/11_3.jpg"
img = rgb2gray(mpimg.imread(image_path))
"""
file_train = "../data/history/electric_mechanical_190411_1.list.test"
f = open(file_train, 'r')
line = f.readline()
img = rgb2gray(mpimg.imread(str(line).strip('\n')))

for i in range(len(np.array(img))):
    for j in range(len(np.array(img)[i])):
        print(np.array(img)[i][j], end=",")
print(" ")

m = int(re.search('/\d/', line).group(0).strip('/'))
print(str(m))
