# get the average size(w x h) of the data
import os
import cv2 as cv
import numpy as np
import matplotlib
import re
from PIL import Image

path = "/home/neusoft/amy/AT-201/data/water_electric_mechanical_190516_0625_train.list.train"

size_w = []
size_h = []
f = open(path, 'r')
while 1:
    line = f.readline()
    if not line:
        break
    if re.search('/\d/', line) == None:
        continue
    img = Image.open(str(line).strip('\n'))
    w, h = img.size
    print(img.size)
    size_w.append(w)
    size_h.append(h)

mean_w = np.mean(size_w)
mean_h = np.mean(size_h)
print(mean_w, mean_h) #28 x 40