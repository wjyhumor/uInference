# convert darknet format:
# <object-class> <x> <y> <width> <height>
# <object-class> - integer number of object from 0 to (classes-1)
# <x> <y> <width> <height> - float values relative to width and height of image
# <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
# <x> <y> - are center of rectangle (are not top-left corner)
# to 
# keras_yolo3 format in ONE file:
# Row format: `image_file_path box1 box2 ... boxN`;  
# Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from PIL import Image

dataset_path = "/home/neusoft/amy/AT-201/data/electric_mechanical_190516/electric_mechanical_190516_gray/"
output_path = "electric_mechanical_190516_gray.txt"

g = os.walk(dataset_path)
for root, dir, files in g:
    for file in files:
        if '.jpg' in file:
            filename, file_extension = os.path.splitext(file)
            #filename += '.txt'
            filetxt = filename + '.txt'
            txt = open(root + filetxt)
            img = Image.open(root + file)
            w, h = img.size
            s = str(root+filename+'.jpg')
            
            while(1):
                line = txt.readline()
                #print(line)
                if len(line) == 0:
                    break
                location = []
                objectclass = str.split(line)[0]
                xmin = (int)(((float)(str.split(line)[1]) - (float)(str.split(line)[3])/2)*w)
                ymin = (int)(((float)(str.split(line)[2]) - (float)(str.split(line)[4])/2)*h)
                xmax = (int)(((float)(str.split(line)[1]) + (float)(str.split(line)[3])/2)*w)
                ymax = (int)(((float)(str.split(line)[2]) + (float)(str.split(line)[4])/2)*h)
                s += " " + ",".join([str(xmin), str(ymin), str(xmax), str(ymax), str(objectclass)])
                
            s += "\n"
            print(s)
            out = open(output_path, 'a+')
            out.write(s)
