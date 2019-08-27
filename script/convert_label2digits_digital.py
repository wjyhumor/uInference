# convert the dataset label to digits for digital meter
import os
import re
import numpy as np
import cv2 as cv
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

dataset = "/home/neusoft/amy/AT-201/data/electric_digital_190402/gray/"
outdir = "/home/neusoft/amy/AT-201/data/electric_digital_190402/digits/"
labeltxt = "/home/neusoft/amy/AT-201/data/electric_digital_190402/label"
listtxt = "/home/neusoft/amy/AT-201/data/electric_digital_190402/list"

# create folder if not exists
if not os.path.exists(outdir):
    os.makedirs(outdir)
for i in range(10):
    if not os.path.exists(outdir + str(i)):
        os.makedirs(outdir + str(i))

f_label = open(labeltxt)
flabel = []
for f in f_label:
    flabel.append(int(f.rstrip('\r\n')))
f_list = open(listtxt)
flist = []
for f in f_list:
    flist.append(f.rstrip('\r\n'))

index = 0
for f in flist:
    filename, file_extension = os.path.splitext(os.path.basename(f))
    #print(filename)
    img = cv.imread(f, 0)
    w, h = img.shape
    #print img.shape
    img = img[95:170, 50:290]
    figure = []
    for i in range(6):
        figure.append(img[:,i*40:(i+1)*40])
        #plt.imshow(figure[i], 'gray')
        #plt.show()
    label = flabel[index]
    labellist = []
    for i in range(6):
        labellist.append(label/pow(10,5-i))
        label = label % pow(10, 5-i)
    for i in range(6):
        savepath = outdir + str(labellist[i]) + '/' + filename + '_' + str(i) + '.jpg'
        print savepath
        cv.imwrite(savepath, figure[i])
    index += 1

