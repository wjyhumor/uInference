# inverse the color of the inputfile images
import os
import cv2 as cv
import numpy as np
import matplotlib
import re
import PIL.ImageOps    
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

outdir = "/home/neusoft/amy/AT-201/data/water_electric_mechanical_190516_0625_train_inverse/"
inputfile = "/home/neusoft/amy/AT-201/data/water_electric_mechanical_190516_0625_train.list.train"

# create folder if not exists
if not os.path.exists(outdir):
    os.makedirs(outdir)
for i in range(10):
    if not os.path.exists(outdir + str(i)):
        os.makedirs(outdir + str(i))

# do 
train_images = []
train_labels = []
f = open(inputfile, 'r')
index = 0
while 1:
    for k in range(1):
        line = f.readline()
    if not line:
        break
    if re.search('/\d/', line)== None:
        continue
    img = Image.open(str(line).strip('\n'))
    #img.show()
    img = PIL.ImageOps.invert(img)
    name = os.path.basename(line)
    m = int(re.search('/\d/', line).group(0).strip('/'))
    savename = outdir + str(m) + '/inv_' + str(index) + '_' + name.rstrip()
    print(savename)
    img.save(savename, "JPEG")
    index += 1
