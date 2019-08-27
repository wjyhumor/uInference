# convert color image to gray and save in 3 channels
import os
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

path = "/home/neusoft/amy/AT-201/data/electric_digital_190402/gray/"

files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    image = cv.imread(f,0)
    plt.imsave(f,image, cmap='gray')
    print(f)

#plt.imshow(binary_sauvola, cmap='gray')
#plt.show() 