# remove the pngs in the path
import os
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

path = "/home/neusoft/amy/AT-201/data/electric_mechanical_190412_1/electric_mechanical_190412_1_result/"

files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            print(file)
            os.remove(path+file)


#plt.imshow(binary_sauvola, cmap='gray')
#plt.show() 