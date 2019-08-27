# convert gray image to binary and save
import os
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

path = "/home/neusoft/amy/AT-201/data/electric_mechanical_190516/electric_mechanical_190516_train_binary/0/"

files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    image = cv.imread(f,0)
    thresh_sauvola = threshold_sauvola(image, window_size=41, k=-0.3)
    binary_sauvola = image > thresh_sauvola
    plt.imsave(f,binary_sauvola, cmap='gray')
    print(f)

#plt.imshow(binary_sauvola, cmap='gray')
#plt.show() 