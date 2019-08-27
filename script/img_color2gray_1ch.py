# convert color image to gray and save in 1 channel
import os

import numpy as np
from PIL import Image
path = "/Users/jinyan/Documents/codes/AT-201/data/electric_mechanical_190625/gray/"

files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))
"""
for f in files:
    im = Image.open(f)
    grey_image = im.convert('L')
    grey_image.save(f)
    print(f)
"""
# check how many channel for the images
for f in files:
    im = Image.open(f)
    im = np.array(im.resize((160, 160), Image.ANTIALIAS))
    print(im.shape)

#plt.imshow(binary_sauvola, cmap='gray')
#plt.show() 