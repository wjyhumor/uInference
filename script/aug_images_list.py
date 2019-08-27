# augmentation of the inputfile images
import os
import cv2 as cv
import numpy as np
import matplotlib
import re
import PIL.ImageOps    
from PIL import Image, ImageEnhance

indir = "/Users/jinyan/Documents/codes/AT-201/data/new.all"

f = open(indir, 'r')
i = 0
while 1:
    i += 1
    for k in range(1):
        line = f.readline()
    if not line:
        break
    
    index = 0
    img = Image.open(str(line).strip('\n'))
    name = str(line).strip('.jpg\n')
    #bright
    para = [0.3, 0.5, 2] 
    for b in para:
        bright = ImageEnhance.Brightness(img)
        bright = bright.enhance(b)
        #contrast.show()
        savename = name + 'bri_' + str(index) + '_' + str(b) + '.jpg'
        print(savename)
        bright.save(savename, "JPEG")
    index += 1

    # invert
    img = PIL.ImageOps.invert(img)
    savename = name + 'inv_' + str(index) + '_' + str(b) + '.jpg'
    print(savename)
    img.save(savename, "JPEG")
    index += 1

    #contrast
    para = [0.3, 2] 
    for b in para:
        bright = ImageEnhance.Contrast(img)
        bright = bright.enhance(b)
        #contrast.show()
        savename = name + 'cont_' + str(index) + '_' + str(b) + '.jpg'
        print(savename)
        bright.save(savename, "JPEG")
    index += 1

    #sharpness
    para = [0, 2] 
    for b in para:
        bright = ImageEnhance.Sharpness(img)
        bright = bright.enhance(b)
        #contrast.show()
        savename = name + 'sharp_' + str(index) + '_' + str(b) + '.jpg'
        print(savename)
        bright.save(savename, "JPEG")
    index += 1