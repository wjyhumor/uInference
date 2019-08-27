# inverse the color of the inputfile images
import os
import cv2 as cv
import numpy as np
import matplotlib
import re
import PIL.ImageOps    
from PIL import Image, ImageEnhance

indir = "/home/neusoft/amy/AT-201/data/new/8/"
name = "4.jpg"
f = open(indir+name, 'r')
index = 0
img = Image.open(indir+name)

#bright
para = [0.3, 0.5, 2] 
for b in para:
    bright = ImageEnhance.Brightness(img)
    bright = bright.enhance(b)
    #contrast.show()
    savename = indir + 'bri_' + str(index) + '_' + str(b) + '_' + name.rstrip()
    print(savename)
    bright.save(savename, "JPEG")
index += 1

# invert
img = PIL.ImageOps.invert(img)
savename = indir + 'inv_' + str(index) + '_' + str(b) + '_' + name.rstrip()
print(savename)
img.save(savename, "JPEG")
index += 1

#contrast
para = [0.3, 2] 
for b in para:
    bright = ImageEnhance.Contrast(img)
    bright = bright.enhance(b)
    #contrast.show()
    savename = indir + 'cont_' + str(index) + '_' + str(b) + '_' + name.rstrip()
    print(savename)
    bright.save(savename, "JPEG")
index += 1

#sharpness
para = [0, 2] 
for b in para:
    bright = ImageEnhance.Sharpness(img)
    bright = bright.enhance(b)
    #contrast.show()
    savename = indir + 'sharp_' + str(index) + '_' + str(b) + '_' + name.rstrip()
    print(savename)
    bright.save(savename, "JPEG")
index += 1