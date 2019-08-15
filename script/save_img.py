# save image as binary
import numpy as np
from PIL import Image
import struct

image_name = '../example.jpg'
output = '../example.img'

resize_w = 16
resize_h = 16

img = Image.open(image_name)
print(image_name)
img = np.array(img.resize((resize_w, resize_h), Image.ANTIALIAS))
print img[1, 0], img[0, 1]
print img.shape


with open(output, 'wb') as fout:
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            fout.write(struct.pack('>B', img[i, j]))
