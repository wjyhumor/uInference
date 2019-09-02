# save image as binary
import numpy as np
from PIL import Image
import struct

image_name = '../ex_od.jpg'
output = '../ex_od.img'
resize_w = 320 
resize_h = 105 
"""
image_name = '../ex_class.jpg'
output = '../ex_class.img'
resize_w = 16
resize_h = 16
"""
#keras: (height, width, channel) - (105, 320, 1)
img = Image.open(image_name)
img = np.array(img.resize((resize_w, resize_h), Image.ANTIALIAS))
print img.shape, img.shape[0], img.shape[1]

# first save width, then height
with open(output, 'wb') as fout:
    for i in range(img.shape[0]): # height
        for j in range(img.shape[1]): # width
            fout.write(struct.pack('>B', img[i, j]))
