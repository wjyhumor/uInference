
import tensorflow as tf
import keras
from keras.models import load_model

import random
import time
import numpy as np
from PIL import Image
import load_data

model_name = 'weights-30.hdf5'
image_name = '/Users/jinyan/Others/codes/AT-201/data/new/9/5.jpg'

# get data
test_images = []
img = Image.open(image_name)
print(image_name)
img = np.array(img.resize(
    (load_data.resize_w, load_data.resize_h), Image.ANTIALIAS))
#img = rgb2gray(img)
test_images.append(np.array(img))
test_images = np.asarray(test_images, dtype=np.float)
test_images = (test_images - load_data.mean) / load_data.std
test_images = test_images.reshape(-1,
                                  load_data.resize_w, load_data.resize_h, 1)

# load model
model = load_model(str(model_name))

# run model
start = time.time()
predictions = model.predict(test_images)
end = time.time()

# show result
result = np.argmax(predictions[0])
print("pross list:" + str(predictions[0]) + "\nprossibility:" + str(
    predictions[0][result]) + "\nresult:" + str(result) + "\ntime:" + str(end-start))
