
import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K

import random
import time
import math
import numpy as np
from PIL import Image


model_name = '../models/save_model.json'
weights_name = '../models/save_model.h5'
image_name = '../example.jpg'

resize_w = 16  # 28
resize_h = 16  # 40
mean = 122.81543917085412
std = 77.03797602437342

# get data
test_images = []
img = Image.open(image_name)
print(image_name)
img = np.array(img.resize((resize_w, resize_h), Image.ANTIALIAS))
#img = rgb2gray(img)
test_images.append(np.array(img))
test_images = np.asarray(test_images, dtype=np.float)
test_images = (test_images - mean) / std
test_images = test_images.reshape(-1, resize_w, resize_h, 1)

# load model
# model = load_model(str(model_name))
json_file = open(model_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(weights_name)

# run model
start = time.time()
predictions = model.predict(test_images)
end = time.time()

# show result
result = np.argmax(predictions[0])
print("pross list:" + str(predictions[0]) + "\nprossibility:" + str(
    predictions[0][result]) + "\nresult:" + str(result) + "\ntime:" + str(end-start))

"""
# batchnormalization
print "-----------------"
print "Test Batchnormalization:"
# 0-gamma, 1-beta, 2-mean, 3-variance;
para = [1.1175389, -0.36551213, 0.01766387, 0.61547184]
inout = [-1.5107958e-01, -0.60569]
print(inout[0] - para[2]) / math.sqrt(para[3] + 0.001) * para[0] + para[1]

# check maxpool2d
print "-----------------"
print "Test Maxpool2d:"
get_layer_output3 = K.function([model.layers[0].input],
                               [model.layers[2].output])
layer_output1 = get_layer_output3([test_images])[0]

get_layer_output4 = K.function([model.layers[0].input],
                               [model.layers[3].output])
layer_output2 = get_layer_output4([test_images])[0]
print np.shape(layer_output1)
print np.shape(layer_output2)
print layer_output1[0][0][0][7], layer_output1[0][0][1][7], layer_output1[0][1][0][7], layer_output1[0][1][1][7]
print layer_output2[0][0][0][7]

# dense layer
print "-----------------"
print "Test Dense:"
get_layer_output5 = K.function([model.layers[0].input],
                               [model.layers[8].output])
layer_output1 = get_layer_output5([test_images])[0]

get_layer_output6 = K.function([model.layers[0].input],
                               [model.layers[9].output])
layer_output2 = get_layer_output6([test_images])[0]
out = []
w = model.layers[9].get_weights()[0]
b = model.layers[9].get_weights()[1]
print w.shape, b.shape 
for j in range(0, 10):
    res = 0
    for i in range(0, 64):
        temp = w[i, j]*layer_output1[0, i]
        res += temp
    out.append(res+b[j])
print layer_output2
print out

# Conv2D
print "-----------------"
print "Test Conv2D:"
get_layer_output1 = K.function([model.layers[0].input],
                               [model.layers[3].output])
layer_output1 = get_layer_output1([test_images])[0]

get_layer_output2 = K.function([model.layers[0].input],
                               [model.layers[4].output])
layer_output2 = get_layer_output2([test_images])[0]

w = model.layers[4].get_weights()[0]
b = model.layers[4].get_weights()[1]

print "layer_output1:", np.shape(layer_output1)
print "layer_output2:", np.shape(layer_output2)
print "w:", w.shape, "b:", b.shape
res = 0
c = 13
# normal
for k in range(0, 8):
    for i in range(0, 5):
        for j in range(0, 5):
            temp = w[i, j, k, c]*layer_output1[0, i, j, k]
            res += temp
out = res + b[c]
print out
print layer_output2[0, 2, 2, c]
# edge paddings
res = 0
c = 13
for k in range(0, 8):
    for i in range(2, 5):
        for j in range(2, 5):
            temp = w[i, j, k, c]*layer_output1[0, i-2, j-2, k]
            res += temp
out = res + b[c]
print out
print layer_output2[0, 0, 0, c]
# find the location
min = 10000
for i in range(0, 8):
    for j in range(0, 8):
        for k in range(0, 16):
            temp = abs(out - layer_output2[0, i, j, k])
            if temp < min:
                min = temp
                K = k
                I = i
                J = j
print min, I, J, K
"""


#======================================
"""
print test_images
"""
# Conv2D - layer 0
print "-----------------"
print "Test Conv2D:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[0].output])
layer_output = get_layer_output([test_images])[0]

w = model.layers[0].get_weights()[0]
b = model.layers[0].get_weights()[1]

print "layer_output1:", np.shape(layer_output)
print "w:", w.shape, "b:", b.shape
"""
for k in range(layer_output.shape[3]):
    for j in range(layer_output.shape[2]):
        for i in range(layer_output.shape[1]):
            print(layer_output[0,j,i,k]),
        print('\n'),
"""
print layer_output[0,0,8,0]

# layer 1
print "-----------------"
print "Test BN:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[1].output])
layer_output = get_layer_output([test_images])[0]

w = model.layers[1].get_weights()[0]

print "layer_output1:", np.shape(layer_output)
print "w:", w.shape
print layer_output[0,5,4,2]

# layer 2
print "-----------------"
print "Test Activation:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[2].output])
layer_output = get_layer_output([test_images])[0]
        
print layer_output.shape
print layer_output[0,5,4,5]

# layer 3
print "-----------------"
print "Test maxpool:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[3].output])
layer_output = get_layer_output([test_images])[0]

#print layer_output
print layer_output[0,6,6,6]

# Conv2D - layer 4
print "-----------------"
print "Test Conv2D:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[4].output])
layer_output = get_layer_output([test_images])[0]

w = model.layers[4].get_weights()[0]
b = model.layers[4].get_weights()[1]

print "layer_output1:", np.shape(layer_output)
print "w:", w.shape, "b:", b.shape
print layer_output[0,5,4,2]

# layer 5
print "-----------------"
print "Test BN:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[5].output])
layer_output = get_layer_output([test_images])[0]

w = model.layers[5].get_weights()[0]

print "layer_output1:", np.shape(layer_output)
print "w:", w.shape
print layer_output[0,5,4,2]

# layer 6
print "-----------------"
print "Test Activation:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[6].output])
layer_output = get_layer_output([test_images])[0]

print layer_output[0,5,4,5]


# layer 7
print "-----------------"
print "Test maxpool:"
get_layer_output = K.function([model.layers[0].input],
                               [model.layers[7].output])
layer_output = get_layer_output([test_images])[0]

print layer_output[0,1,1,2]
print layer_output.shape
for k in range(layer_output.shape[3]):
    for j in range(layer_output.shape[2]):
        for i in range(layer_output.shape[1]):
            print(layer_output[0,j,i,k]),
        print('\n'),