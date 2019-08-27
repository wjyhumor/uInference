
import os
import re
import numpy as np
from PIL import Image

reload_train = True
reload_test = True#False#

resize_w = 16  # 28
resize_h = 16  # 40
mean = 122.81543917085412
std = 77.03797602437342
save_path = "./tmp/"

"""
# get mnist data
mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255, test_images / 255
# get fashion_mnist data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
"""

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def load_data_train(file_train):
    load_train_data_flag = 0
    # check if data already saved in .npy format
    if os.path.exists(save_path+os.path.basename(file_train)+"_image.npy"):
        train_images = np.load(
            save_path+os.path.basename(file_train)+"_image.npy")
        load_train_data_flag = 1
    if os.path.exists(save_path+os.path.basename(file_train)+"_label.npy"):
        train_labels = np.load(
            save_path+os.path.basename(file_train)+"_label.npy")
        load_train_data_flag = 1
    # reload by force
    if reload_train:
        load_train_data_flag = 0

    if load_train_data_flag == 0:
        train_images = []
        train_labels = []
        f = open(file_train, 'r')
        i = 0
        while 1:
            i += 1
            for k in range(1):
                line = f.readline()
            if not line:
                break
            if re.search('/\d/', line) == None:
                continue
            img = Image.open(str(line).strip('\n'))
            img = np.array(img.resize((resize_w, resize_h), Image.ANTIALIAS))
            if len(img.shape) == 3:
                img = rgb2gray(img)
            train_images.append(np.array(img))
            m = int(re.search('/\d/', line).group(0).strip('/'))
            train_labels.append(m)
            print("Read train image:" + str(i))
        # normalization----------------------------------------------
        train_images = np.asarray(train_images, dtype=np.float)
        train_labels = np.asarray(train_labels, dtype=np.int)
        train_images = (train_images - mean) / std
        # Save loaded data to .npy to save loading time la prochaine fois
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path+os.path.basename(file_train) +
                "_image.npy", train_images)
        np.save(save_path+os.path.basename(file_train) +
                "_label.npy", train_labels)
        f.close()
    elif load_train_data_flag == 1:
        train_images = np.asarray(train_images, dtype=np.float)
        train_labels = np.asarray(train_labels, dtype=np.int)
    return train_images, train_labels


def load_data_test(file_test):
    load_test_data_flag = 0
    if os.path.exists(save_path+os.path.basename(file_test)+"_image.npy"):
        test_images = np.load(
            save_path+os.path.basename(file_test)+"_image.npy")
        load_test_data_flag = 1
    if os.path.exists(save_path+os.path.basename(file_test)+"_label.npy"):
        test_labels = np.load(
            save_path+os.path.basename(file_test)+"_label.npy")
        load_test_data_flag = 1
    
    # reload by force
    if reload_test:
        load_test_data_flag = 0

    if load_test_data_flag == 0:
        test_images = []
        test_labels = []
        f = open(file_test, 'r')
        i = 0
        while 1:
            i += 1
            for k in range(1):
                line = f.readline()
            if not line:
                break
            if re.search('/\d/', line) == None:
                continue
            img = Image.open(str(line).strip('\n'))
            img = np.array(img.resize((resize_w, resize_h), Image.ANTIALIAS))
            if len(img.shape) == 3:
                img = rgb2gray(img)
            test_images.append(np.array(img))
            m = int(re.search('/\d/', line).group(0).strip('/'))
            test_labels.append(m)
            print("Read test image:" + str(i))
        test_images = np.asarray(test_images, dtype=np.float)
        test_labels = np.asarray(test_labels, dtype=np.int)
        test_images = (test_images - mean) / std
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path+os.path.basename(file_test) +
                "_image.npy", test_images)
        np.save(save_path+os.path.basename(file_test) +
                "_label.npy", test_labels)
        f.close()
    elif load_test_data_flag == 1:
        test_images = np.asarray(test_images, dtype=np.float)
        test_labels = np.asarray(test_labels, dtype=np.int)
    return test_images, test_labels
