
import os
import random
import subprocess
#import cv2 as cv
import numpy as np
import matplotlib
import re
import PIL.ImageOps    
from PIL import Image, ImageEnhance

import train
import test

def WriteList(inputfolder, output):
    g = os.walk(inputfolder)
    subprocess.call("rm -rf " + output, shell=True)
    f = open(output,'w')
    for path,dir_list,file_list in g:
        for file_name in file_list:
                if file_name != '.DS_Store':
                    f.write(os.path.join(path, file_name))
                    f.write('\n')

def ShuffleAllList(datalist):
    vaild_list = []
    data_number = 0
    f = open(datalist,'r')
    while 1:
        line = f.readline()
        if not line:
            break
        vaild_list.append(line)
        data_number += 1
    f.close()
    #print("Data in all: ",data_number)
    random.shuffle(vaild_list)
    subprocess.call("rm -rf " + datalist, shell=True)
    f = open(datalist,'a')
    for item in vaild_list:
        f.write(item)
    f.close()
    return data_number

# Augmentation
def Augmentation(datalist):
    f = open(datalist, 'r')
    flag = 0
    while 1:
        for k in range(1):
            line = f.readline()
        if not line:
            break
        index = 0
        img = Image.open(str(line).strip('\n'))
        basename = os.path.basename(str(line).strip('.jpg\n'))
        aug_path = os.path.dirname(os.path.dirname(line)) + "_aug"
        aug_sub_path = os.path.basename(os.path.dirname(line))
        #print(basename)
        #print(aug_path)
        #print(aug_sub_path)
        # create folder if not exists
        if flag == 0:
            if os.path.exists(aug_path):
                subprocess.call("rm -r " + aug_path, shell=True)
            if not os.path.exists(aug_path):
                os.makedirs(aug_path)
            for i in range(10):
                if not os.path.exists(aug_path + '/' + str(i)):
                    os.makedirs(aug_path + '/' + str(i))
            flag = 1

        name = aug_path + '/' + aug_sub_path + '/' + basename
        #print(name)
        # save original img
        img.save(name + '.jpg', "JPEG")
        #bright
        para = [0.3, 0.5, 2] 
        for b in para:
            bright = ImageEnhance.Brightness(img)
            bright = bright.enhance(b)
            #contrast.show()
            savename = name + '_bri_' + str(index) + '_' + str(b) + '.jpg'
            bright.save(savename, "JPEG")
        index += 1
        # invert
        img = PIL.ImageOps.invert(img)
        savename = name + '_inv_' + str(index) + '_' + str(b) + '.jpg'
        #print(savename)
        img.save(savename, "JPEG")
        index += 1
        #contrast
        para = [0.3, 2] 
        for b in para:
            bright = ImageEnhance.Contrast(img)
            bright = bright.enhance(b)
            #contrast.show()
            savename = name + '_cont_' + str(index) + '_' + str(b) + '.jpg'
            #print(savename)
            bright.save(savename, "JPEG")
        index += 1
        #sharpness
        para = [0, 2] 
        for b in para:
            bright = ImageEnhance.Sharpness(img)
            bright = bright.enhance(b)
            #contrast.show()
            savename = name + '_sharp_' + str(index) + '_' + str(b) + '.jpg'
            #print(savename)
            bright.save(savename, "JPEG")
        index += 1
    WriteList(aug_path, "retrain_newdata_aug.txt")

# newdata_number:0-load all the data, other-load 9xdata
def merge_new_old(newdata_number, new_data, original_data, merged_data):
    original_data_number = newdata_number * 9
    datalist = []
    # read new data
    f = open(new_data,'r')
    while 1:
        line = f.readline()
        if not line:
            break
        datalist.append(line)
    f.close()
    # read original data
    f = open(original_data,'r')
    if original_data_number == 0:
        while 1:
            line = f.readline()
            if not line:
                break
            datalist.append(line)
    else:
        for i in range(original_data_number):
            line = f.readline()
            if not line:
                break
            datalist.append(line)
    f.close()
    # shuffle
    random.shuffle(datalist)
    # write into new_training.txt
    subprocess.call("rm -rf " + merged_data, shell=True)
    f = open(merged_data,'a')
    for item in datalist:
        f.write(item)
    f.close()    

def _main():
    inputfolder = "/home/neusoft/amy/AT-201/data/beilu_0819/"
    originallist = "retrain_merge.txt"#"/home/neusoft/amy/AT-201/data/water_elec_0516_0625.digits.train"
    newlist = "merge_beilu.txt"
    WriteList(inputfolder, newlist)
    newdata_number = ShuffleAllList(newlist)
    file_train = 'retrain_merge.txt'
    merge_new_old(0, newlist, originallist, file_train)
  
    # train
    load_model_flag = False 
    load_model_name = ''
    batch_size = 16
    epochs = 50
    file_test = newlist
    train.train(file_train, file_test, load_model_flag, load_model_name, batch_size, epochs)
    
    # test
    #retrained_model_name = "weights-50.hdf5"
    #test.test(newlist, retrained_model_name)


if __name__ == '__main__':
    _main()