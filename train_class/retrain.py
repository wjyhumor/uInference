import argparse
import os
import re
import random
import subprocess
import numpy as np
import matplotlib
from PIL import Image, ImageEnhance

import models
import load_data
import save_model_builder
from train import train
from test import test

argparser = argparse.ArgumentParser(description='Retrain OCR model')

argparser.add_argument(
    '-type',
    type=int,
    help='type of model to retrain, 1: for edge unit, 3: for cloud')

argparser.add_argument(
    '-pretrain',
    type=int,
    help='use pretrained model flag, 1: use pretrained model, 0: not use')

argparser.add_argument(
    '-pretrain_model',
    help='path for pretrained model(option)')

argparser.add_argument(
    '-original_images_list',
    help='path for the list of original image data')

argparser.add_argument(
    '-new_images',
    help='path for the new image data')

argparser.add_argument(
    '-save_model',
    help='name for model to save')

argparser.add_argument(
    '-save_builder',
    help='name for model builder to save')


def WriteList(input, output):
    subprocess.call("rm -rf " + output, shell=True)
    f_out = open(output, 'w')

    if os.path.isdir(input):  
        g = os.walk(input)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name != '.DS_Store':
                    f_out.write(os.path.join(path, file_name))
                    f_out.write('\n')        
    elif os.path.isfile(input):  
        f_in = open(input, 'r')
        while 1:
            line = f_in.readline().strip()
            if not line:
                break
            g = os.walk(line)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name != '.DS_Store':
                        f_out.write(os.path.join(path, file_name))
                        f_out.write('\n')  
        f_in.close()
    f_out.close()



def ShuffleAllList(datalist):
    vaild_list = []
    data_number = 0
    f = open(datalist, 'r')
    while 1:
        line = f.readline()
        if not line:
            break
        vaild_list.append(line)
        data_number += 1
    f.close()
    random.shuffle(vaild_list)
    subprocess.call("rm -rf " + datalist, shell=True)
    f = open(datalist, 'a')
    for item in vaild_list:
        f.write(item)
    f.close()
    return data_number

# Augmentation


def Augmentation(datalist, save_name):
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
        # save original img
        img.save(name + '.jpg', "JPEG")

        # bright
        para = [0.3, 0.5, 2]
        for b in para:
            bright = ImageEnhance.Brightness(img)
            bright = bright.enhance(b)
            # contrast.show()
            savename = name + '_bri_' + str(index) + '_' + str(b) + '.jpg'
            bright.save(savename, "JPEG")
        index += 1

        # invert
        img = PIL.ImageOps.invert(img)
        savename = name + '_inv_' + str(index) + '_' + str(b) + '.jpg'
        img.save(savename, "JPEG")
        index += 1

        # contrast
        para = [0.3, 2]
        for b in para:
            bright = ImageEnhance.Contrast(img)
            bright = bright.enhance(b)
            # contrast.show()
            savename = name + '_cont_' + str(index) + '_' + str(b) + '.jpg'
            # print(savename)
            bright.save(savename, "JPEG")
        index += 1

        # sharpness
        para = [0, 2]
        for b in para:
            bright = ImageEnhance.Sharpness(img)
            bright = bright.enhance(b)
            savename = name + '_sharp_' + str(index) + '_' + str(b) + '.jpg'
            bright.save(savename, "JPEG")
        index += 1

    WriteList(aug_path, save_name)

# new_data_number:0-load all the data, other-load 9xdata


def merge_new_old(new_data_number, new_list, original_list, merged_list):
    original_list_number = new_data_number * 9
    datalist = []
    # read new data
    f = open(new_list, 'r')
    while 1:
        line = f.readline()
        if not line:
            break
        datalist.append(line)
    f.close()
    # read original data
    f = open(original_list, 'r')
    if original_list_number == 0:
        while 1:
            line = f.readline()
            if not line:
                break
            datalist.append(line)
    else:
        for i in range(original_list_number):
            line = f.readline()
            if not line:
                break
            datalist.append(line)
    f.close()
    # shuffle
    random.shuffle(datalist)
    # save to merged_list
    subprocess.call("rm -rf " + merged_list, shell=True)
    f = open(merged_list, 'a')
    for item in datalist:
        f.write(item)
    f.close()


def _main_(args):
    model_type = args.type
    pretrain_flag = args.pretrain
    original_images_list = args.original_images_list
    new_images = args.new_images
    save_model_name = args.save_model
    batch_size = 16
    epochs = 50
    
    print("===========Parameters==============")
    print("model_type: " + str(model_type))
    print("pretrain_flag: " + str(pretrain_flag)) 
    print("original_images_list: " + original_images_list)
    print("new_images: " + new_images)
    print("save_model_name: " + save_model_name)
    print("===================================")

    new_images_list = "./tmp/new_images.list"
    train_list = './tmp/retrain.list'
    test_list = './tmp/retrain.list'
    WriteList(new_images, new_images_list)
    new_images_number = ShuffleAllList(new_images_list)
    print("new_images_number:" + str(new_images_number))
    #Augmentation(new_images_list)
    if model_type == 1:
        merge_new_old(new_images_number, new_images_list, original_images_list, train_list)
    elif model_type == 3:
        merge_new_old(0, new_images_list, original_images_list, train_list)
    
    # train
    if pretrain_flag:
        pretrained_model_name = args.pretrain_model
        train(model_type, train_list, test_list, pretrain_flag, 
            batch_size, epochs, save_model_name, pretrained_model_name,
            reload_train=True, reload_test=False)
    else:
        train(model_type, train_list, test_list, pretrain_flag, 
            batch_size, epochs, save_model_name,
            reload_train=False, reload_test=False)
    
    # save model builder 
    if model_type == 3:
        save_model_builder_name = args.save_builder
        save_model_builder.save_model_builder(
            save_model_name, save_model_builder_name)
    
    # test
    #retrained_model_name = "weights-50.hdf5"
    #test.test(new_images_list, retrained_model_name)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)