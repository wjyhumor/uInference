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
from save_model_builder import save_model_builder
from list_oper import WriteList
from list_oper import ShuffleAllList
from list_oper import merge_new_old
from train import train
from test import test

argparser = argparse.ArgumentParser(description='Retrain OCR model')

argparser.add_argument(
    '-type',
    type=int,
    help='type of model to retrain, 1: for edge unit, 3: for cloud')

argparser.add_argument(
    '-batch_size',
    type=int,
    help='batch size to train')

argparser.add_argument(
    '-epochs',
    type=int,
    help='epochs to train')

argparser.add_argument(
    '-pretrain',
    type=int,
    help='use pretrained model flag, 1: use pretrained model, 0: not use')

argparser.add_argument(
    '-pretrain_model',
    help='path for pretrained model(option)')

argparser.add_argument(
    '-reload_train',
    type=int,
    help='flag for reload training data or not, 1: reload, 0: do not reload')

argparser.add_argument(
    '-reload_test',
    type=int,
    help='flag for reload test data or not, 1: reload, 0: do not reload')

argparser.add_argument(
    '-origin_images',
    help='path for the list of original image data')

argparser.add_argument(
    '-new_images',
    help='path for the new image data')

argparser.add_argument(
    '-test',
    help='path for the test image data')

argparser.add_argument(
    '-save_model',
    help='name for model to save')


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


def _main_(args):
    model_type = args.type
    pretrain_flag = args.pretrain
    origin_images = args.origin_images
    new_images = args.new_images
    save_model_name = args.save_model
    batch_size = args.batch_size
    epochs = args.epochs
    
    print("===========Parameters==============")
    print("model_type: " + str(model_type))
    print("pretrain_flag: " + str(pretrain_flag)) 
    print("origin_images: " + origin_images)
    print("new_images: " + new_images)
    print("save_model_name: " + save_model_name)
    print("batch_size: " + str(batch_size))
    print("epochs: " + str(epochs))
    print("===================================")

    # retrain data
    save_path = "./tmp/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    new_images_list = "./tmp/new_images.list"
    train_list = './tmp/retrain.list'
    WriteList(new_images, new_images_list)
    new_images_number = ShuffleAllList(new_images_list)
    print("new_images_number:" + str(new_images_number))
    #Augmentation(new_images_list)
    if model_type == 1:
        merge_new_old(new_images_number, new_images_list, origin_images, train_list)
    elif model_type == 3:
        merge_new_old(0, new_images_list, origin_images, train_list)
    
    # test data
    if args.test is not None:
        test_list = "./tmp/test.list"
        WriteList(arg.test, test_list)
    else:
        test_list = new_images_list

    # train
    if pretrain_flag:
        pretrained_model_name = args.pretrain_model
        train(model_type, train_list, test_list, pretrain_flag, 
            batch_size, epochs, save_model_name, pretrained_model_name,
            reload_train=args.reload_train, reload_test=args.reload_test)
    else:
        train(model_type, train_list, test_list, pretrain_flag, 
            batch_size, epochs, save_model_name,
            reload_train=args.reload_train, reload_test=args.reload_test)
    
    # save model builder 
    if model_type == 3:
        save_model_builder(save_model_name, os.path.splitext(save_model_name)[0])
    
    # test
    test(test_list, save_model_name, reload_test=args.reload_test)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

