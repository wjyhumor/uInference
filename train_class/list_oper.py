import os
import random
import subprocess
from sklearn.model_selection import train_test_split


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


def SplitList(input_all, output_train, output_test, ratio):
    vaild_list = []
    data_number = 0

    f = open(input_all, 'r')
    while 1:
        line = f.readline()
        if not line:
            break
        vaild_list.append(line)
        data_number += 1
    f.close()
    print("All Set: ", data_number)

    random.shuffle(vaild_list)
    train_list = vaild_list[0:int(ratio*data_number)]
    print("Train Set: ", int(ratio*data_number))
    test_list = vaild_list[int(ratio*data_number):]
    print("Test Set: ", data_number - int(ratio*data_number))

    subprocess.call("rm -rf " + output_train, shell=True)
    f = open(output_train, 'a')
    for item in train_list:
        f.write(item)
    f.close()

    subprocess.call("rm -rf " + output_test, shell=True)
    f = open(output_test, 'a')
    for item in test_list:
        f.write(item)
    f.close()

# new_data_number:0-load all the data, other-load 9 x #data
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
