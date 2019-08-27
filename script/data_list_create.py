"""
    generate the images' name list 
    AND
    seperate them in to train and test 
"""

import os
import random
import subprocess

"""
    inputsubfolder: the dataset annotated
    output_all: output all list name
    output_train: output train list name
    output_test: output test list name 
"""
inputfolder = "/home/neusoft/amy/AT-201/data/beilu_0820_blank"
output_all = "/home/neusoft/amy/AT-201/data/beilu_0820_blank.all"
#output_train = "/home/neusoft/amy/AT-201/data/list/electric_digital_190402.train"
#output_test = "/home/neusoft/amy/AT-201/data/list/electric_digital_190402.test"

data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(data_folder)

def Rename():
    g = os.walk(inputfolder)
    subprocess.call("rm -rf " + output_all, shell=True)
    f = open(output_all,'a')

    for path,dir_list,file_list in g:
        num = 0
        for file_name in file_list:
            print(os.path.join(path, file_name))
            subprocess.call("mv " + os.path.join(path, file_name).replace(' ','\ ').replace('(','\(').replace(')','\)') + " " + os.path.join(path, str(num)+".jpg"), shell=True)
            num = num + 1

def WriteList():
    g = os.walk(inputfolder)
    subprocess.call("rm -rf " + output_all, shell=True)
    f = open(output_all,'w')

    for path,dir_list,file_list in g:
        for file_name in file_list:
                f.write(os.path.join(path, file_name))
                f.write('\n')

def ShuffleAllList():
    vaild_list = []
    data_number = 0

    f = open(output_all,'r')
    while 1:
        line = f.readline()
        if not line:
            break
        vaild_list.append(line)
        data_number += 1
    f.close()
    print("Data in all: ",data_number)

    random.shuffle(vaild_list)

    subprocess.call("rm -rf " + output_all, shell=True)
    f = open(output_all,'a')
    for item in vaild_list:
        f.write(item)
    f.close()


def SplitList():
    vaild_list = []
    data_number = 0

    f = open(output_all,'r')
    while 1:
        line = f.readline()
        if not line:
            break
        vaild_list.append(line)
        data_number += 1
    f.close()
    print("Data in all: ",data_number)

    random.shuffle(vaild_list)
    train_list = vaild_list[0:int(0.8*data_number)]
    print("Train Set: ",int(0.8*data_number))
    test_list = vaild_list[int(0.8*data_number):]
    print("Test Set: ", data_number - int(0.8*data_number))

    subprocess.call("rm -rf " + output_train, shell=True)
    f = open(output_train,'a')
    for item in train_list:
        f.write(item)
    f.close()

    subprocess.call("rm -rf " + output_test, shell=True)
    f = open(output_test,'a')
    for item in test_list:
        f.write(item)
    f.close()

#Rename()
WriteList()
ShuffleAllList()
#SplitList()

