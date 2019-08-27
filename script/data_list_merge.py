#python3
import glob
import os
from sklearn.model_selection import train_test_split

#data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
output_train = '/home/neusoft/amy/AT-201/data/water_elec_0516_0625.gray.train'
output_vaild = '/home/neusoft/amy/AT-201/data/water_elec_0516_0625.gray.test'
output_all = '/home/neusoft/amy/AT-201/data/water_elec_0516_0625.gray.all'
#wd = os.getcwd()
list_file1 = open(output_train, 'w')
list_file2 = open(output_vaild, 'w')
list_file_all = open(output_all, 'w')
txt_name_list = []

for filename in glob.iglob('/home/neusoft/amy/AT-201/data/electric_mechanical_190516/gray/**/*.jpg', recursive=True):
    txt_name_list.append(filename)
for filename in glob.iglob('/home/neusoft/amy/AT-201/data/water_mechanical_190516/gray/**/*.jpg', recursive=True):
    txt_name_list.append(filename)
for filename in glob.iglob('/home/neusoft/amy/AT-201/data/electric_mechanical_190625/gray/**/*.jpg', recursive=True):
    txt_name_list.append(filename)
for filename in glob.iglob('/home/neusoft/amy/AT-201/data/water_mechanical_190625/gray/**/*.jpg', recursive=True):
    txt_name_list.append(filename)

x_train, x_test = train_test_split(txt_name_list, test_size=0.2)

for txt_name in x_train:
    list_file1.write('%s\n' %txt_name)
    list_file_all.write('%s\n' %txt_name)

for txt_name in x_test:
    list_file2.write('%s\n' %txt_name)
    list_file_all.write('%s\n' %txt_name)

list_file1.close()
list_file2.close()
list_file_all.close()

"""
data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
output_train = '/Linux/data/electric_digital.list.train'
output_vaild = '/Linux/data/electric_digital.list.valid'

#wd = os.getcwd()
list_file1 = open(data_folder + output_train, 'w')
list_file2 = open(data_folder + output_vaild, 'w')
txt_name_list = []

for filename in glob.iglob(data_folder + '/Linux/data/electric_digital_190329_train/**/*.jpg', recursive=True):
    txt_name_list.append(filename)

for filename in glob.iglob(data_folder + '/Linux/data/electric_digital_190401_train/**/*.jpg', recursive=True):
    txt_name_list.append(filename)

for filename in glob.iglob(data_folder + '/Linux/data/electric_digital_190402_train/**/*.jpg', recursive=True):
    txt_name_list.append(filename)

#print(txt_name_list)
x_train, x_test = train_test_split(txt_name_list, test_size=0.2)

for txt_name in x_train:
    list_file1.write('%s\n' %txt_name)

for txt_name in x_test:
    list_file2.write('%s\n' %txt_name)

list_file1.close()
list_file2.close()
"""