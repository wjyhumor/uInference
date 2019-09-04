#python3
import glob
import os
from sklearn.model_selection import train_test_split

#data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
output_train = '/home/neusoft/amy/uInference/data/water_elec_0516_0625.digits.train'
output_vaild = '/home/neusoft/amy/uInference/data/water_elec_0516_0625.digits.test'
output_all = '/home/neusoft/amy/uInference/data/water_elec_0516_0625.digits.all'
#wd = os.getcwd()
list_train = open(output_train, 'w')
list_test = open(output_vaild, 'w')
list_all = open(output_all, 'w')
txt_name_list = []

for filename in glob.iglob('/home/neusoft/amy/uInference/data/electric_mechanical_190516/digits/**/*.jpg', recursive=True):
    txt_name_list.append(filename)
for filename in glob.iglob('/home/neusoft/amy/uInference/data/water_mechanical_190516/digits/**/*.jpg', recursive=True):
    txt_name_list.append(filename)
for filename in glob.iglob('/home/neusoft/amy/uInference/data/electric_mechanical_190625/digits/**/*.jpg', recursive=True):
    txt_name_list.append(filename)
for filename in glob.iglob('/home/neusoft/amy/uInference/data/water_mechanical_190625/digits/**/*.jpg', recursive=True):
    txt_name_list.append(filename)

x_train, x_test = train_test_split(txt_name_list, test_size=0.2)

for txt_name in x_train:
    list_train.write('%s\n' %txt_name)
    list_all.write('%s\n' %txt_name)

for txt_name in x_test:
    list_test.write('%s\n' %txt_name)
    list_all.write('%s\n' %txt_name)

list_train.close()
list_test.close()
list_all.close()
