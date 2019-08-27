# merge the inputs to a output with random shuffle
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import numpy as np
import random

input1 = "/home/neusoft/amy/AT-201/data/list/new.all"
input2 = "/home/neusoft/amy/AT-201/data/list/train"
output = "/home/neusoft/amy/AT-201/data/list/train.txt"

txt1 = open(input1)
txt2 = open(input2)
s = []
while(1):
    line = txt1.readline()
    if len(line) == 0:
        break
    if os.path.isfile(str.split(line)[0]):
        s.append(line)
while(1):
    line = txt2.readline()
    if len(line) == 0:
        break
    if os.path.isfile(str.split(line)[0]):
        s.append(line)
random.shuffle(s)
out = open(output, 'a+')
for i in range(len(s)):
    out.write(s[i])
