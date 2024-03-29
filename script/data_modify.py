# used for correct the labeling errors in the datasets

import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import webbrowser

LINE = [118 , 214 , 278 , 715 , 978 , 1067 , 1255 , 1302 , 1314 , 1464 , 1575 , 1628 , 1749 , 1777 , 1881 , 1895 , 1896 , 2041 , 2043 , 2204 , 2239 , 2323 , 2432 , 2569 , 2751 , 2767 , 2892 , 3194 , 3287 , 3464 , 3923 , 3944 , 4249 , 4666 , 4701 , 4790 , 4905 , 5018 , 5040 , 5068 , 5109 , 5145 , 5525 , 5574 , 5808 , 5872 , 5944 , 6100 , 6216 , 6560 , 6577 , 6638 , 6700 , 6936 , 6968 , 8025 , 8710 , 8830 , 9422 , 9500 , 9690 , 9725 , 9796 , 9864 , 10235 , 10245 , 10353 , 10482 , 10538 , 10652 , 10818 , 10860 , 10936 , 11031 , 11126 , 11143 , 11200 , 11217 , 11358 , 11456 , 11495 , 11525 , 11588 , 11636 , 11918 , 11939 , 11964 , 12044 , 12197 , 12240 , 12316 , 12609 , 12621 , 12670 , 12842 , 13050 , 13308 , 13619 , 13796 , 13950 , 14066 , 14636 , 14876 , 15342 , 15775 , 15909 , 15997 , 16284 , 16617 , 16665 , 16719 , 16983 , 17035 , 17156 , 17274 , 17421 , 17677 , 17808 , 17847 , 17874 , 17963 , 18086 , 18225 , 18241 , 18754 , 18835 , 19533 , 19626 , 19679 , 19916 , 19961 , 20187 , 20469 , 20493 , 20807 , 20982 , 21334 , 21478 , 21494 , 21598 , 21696 , 21726 , 21960 , 22054 , 22097 , 22123 , 22297 , 22505 , 22665 , 22685 , 22967 , 23016 , 23159 , 23182 , 23249 , 23619 , 23714 , 24032 , 24119 , 24275 , 24369 , 24399 , 24434 , 24565 , 24751 , 25031 , 25060 , 25098 , 25100 , 25131 , 25176 , 25297 , 25454 , 25776 , 26031 , 26106 , 26194 , 26209 , 26252 , 26303 , 26329 , 26331 , 26392 , 26419 , 26974 , 27125 , 27312 , 27420 , 27736 , 27767 , 27811 , 28465 , 28608 , 28657 , 28767 , 28793 , 28892 , 28925 , 28959 , 29097 , 29220 , 29315 , 29704 , 29842 , 29878 , 29891 , 30003 , 30088 , 30356 , 30441 , 30546 , 30783 , 31338 , 31401 , 31457 , 31508 , 31724 , 31814 , 31852 , 32314 , 32528 , 32670 , 32787 , 33247 , 33617 , 33623 , 34083 , 34342 , 34543 , 34686 , 34797 , 35097 , 35139 , 35208 , 35299 , 35551 , 35596 , 35615 , 35777 , 36112 , 36294 , 36512 , 36717]
file_test = '/home/neusoft/amy/AT-201/data/list/water_elec_0516_0625.train'

fp = open(file_test, 'r')
for i, line in enumerate(fp):
    for j in range(len(LINE)):
        if i == LINE[j]:
            name = str(line).strip('\n')
            print(j)
            print(name)
            replaced = re.sub('digits/\d/', 'gray/', name)
            replaced = re.sub('_\d.jpg','.jpg',replaced)
            #print(replaced)
            replaced_txt = re.sub('.jpg','.txt',replaced)
            img = Image.open(replaced)
            #img = np.asarray(img, dtype=np.float)
            webbrowser.open(replaced_txt)
            plt.imshow(img)
            plt.show()

fp.close()