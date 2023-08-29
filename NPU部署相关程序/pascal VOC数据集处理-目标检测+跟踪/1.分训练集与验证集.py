# -*- coding: UTF-8 -*-
'''
@author: gu
@contact: 1065504814@qq.com
@time: 2021/3/4 上午11:52
@file: generate_txt.py
@desc: reference https://blog.csdn.net/qqyouhappy/article/details/110451619
'''

import os
import random

trainval_percent = 1
train_percent = 0.9
xmlfilepath = 'F:\AiTotalDatabase\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012/Annotations'
txtsavepath = 'F:\AiTotalDatabase\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012/ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('F:\AiTotalDatabase\VOC2012\VOC_dataset_YOLO/trainval.txt', 'w')
ftest = open('F:\AiTotalDatabase\VOC2012\VOC_dataset_YOLO/test.txt', 'w')
ftrain = open('F:\AiTotalDatabase\VOC2012\VOC_dataset_YOLO/train.txt', 'w')
fval = open('F:\AiTotalDatabase\VOC2012\VOC_dataset_YOLO/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    print(name)
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()