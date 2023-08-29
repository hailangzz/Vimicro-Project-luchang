"""
@Author: HuKai
@Date: 2022/5/29  10:44
@github: https://github.com/HuKai97
"""
# 将原始数据集划分为train、val、test等三部分数据集
import os
import random

import shutil
from shutil import copy2
origin_sample_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD2019\ccpd_weather"
trainfiles = os.listdir(origin_sample_path)  #（图片文件夹）
num_train = len(trainfiles[:40000])
print("num_train: " + str(num_train) )
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0
trainDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset_weather\train\images"   #（将图片文件夹中的6份放在这个文件夹下）
validDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset_weather\val\images"     #（将图片文件夹中的2份放在这个文件夹下）
detectDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset_weather\test\images"   #（将图片文件夹中的2份放在这个文件夹下）
for i in index_list:
    fileName = os.path.join(origin_sample_path, trainfiles[i])  #（图片文件夹）+图片名=图片地址
    if num < num_train*0.9:  # 7:1:2
        # print(str(fileName))
        copy2(fileName, trainDir)
    elif num < num_train*0.95:
        # print(str(fileName))
        copy2(fileName, validDir)
    else:
        # print(str(fileName))
        copy2(fileName, detectDir)
    num += 1


