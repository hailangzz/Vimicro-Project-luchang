# 注由于在文字识别流程中，存在样本不均衡的情况，因此要提前除皖之外的其他城市车牌到识别样本内；

import os
import os
import random

import shutil
from shutil import copy2

origin_sample_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD2019\ccpd_base"
trainfiles = os.listdir(origin_sample_path)  #（图片文件夹）
print(trainfiles)

trainDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\train\images_add"   #（将图片文件夹中的6份放在这个文件夹下）
validDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\val\images"     #（将图片文件夹中的2份放在这个文件夹下）
detectDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\test\images"   #（将图片文件夹中的2份放在这个文件夹下）


for image_name in trainfiles:
    # print(image_name.split('-')[4])
    if image_name.split('-')[4].split('_')[0]!='0':
        fileName = os.path.join(origin_sample_path, image_name) # （图片文件夹）+图片名=图片地址
        copy2(fileName, trainDir)
# for i in index_list:
#     fileName = os.path.join(origin_sample_path, trainfiles[i])  #（图片文件夹）+图片名=图片地址
#     if num < num_train*0.9:  # 7:1:2
#         # print(str(fileName))
#         copy2(fileName, trainDir)
#     elif num < num_train*0.95:
#         # print(str(fileName))
#         copy2(fileName, validDir)
#     else:
#         # print(str(fileName))
#         copy2(fileName, detectDir)
#     num += 1