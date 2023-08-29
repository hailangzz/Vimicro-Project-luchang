import numpy as np
import os
from shutil import copy2

total_images_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\train\images'
total_labels_images_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\train\labels'

origin_sample_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_buff\train\images"
origin_labels_sample_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_buff\train\labels"


total_images_list = os.listdir(total_images_path)
trainfiles = os.listdir(origin_sample_path)  #（图片文件夹）

for single_image in trainfiles:

    # fileName = os.path.join(origin_sample_path, single_image)  # （图片文件夹）+图片名=图片地址
    # copy2(fileName, total_images_path)

    # 拷贝标注数据过去
    labelName = os.path.join(origin_labels_sample_path, single_image.replace('.jpg','.txt'))
    # print(labelName)
    copy2(labelName, total_labels_images_path)