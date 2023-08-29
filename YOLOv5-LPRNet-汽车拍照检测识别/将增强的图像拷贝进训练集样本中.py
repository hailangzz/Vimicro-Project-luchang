import os
import random

import shutil
from shutil import copy2
from tqdm import tqdm


origin_sample_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_images_strengthen"
trainfiles = os.listdir(origin_sample_path)  #（图片文件夹）
num_train = len(trainfiles[:])
print("num_train: " + str(num_train) )
index_list = list(range(num_train))
# print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0
trainDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_images\train\images"   #（将图片文件夹中的6份放在这个文件夹下）

for i in tqdm(index_list):
    fileName = os.path.join(origin_sample_path, trainfiles[i])  #（图片文件夹）+图片名=图片地址
    copy2(fileName, trainDir)
    num += 1

