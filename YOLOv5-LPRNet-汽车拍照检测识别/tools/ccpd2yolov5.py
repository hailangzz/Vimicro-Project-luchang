"""
@Author: HuKai
@Date: 2022/5/29  10:47
@github: https://github.com/HuKai97
"""
# 制作检测模型的COCO数据集

import shutil
import cv2
import os
import numpy as np

def txt_translate(path, txt_path):
    for filename in os.listdir(path):
        # print(filename)

        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割。 （3:代表只切割三个）
        # print('list1:', list1)
        subname = list1[2]  # 268&497_395&526 读取框体坐标信息
        # print('subname:',subname)
        list2 = filename.split(".", 1)
        subname1 = list2[1] # jpg 文件类型
        # print('subname1:',subname1)
        if subname1 == 'txt':
            continue

        # 以下为分解标注框信息：
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点

        # 注：不需要做以下校验，因为很费时间
        # img = cv2.imread(os.path.join(path + filename))
        # img = cv2.imdecode(np.fromfile(os.path.join(path + filename), dtype=np.uint8), -1)
        # if img is None:  # 自动删除失效图片（下载过程有的图片会存在无法读取的情况）
        #     print(filename,' : this picture is error!!!')
        #     os.remove(os.path.join(path, filename))
        #     continue
        image_shape=(1160, 720)
        width = width / image_shape[1]
        height = height / image_shape[0]
        cx = cx / image_shape[1]
        cy = cy / image_shape[0]
        #
        txtname = filename.split(".", 1)
        txtfile = txt_path + txtname[0] + ".txt"

        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))


if __name__ == '__main__':
    # det图片存储地址
    trainDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_buff\train\images\\"
    # validDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset_weather\val\images\\"
    # testDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset_weather\test\images\\"
    # det txt存储地址
    train_txt_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_buff\train\labels\\"
    # val_txt_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset_weather\val\labels\\"
    # test_txt_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset_weather\test\labels\\"
    txt_translate(trainDir, train_txt_path)
    # txt_translate(validDir, val_txt_path)
    # txt_translate(testDir, test_txt_path)