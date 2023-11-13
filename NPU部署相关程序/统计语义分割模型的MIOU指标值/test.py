import os
from PIL import Image
import numpy as np


def bool_image_value(image_array,threshold_value = 125):
    # 图像矩阵元素以125作为区分点区分为0、1
    # 根据条件操作进行元素重定义
    image_array[threshold_value > 125] = True
    image_array[threshold_value <= 125] = False
    pass

def get_single_iou_value(bool_array_left,bool_array_right):
    # 计算两个矩阵中元素为1且相同位置的个数
    count_and = np.sum(bool_array_left & bool_array_right)
    count_or = np.sum(bool_array_left | bool_array_right)

    single_iou = count_and/count_or
    return single_iou

def read_predict_image(image_path):
    image = Image.open(image_path)  # 将 'images.jpg' 替换为您要打开的图像文件路径
    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 打印图像矩阵
    print(image_array)
    print(image)
    # 使用np.unique函数统计元素频率
    unique_elements, counts = np.unique(image_array, return_counts=True)

    pix_value_255_num=0
    # 统计一下255像素的个数
    for row_index in range(image_array.shape[0]):
        for array_index in range(image_array.shape[1]):
            if image_array[row_index][array_index]==248:
                pix_value_255_num+=1
    print(pix_value_255_num)

    # 打印每个元素及其频率
    for element, count in zip(unique_elements, counts):
        print(f"像素值 {element} 的频率为 {count}")
    pass

    # 根据条件操作进行元素重定义
    image_array[image_array > 125] = 255
    image_array[image_array <= 125] = 0
    # 创建新的图像对象并显示（可选）
    new_image = Image.fromarray(image_array)
    new_image.show()

image_path = r"F:\开发板U盘系统备份\toluchang_20230608\result\6.jpg"
image_path2 = r"C:\Users\zhangzuo\Pictures\微信图片_20230606142220.jpg"
read_predict_image(image_path)