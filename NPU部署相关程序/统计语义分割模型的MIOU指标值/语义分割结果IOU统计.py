import os
from PIL import Image
import numpy as np
import copy


def bool_image_value(image_array,threshold_value = 125):
    bool_array = copy.deepcopy(image_array)

    # 图像矩阵元素以125作为区分点区分为0、1
    # 根据条件操作进行元素重定义
    bool_array[image_array > threshold_value] = True
    bool_array[image_array <= threshold_value] = False

    # print(np.sum(bool_array))
    # # 使用np.unique函数统计元素频率
    # unique_elements, counts = np.unique(bool_array, return_counts=True)
    # # 打印每个元素及其频率
    # for element, count in zip(unique_elements, counts):
    #     print(f"像素值 {element} 的频率为 {count}")
    # pass
    return bool_array

def get_single_iou_value(bool_array_left,bool_array_right):
    # 计算两个矩阵中元素为1且相同位置的个数
    count_and = np.sum(bool_array_left & bool_array_right)
    count_or = np.sum(bool_array_left | bool_array_right)

    single_iou = count_and/count_or
    return single_iou

def read_predict_image(image_path):
    image = Image.open(image_path)  # 将 'image.jpg' 替换为您要打开的图像文件路径
    # 将图像转换为NumPy数组
    image_array = np.array(image)
    return image_array



image_path = r"F:\开发板U盘系统备份\toluchang_20230608\result\79.jpg"
image_path2 = r"F:\开发板U盘系统备份\toluchang_20230608\result\80.jpg"
image_array1 = read_predict_image(image_path)
bool_array1 = bool_image_value(image_array1)

image_array2 = read_predict_image(image_path2)
bool_array2 = bool_image_value(image_array2)


single_iou = get_single_iou_value(bool_array1,bool_array2)
print(single_iou)