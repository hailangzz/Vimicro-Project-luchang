"""
@Author: HuKai
@Date: 2022/5/29  21:24
@github: https://github.com/HuKai97
"""
# 制作车牌识别数据集~~~

import cv2
import os
import numpy as np

# 参考 https://blog.csdn.net/qq_36516958/article/details/114274778
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-create-labels
from PIL import Image
# CCPD车牌有重复，应该是不同角度或者模糊程度

total_rec_path_info={
                     'train':r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_dataset\train\images',
                     'val':r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_dataset\val\images',
                     'test':r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_dataset\test\images',
                     }


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


train_label_CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                     '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                     '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                     '新',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                     'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z', 'I', 'O', '-'
                    ]
CHARS_DICT = {char:i for i, char in enumerate(train_label_CHARS)}   # {'京': 0, '沪': 1, '津': 2, '渝': 3, '冀': 4, '晋':


def create_rec_image_dataset(rec_type,image_path):

    num = 0
    for filename in os.listdir(image_path):

        result = ""
        _, _, box, points, plate, brightness, blurriness = filename.split('-')
        list_plate = plate.split('_')  # 读取车牌
        # print(list_plate)

        result += str(CHARS_DICT[provinces[int(list_plate[0])]])
        result += str(CHARS_DICT[alphabets[int(list_plate[1])]])
        result += str(CHARS_DICT[ads[int(list_plate[2])]]) + str(CHARS_DICT[ads[int(list_plate[3])]]) + str(CHARS_DICT[ads[int(list_plate[4])]]) + str(CHARS_DICT[ads[int(list_plate[5])]]) + str(CHARS_DICT[ads[int(list_plate[6])]])
        # 新能源车牌的要求，如果不是新能源车牌可以删掉这个if
        # if result[2] != 'D' and result[2] != 'F' \
        #         and result[-1] != 'D' and result[-1] != 'F':
        #     print(filename)
        #     print("Error label, Please check!")
        #     assert 0, "Error label ^~^!!!"
        # print(result) #result为将要识别的车牌号码！！
        img_path = os.path.join(image_path, filename)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        # img = cv2.imread(img_path)
        assert os.path.exists(img_path), "images file {} dose not exist.".format(img_path)

        box = box.split('_')  # 车牌边界
        box = [list(map(int, i.split('&'))) for i in box]

        xmin = box[0][0]
        xmax = box[1][0]
        ymin = box[0][1]
        ymax = box[1][1]

        img = Image.fromarray(img)
        img = img.crop((xmin, ymin, xmax, ymax))  # 裁剪出车牌位置
        img = img.resize((94, 24), Image.LANCZOS)
        img = np.asarray(img)  # 转成array,变成24*94*3

        cv2.imencode('.jpg', img)[1].tofile(r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_images\{}\images\{}_{}.jpg".format(rec_type,result,str(num)))
        # 图片中文名会报错
        # cv2.imwrite(r"K:\MyProject\datasets\ccpd\new\ccpd_2020\rec_images\train\{}.jpg".format(result), img)  # 改成自己存放的路径
        num += 1
    print("{}:共生成{}张识别图片".format(rec_type,num))


for rec_type,image_path in total_rec_path_info.items():
    create_rec_image_dataset(rec_type, image_path)
