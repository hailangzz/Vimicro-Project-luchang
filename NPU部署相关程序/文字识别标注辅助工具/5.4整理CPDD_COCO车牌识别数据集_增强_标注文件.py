# 制作车牌识别数据集~~~

import cv2
import os
import numpy as np

# 参考 https://blog.csdn.net/qq_36516958/article/details/114274778
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-create-labels
from PIL import Image
# CCPD针对省份样本不均衡，采取的数据增强操作后。为其设计标注数据文件



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

def create_rec_image_dataset(origin_image_path,save_CPDD_labels_path):

    save_labels_cur = open(save_CPDD_labels_path,'w',encoding='utf-8')

    for filename in os.listdir(origin_image_path):
        real_label_str =''
        _, _, box, points, plate, brightness, blurriness = filename.split('-')
        list_plate = plate.split('_')  # 读取车牌

        for char_index in range(len(list_plate)):
            int_list_value = int(list_plate[char_index])
            if char_index==0:
                real_label_str+=provinces[int_list_value]
            elif char_index==1:
                real_label_str += alphabets[int_list_value]
            else:
                real_label_str += ads[int_list_value]

        save_labels_cur.write(filename+'\t'+real_label_str+'\n')

    save_labels_cur.close()
        #注以下为lprnet训练样本构造是的操作
        # print(filename)
        # result = ""
        # _, _, box, points, plate, brightness, blurriness = filename.split('-')
        # list_plate = plate.split('_')  # 读取车牌
        # if len(list_plate)==8:
        #     print(filename)
        #     print(list_plate)
        #
        #
        # result += str(CHARS_DICT[provinces[int(list_plate[0])]])
        # result += str(CHARS_DICT[alphabets[int(list_plate[1])]])
        # result += str(CHARS_DICT[ads[int(list_plate[2])]]) + str(CHARS_DICT[ads[int(list_plate[3])]]) + str(
        #     CHARS_DICT[ads[int(list_plate[4])]]) + str(CHARS_DICT[ads[int(list_plate[5])]]) + str(
        #     CHARS_DICT[ads[int(list_plate[6])]])
        # print(result)

total_strengthen_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\detect_CPDD_coco_car_brand_crops_focus_8_pt_strengthen'

save_CPDD_labels_path=r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\all_detect_crops_database_labels_file\detect_CPDD_coco_car_brand_crops_focus_8_pt_strengthen_labels.txt'
create_rec_image_dataset(total_strengthen_path,save_CPDD_labels_path)
