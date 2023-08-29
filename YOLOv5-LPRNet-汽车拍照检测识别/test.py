# import numpy as np
# import os
# from shutil import copy2
#
# total_images_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\train\images'
#
#
# origin_sample_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD2019\ccpd_base"
#
# save_images_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_buff\train\images'
# total_images_list = os.listdir(total_images_path)
# trainfiles = os.listdir(origin_sample_path)  #（图片文件夹）
#
# print(len(total_images_list),len(trainfiles))
#
#
# reduce_number = 0
# for single_image in trainfiles:
#     if single_image not in total_images_list:
#         reduce_number+=1
#         fileName = os.path.join(origin_sample_path, single_image)  # （图片文件夹）+图片名=图片地址
#         copy2(fileName, save_images_path)
#
# print(reduce_number)

CHARS = ['0','1','2','3','4','5','6','7','8','9',
         'A','B','C','D','E','F','G','H','J','K',
         'L','M','N','P','Q','R','S','T','U','V',
         'W','X','Y','Z',
         '京','津','冀','蒙','青','陕','甘','宁',
         '晋','鲁','豫','皖','鄂','云','贵','桂',
         '川','渝','新','藏','黑','吉','辽','浙',
         '苏','沪','湘','闽','赣','琼',
         '粤','港','澳','学','挂','警','使','应','急', '-'
         ]

import os

all_pictrue_name_list = os.listdir(r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\rec_car_brand_train_database_focus8_LPR')
for single_image_name in all_pictrue_name_list:
    for char in single_image_name.split('_')[0]:
        if char not in CHARS:
            print(single_image_name)