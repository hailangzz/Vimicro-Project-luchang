import numpy as np
import os
from shutil import copy2,move
import random

origin_train_path = r'D:\迅雷下载\AI数据集汇总\test_vidoe_Image_dataset\train'
save_val_path = r'D:\迅雷下载\AI数据集汇总\test_vidoe_Image_dataset\val'

def create_val_dataset(val_rate=0.1,move_flag = False):
    search_val_info_dict = {'val_num':0,'val_image_label_list':[]}
    if not os.path.exists(save_val_path):
        print(save_val_path)
        os.mkdir(save_val_path)
        os.mkdir(os.path.join(save_val_path,'images'))
        os.mkdir(os.path.join(save_val_path, 'labels'))

    total_origin_label_list = os.listdir(os.path.join(origin_train_path,'labels'))
    search_val_info_dict['val_num'] = int(len(total_origin_label_list)*val_rate)
    search_val_info_dict['val_image_label_list'] = random.sample(total_origin_label_list, search_val_info_dict['val_num'])
    # print(search_val_info_dict)


    # 将样本imges、labels移动到指定的测试集目录中
    for search_sample in search_val_info_dict['val_image_label_list']:
        origin_image_path = os.path.join(origin_train_path,'images',search_sample.replace('.txt','.jpg'))
        origin_label_path = os.path.join(origin_train_path,'labels',search_sample)
        # print(origin_image_path,origin_label_path)

        dst_image_path = os.path.join(save_val_path,'images',search_sample.replace('.txt','.jpg'))
        dst_label_path = os.path.join(save_val_path, 'labels', search_sample)
        # 复制文件到测试集
        if move_flag:
            move(origin_image_path, dst_image_path)
            move(origin_label_path, dst_label_path)
        else:
            copy2(origin_image_path,dst_image_path)
            copy2(origin_label_path, dst_label_path)

create_val_dataset(move_flag=True)


