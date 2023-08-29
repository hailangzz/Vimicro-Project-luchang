import numpy as np
import os
from shutil import copy2,move
import random


def create_val_dataset(split_number = {'train':50000,'val':3000},move_flag = False):

    search_val_info_dict = {'val_num': 0, 'val_image_label_list': []}

    for key_name in split_number:
        total_origin_label_list = os.listdir(os.path.join(origin_train_path, 'labels'))
        search_val_info_dict['val_num'] = split_number[key_name]
        search_val_info_dict['val_image_label_list'] = random.sample(total_origin_label_list,
                                                                     search_val_info_dict['val_num'])

        if key_name == 'train':
            save_path=save_train_path
        else:
            save_path = save_val_path
        if not os.path.exists(save_path):
            print(save_path)
            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path,'images'))
            os.mkdir(os.path.join(save_path, 'labels'))





        # 将样本imges、labels移动到指定的测试集目录中
        for search_sample in search_val_info_dict['val_image_label_list']:
            origin_image_path = os.path.join(origin_train_path,'images',search_sample.replace('.txt','.jpg'))
            origin_label_path = os.path.join(origin_train_path,'labels',search_sample)
            # print(origin_image_path,origin_label_path)

            dst_image_path = os.path.join(save_path,'images',search_sample.replace('.txt','.jpg'))
            dst_label_path = os.path.join(save_path, 'labels', search_sample)
            # 复制文件到测试集
            if move_flag:
                move(origin_image_path, dst_image_path)
                move(origin_label_path, dst_label_path)
            else:
                copy2(origin_image_path,dst_image_path)
                copy2(origin_label_path, dst_label_path)

origin_train_path = r'/home/zhangzhuo/git_workspace/yolov5_FocusNewnet/datasets/PrivacyMaskingCarPlateFaceDatasets/train_total/'
save_train_path = r'/home/zhangzhuo/git_workspace/yolov5_FocusNewnet/datasets/PrivacyMaskingCarPlateFaceDatasets/train'
save_val_path = r'/home/zhangzhuo/git_workspace/yolov5_FocusNewnet/datasets/PrivacyMaskingCarPlateFaceDatasets/val'

split_number = {'train':50000,'val':5000}

create_val_dataset(split_number,move_flag=True)


