import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans

from tqdm import tqdm, trange
import os
import random

def get_thr_coco_value(path='./data/ccpd.yaml'): # 获取标注框的长宽比范围值
    with open(path, 'rb') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        print(data_dict)

    total_images_list = os.listdir(data_dict['train']+'/../labels')

    w_h_min_rate=10
    w_h_max_rate=0
    for single_images_name in total_images_list:
        # print(single_images_name)
        save_label_cur = open(os.path.join(r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\traffic_light_detection_coco_buff\val\labels',single_images_name),'w')

        with open(os.path.join(data_dict['val']+'/../labels',single_images_name),'r') as label_cur:
            all_target_row_info = label_cur.readlines()
            for single_row in all_target_row_info:
                save_label_cur.writelines(single_row.replace('  ',' '))
        save_label_cur.close()


get_thr_coco_value()