import os
import random

import shutil
from shutil import copy2
origin_sample_path = r"/home/zhangzhuo/total_dataset/Vimicro_Image_dataset"

origin_label_path = r"/home/zhangzhuo/git_workspace/yolov5/runs/detect/exp5/labels"
save_extract_traffic_light = r'/home/zhangzhuo/total_dataset/extract_detect_traffic_light'

all_image_name_list = [txt_file_name.replace('.txt','.jpg') for txt_file_name in os.listdir(origin_label_path)]  #（图片文件夹）
# print(all_image_name_list)
for image_name in all_image_name_list:
    image_path = os.path.join(origin_sample_path,image_name)
    copy2(image_path, save_extract_traffic_light)

