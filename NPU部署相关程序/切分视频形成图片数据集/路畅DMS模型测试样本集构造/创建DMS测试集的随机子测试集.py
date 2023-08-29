import numpy as np
import os
from shutil import copy2,move
import random

origin_train_path = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\LuChang'
save_val_path = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\LuChang_part'

val_rate =0.25
total_origin_label_list = os.listdir(origin_train_path)
search_image_number = int(len(total_origin_label_list)*val_rate)

search_image_name_list = random.sample(total_origin_label_list, search_image_number)

for search_sample in search_image_name_list:
    origin_image_path = os.path.join(origin_train_path,search_sample)
    dst_image_path = os.path.join(save_val_path, search_sample)
    copy2(origin_image_path, dst_image_path)