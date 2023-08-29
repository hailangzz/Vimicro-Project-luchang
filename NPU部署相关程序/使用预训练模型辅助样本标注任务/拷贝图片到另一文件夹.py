import numpy as np
import os
from shutil import copy2

total_images_path = r'/home/database/IntelligentTransportation/Vimicro_Image_dataset'
origin_sample_path = r"/home/zhangzhuo/total_dataset/Vimicro_Image_dataset"


total_images_list = os.listdir(total_images_path)
trainfiles = os.listdir(origin_sample_path)

for single_image in trainfiles:
    if single_image not in total_images_list:
        fileName = os.path.join(origin_sample_path, single_image)

        copy2(fileName, total_images_path)
