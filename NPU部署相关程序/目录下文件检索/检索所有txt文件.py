import numpy as np
import os
from shutil import copy2

total_images_path = r'/home/zhangzhuo/git_workspace/yolov5_FocusNewnet/datasets/PrivacyMaskingCarPlateFaceDatasets/train_total/images'

total_file = os.listdir(total_images_path)
for single_file in total_file:
    if '.txt' in single_file:
        print(single_file)