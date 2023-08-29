import os
import pickle

import shutil
from shutil import copy
from scipy import spatial

# # 将图片拷贝到同一目录，并进行统一编码，计算相似度。
#
# CircularBackground_output_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground'
# total_image_save_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\total_image_save\\'
#
# for classify_name in os.listdir(CircularBackground_output_path):
#     image_path = os.path.join(CircularBackground_output_path,classify_name)
#     for image_name in os.listdir(image_path):
#         copy(os.path.join(image_path,image_name), total_image_save_path+classify_name+'_'+image_name)

origin_image_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\test'
sort_image_save_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\sort_total_image_save'
# 读取total_image_embedding 存储信息：
total_image_embedding_pickle = pickle.load(open(os.path.join(origin_image_path,'total_image_embedding.p'), mode='rb'))

# 计算余弦相似度时的标准图像
stand_image = list(total_image_embedding_pickle.keys())[0]
stand_image_matrix = total_image_embedding_pickle[stand_image]

image_embedding_dict = {'image_name':[],'cosine_similar_value':[],'similar_sort_index':[]}

from numpy import dot
from numpy.linalg import norm
import numpy as np


for image_name in total_image_embedding_pickle:
    image_embedding_dict['image_name'].append(image_name)
    b_image_matrix = total_image_embedding_pickle[image_name]

    # res = sum(abs(stand_image_matrix- b_image_matrix))
    res = 1 - spatial.distance.cosine(stand_image_matrix, b_image_matrix)
    print(res)
    image_embedding_dict['cosine_similar_value'].append(res)

image_embedding_dict['similar_sort_index']=np.argsort(image_embedding_dict['cosine_similar_value'])


for index in range(len(image_embedding_dict['similar_sort_index'])):
    image_name = image_embedding_dict['image_name'][image_embedding_dict['similar_sort_index'][index]]

    print(os.path.join(origin_image_path, image_name))
    copy(os.path.join(origin_image_path, image_name),sort_image_save_path+'\\'+str(index)+'.png')

    pass

