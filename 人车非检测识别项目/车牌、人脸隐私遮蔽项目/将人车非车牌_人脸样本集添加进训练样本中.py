#coding=utf-8
import os
from shutil import copy2

origin_total_image_datasets = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\train"
now_image_datasets = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\val"
save_file_path = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\person_car_other_add_datasets"

origin_total_file_name = [file_name.split('.txt')[0] for file_name in os.listdir(os.path.join(origin_total_image_datasets,'labels'))]
now_total_file_name = [now_file_name.split('.txt')[0] for now_file_name in os.listdir(os.path.join(now_image_datasets,'labels'))]

def create_save_datasets_direct(save_file_path):
    save_image_path = os.path.join(save_file_path,'images')
    save_label_path = os.path.join(save_file_path, 'labels')
    if 'images' not in os.listdir(save_file_path):
        os.mkdir(save_image_path)
        os.mkdir(save_label_path)

    return save_image_path,save_label_path

save_image_path,save_label_path = create_save_datasets_direct(save_file_path)
for file_name in origin_total_file_name:

    if len(file_name)>=77 and file_name not in now_total_file_name:
        origin_image_name = os.path.join(origin_total_image_datasets,'images',file_name+'.jpg')
        origin_label_name = os.path.join(origin_total_image_datasets, 'labels', file_name + '.txt')
        print(file_name)

        #拷贝图像
        copy2(origin_image_name, save_image_path)
        #拷贝标注
        copy2(origin_label_name, save_label_path)



# min_lenth = 9999
# number_image = 0
# number_image2 = 0
# for file_name in origin_total_file_name:
#     if len(file_name)>=81:
#         number_image+=1
#     if len(file_name)<min_lenth:
#         min_lenth = len(file_name)
#     number_image2 += 1
# print(min_lenth,number_image,number_image2)