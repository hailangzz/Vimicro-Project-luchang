import numpy as np
import os
from shutil import copy2


def check_save_dir(save_train_datasets_path):
    if "train" not in os.listdir(save_train_datasets_path):
        train_path = os.path.join(save_train_datasets_path,'train')
        os.mkdir(train_path)
        os.mkdir(os.path.join(train_path,'images'))
        os.mkdir(os.path.join(train_path, 'labels'))

def change_labels_map_and_create_datasets(origin_labels_sample_path,origin_images_path,save_train_datasets_path,
                                          change_label_map={'0':'1'}
                                          #change_label_map={'oringin_label':'0','change_label':'1'}
                                          ):
    check_save_dir(save_train_datasets_path)
    all_labels_list = os.listdir(origin_labels_sample_path)
    for single_label_name in all_labels_list:
        single_image_name = single_label_name.replace('.txt','.png')

        label_cur = open(os.path.join(origin_labels_sample_path,single_label_name),'r')
        save_label_cur = open(os.path.join(save_train_datasets_path,'train','labels',single_label_name),'w')
        label_info = label_cur.readlines()
        for row_info in label_info:
            # wright_row_info = row_info.replace(change_label_map['oringin_label'],change_label_map['change_label'],1)
            labelname,box_info = row_info.split(' ',1)

            if labelname in change_label_map:
                labelname = change_label_map[labelname]
            wright_info = ' '.join([labelname,box_info])
            save_label_cur.write(wright_info)
        label_cur.close()
        save_label_cur.close()
        # 拷贝图片到指定目录下
        image_path = os.path.join(origin_images_path, single_image_name)  # （图片文件夹）+图片名=图片地址
        copy2(image_path, os.path.join(save_train_datasets_path,'train','images'))

origin_labels_sample_path = r"D:\PycharmProgram\ZZ_Total_DeepLearning_Model\yolov7-face-main\VOC2020_person_street\wide_face_abroad_detect\labels"
origin_images_path = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\INRIAPerson-823\JPEGImages"
save_train_datasets_path = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\add_person_street_face_datasets"

change_label_map={'0':'1'}
change_labels_map_and_create_datasets(origin_labels_sample_path,origin_images_path,save_train_datasets_path,change_label_map={'0':'1'})
