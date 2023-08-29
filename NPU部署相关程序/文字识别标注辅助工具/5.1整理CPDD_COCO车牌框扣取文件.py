import os
from shutil import copy2
from tqdm import tqdm



detect_car_brand_label_path = r'D:\Git_WareHouse\yolov5-master\runs\detect\CPDD_car_brand\labels'
total_detect_car_brand_label_name_list = os.listdir(detect_car_brand_label_path)
save_search_CPDD_car_brand_labels_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\all_detect_crops_database_labels_file\detect_CPDD_coco_car_brand_focus_8_pt_labels'

def check_useful_labels(detect_car_brand_label_path,total_detect_car_brand_label_name_list,save_search_CPDD_car_brand_labels_path):
    number = 0
    # for single_label_name in total_detect_car_brand_label_name_list:
    for index in tqdm(range(len(total_detect_car_brand_label_name_list))):
        single_label_name = total_detect_car_brand_label_name_list[index]
        single_label_path = os.path.join(detect_car_brand_label_path,single_label_name)
        label_file_info = open(single_label_path,'r').readlines()
        if len(label_file_info)==1: #当每张图片只有一张车牌检测框时，其能对应获取标注车牌文本识别信息。
            # number+=1
            # print(single_label_path)
            # # print(label_file_info)
            # print(number)
            copy2(single_label_path, save_search_CPDD_car_brand_labels_path)

check_useful_labels(detect_car_brand_label_path,total_detect_car_brand_label_name_list,save_search_CPDD_car_brand_labels_path)