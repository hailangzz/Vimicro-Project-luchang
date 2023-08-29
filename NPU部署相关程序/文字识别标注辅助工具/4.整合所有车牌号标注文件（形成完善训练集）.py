import os
from shutil import copy2

save_car_brand_train_database_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\rec_car_brand_train_database_focus8\images'
save_real_rec_labels_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\rec_car_brand_train_database_focus8\Total_real_rec_labels.txt'


origin_file_path_dict={'images_origin_path':r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops',
                       'rec_labels_origin_path':r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\all_detect_crops_database_labels_file'}

def deal_train_database(origin_file_path_dict,save_car_brand_train_database_path,save_real_rec_labels_path):


    #注打开所有标签整合保存txt文件的句柄；
    save_real_rec_file_cur = open(save_real_rec_labels_path,'w',encoding='utf-8')

    # 首先处理标注文件信息
    all_rec_labels_file_name_list = os.listdir(origin_file_path_dict['rec_labels_origin_path'])

    for single_rec_label_file_name in all_rec_labels_file_name_list:
        if single_rec_label_file_name in ['detect_SZ_DVR_car_brand_crops_focus_8_pt_labels.txt',
                                          'detect_CPDD_coco_car_brand_crops_focus_8_pt_labels.txt',
                                          'detect_PersonCarOther_car_brand_crops_focus_8_pt_labels.txt',
                                          'detect_CPDD_coco_car_brand_crops_focus_8_pt_strengthen_labels.txt']:

            # print(single_rec_label_file_name)
            rec_label_file_cur = open(os.path.join(origin_file_path_dict['rec_labels_origin_path'],single_rec_label_file_name),'r',encoding='utf-8')
            all_rec_label_info = rec_label_file_cur.readlines()
            for row_info in all_rec_label_info:
                save_real_rec_file_cur.write('./rec_car_brand_train_database_focus8/images/'+row_info.split('\\')[-1])

    save_real_rec_file_cur.close()

    from tqdm import tqdm

    # 以下将车牌号识别图像拷贝到指定路径下
    all_images_file_name_list = os.listdir(origin_file_path_dict['images_origin_path'])
    for single_image_file_name in all_images_file_name_list:
        if single_image_file_name in ['detect_CPDD_coco_car_brand_crops_focus_8_pt',
                                        'detect_PersonCarOther_car_brand_crops_focus_8_pt',
                                        'detect_SZ_DVR_car_brand_crops_focus_8_pt',
                                        'detect_CPDD_coco_car_brand_crops_focus_8_pt_strengthen']:

            now_images_path = os.path.join(origin_file_path_dict['images_origin_path'],single_image_file_name)
            print(single_image_file_name)
            # for image_name in os.listdir(now_images_path):
            now_total_image_list = os.listdir(now_images_path)
            for index in tqdm(range(len(now_total_image_list))):
                image_name = now_total_image_list[index]
                try:
                    copy2(os.path.join(now_images_path,image_name),save_car_brand_train_database_path)
                except:
                    continue


deal_train_database(origin_file_path_dict,save_car_brand_train_database_path,save_real_rec_labels_path)


"""
实战场景: 如何压缩一个文件夹
"""

# 导入系统包
import platform
import os
import zipfile

def do_zip_compress(dirpath):
    print("原始文件夹路径：" + dirpath)
    output_name = f"{dirpath}.zip"
    parent_name = os.path.dirname(dirpath)
    print("压缩文件夹目录：", parent_name)
    zip = zipfile.ZipFile(output_name, "w", zipfile.ZIP_DEFLATED)
    # 多层级压缩
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if str(file).startswith("~$"):
                continue
            filepath = os.path.join(root, file)
            print("压缩文件路径：" + filepath)
            writepath = os.path.relpath(filepath, parent_name)
            zip.write(filepath, writepath)
    zip.close()

# 需要先创建文件夹resources
dirpath = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\rec_car_brand_train_database_focus8"
# 压缩文件夹
do_zip_compress(dirpath)

