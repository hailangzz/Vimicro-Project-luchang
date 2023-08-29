import os
from tqdm import tqdm
from shutil import copy2

origin_database_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别'
Total_real_rec_labels_name = r'rec_car_brand_train_database_focus8/Total_real_rec_labels.txt'

all_rec_labels = open(os.path.join(origin_database_path,Total_real_rec_labels_name),'r',encoding='utf-8').readlines()
# for row_rec_info in all_rec_labels:

image_num =0
for image_index in tqdm(range(len(all_rec_labels))):
    image_num += 1
    row_rec_info = all_rec_labels[image_index]
    buff_row_info_list = row_rec_info.strip().split('\t')
    origin_image_path = os.path.join(origin_database_path,buff_row_info_list[0]).replace('./','/')
    dest_image_name = buff_row_info_list[-1]

    try:
        copy2(origin_image_path,os.path.join(origin_database_path,'rec_car_brand_train_database_focus8_LPR',dest_image_name+'_'+str(image_num)+'.jpg'))
    except Exception as e:
        print(e)
        continue


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
dirpath = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\rec_car_brand_train_database_focus8_LPR"
# 压缩文件夹
do_zip_compress(dirpath)
