import os,shutil
from tqdm import tqdm
from PIL import Image

total_images_path = r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\search_extract_detect_traffic_light4'
total_labels_path = r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\Label_search_extract_detect_traffic_light4'

imagefiles = os.listdir(total_images_path)  #（图片文件夹）
total_labels_name_list = os.listdir(total_labels_path)

def  cheack_number_of_image_nothave_labels():
    number =0
    for image_index in tqdm(range(len(imagefiles))):

        # fileName = os.path.join(total_images_path, labelfiles[image_index].replace('.txt','.jpg'))
        fileName = imagefiles[image_index]
        if fileName.replace('.jpg','.txt') not in total_labels_name_list:
            # print(fileName)
            number += 1
    print('this have %d numbers image have not labels file!'%number)

# 将不需要的样本移除到指定路径下()
def move_cheack_images(move_path=r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\move_image_path\move_search_extract_detect_traffic_light4'):
    number = 0
    for image_index in tqdm(range(len(imagefiles))):

        # fileName = os.path.join(total_images_path, labelfiles[image_index].replace('.txt','.jpg'))
        fileName = imagefiles[image_index]
        if fileName.replace('.jpg', '.txt') not in total_labels_name_list:
            # print(fileName)
            number += 1

            src_path = os.path.join(total_images_path,fileName)
            dest_path = os.path.join(move_path,fileName)
            # os.remove(fileName)  # 删除此图像
            shutil.move(src_path, dest_path)
    print('this have %d numbers image have been removed!' % number)

cheack_number_of_image_nothave_labels()
move_cheack_images()
