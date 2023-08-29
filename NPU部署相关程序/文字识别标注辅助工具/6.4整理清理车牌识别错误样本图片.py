import os
from tqdm import tqdm
from PIL import Image

origin_database_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别'
Total_real_rec_labels_name = r'rec_car_brand_train_database_focus8/Total_real_rec_labels.txt'

all_rec_labels = open(os.path.join(origin_database_path,Total_real_rec_labels_name),'r',encoding='utf-8').readlines()
# for row_rec_info in all_rec_labels:
for image_index in tqdm(range(len(all_rec_labels))):

    row_rec_info = all_rec_labels[image_index]
    image_path = (origin_database_path+row_rec_info.split('\t')[0]).replace('./','/')

    # if '20221112_185429_NF_1771_1.jpg' in image_path:
    #     print(image_path)

    try:
        image_cur = Image.open(image_path)
    except Exception as e:
        print(image_path)
        continue

# trainfiles = os.listdir(total_images_path)  #（图片文件夹）
#
# number =0
# for image_index in tqdm(range(len(trainfiles))):
#     number+=1
#     fileName = os.path.join(total_images_path, trainfiles[image_index])
#     try:
#
#         # test=os.path.join(total_images_path, r'0104275173611-93_259-224&464_379&531-379&531_230&513_224&464_375&474-0_0_3_25_33_29_29_29-75-6.jpg')
#         # print(test)
#         image_cur = Image.open(fileName)
#
#     except Exception as e:
#         print(fileName)
#         continue






