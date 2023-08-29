# 注由于在文字识别流程中，存在样本不均衡的情况，因此要提前除皖之外的其他城市车牌到识别样本内；

import os
import os
import random
import copy
import imgaug.augmenters as iaa
import shutil
from shutil import copy2
from PIL import Image
import imageio


origin_sample_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\detect_CPDD_coco_car_brand_crops_focus_8_pt"
trainfiles = os.listdir(origin_sample_path)  #（图片文件夹）
# print(trainfiles)

trainDir_strengthen = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\detect_CPDD_coco_car_brand_crops_focus_8_pt_strengthen"   #（将图片文件夹中的6份放在这个文件夹下）
# validDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\val\images"     #（将图片文件夹中的2份放在这个文件夹下）
# detectDir = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\test\images"   #（将图片文件夹中的2份放在这个文件夹下）

def random_generate(input_img):
    GaussianBlur_value = random.uniform(0, 1.5)
    GaussianNoise_value = (random.randint(0, 25), random.randint(0, 25))
    Multiply_value = (random.uniform(0.95, 1), random.uniform(1, 1.05))
    GammaContrast_value = (random.uniform(0.95, 1), random.uniform(1, 1.05))

    seq = iaa.Sequential([
        # iaa.Crop(px=(0, 16)),  # 从每侧裁剪图像0到16px（随机选择）
        # iaa.Fliplr(0.5),  # 水平翻转图像
        iaa.GaussianBlur(sigma=(0, GaussianBlur_value)),  # 使用0到3.0的sigma模糊图像
        iaa.AdditiveGaussianNoise(GaussianNoise_value[0], GaussianNoise_value[1]),  # 10~40的高斯噪点
        # # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # 锐化每个图像，使用介于0（无锐化）和1（完全锐化效果）之间的alpha将结果与原始图像覆盖
        #
        iaa.Multiply(Multiply_value, per_channel=0.9),  # 更改图像亮度（原始值的50-150%）。
        iaa.GammaContrast(GammaContrast_value, per_channel=0.9),  # 改善或恶化图像的对比度。
    ])

    images_aug = seq.augment_images(input_img)
    image = Image.fromarray(images_aug)
    return image

# generate_image = self.random_generate(imageio.imread(input_image))
# generate_image.save(os.path.join(save_generate_path,single_info['image_path'].split('\\')[-1].replace('.jpg','g.jpg')))
def add_province_car_number_info(strengthen_num = 5000):
    province_car_number_dict={}
    single_province_info_dict ={'origin_num':0,'add_num':strengthen_num,'origin_image_path_list':[]}
    for image_name in trainfiles:
        # # province_name = image_name.split('-')[0]
        # _, _, box, points, plate, brightness, blurriness = image_name.split('-')
        # # list_plate = plate.split('_')  # 读取车牌
        province_name = image_name.split('-')[4].split('_')[0]
        if province_name not in province_car_number_dict:
            province_car_number_dict[province_name]= copy.deepcopy(single_province_info_dict)
            province_car_number_dict[province_name]['origin_num']=1


        else:
            province_car_number_dict[province_name]['origin_num'] +=1
        province_car_number_dict[province_name]['origin_image_path_list'].append(os.path.join(origin_sample_path, image_name))

    for province_name in province_car_number_dict:
        if province_car_number_dict[province_name]['origin_num']<strengthen_num:
            province_car_number_dict[province_name]['add_num'] -= province_car_number_dict[province_name]['origin_num']
        else:
            province_car_number_dict[province_name]['add_num']=0

    for province_name in province_car_number_dict:
        print(province_name,province_car_number_dict[province_name]['add_num'])

    # 循环创建数据增强样本：
    for province_name in province_car_number_dict:
        print(province_name)
        for index in range(0,province_car_number_dict[province_name]['add_num']):
            random_choince_origin_image_path = random.choice(province_car_number_dict[province_name]['origin_image_path_list'])
            # 增强图像创建
            # print(random_choince_origin_image_path)
            generate_image = random_generate(imageio.imread(random_choince_origin_image_path))
            strengthen_image_name = random_choince_origin_image_path.split('\\')[-1].split('.')[0]+'strengthen'+str(index)+'.jpg'
            generate_image.save(
                os.path.join(trainDir_strengthen, strengthen_image_name))

    # print(sorted(province_car_number_dict.items(), key=lambda item:item[1], reverse=True))

    # for sort_province_num_info in sorted(province_car_number_dict.items(), key=lambda item:item[1], reverse=True):
    #     print(sort_province_num_info)
    #     if



    # return province_car_number_dict

#     # print(image_name.split('-')[4])
#     if image_name.split('-')[4].split('_')[0]!='0':
#         fileName = os.path.join(origin_sample_path, image_name) # （图片文件夹）+图片名=图片地址
#         copy2(fileName, trainDir)
# # for i in index_list:
#     fileName = os.path.join(origin_sample_path, trainfiles[i])  #（图片文件夹）+图片名=图片地址
#     if num < num_train*0.9:  # 7:1:2
#         # print(str(fileName))
#         copy2(fileName, trainDir)
#     elif num < num_train*0.95:
#         # print(str(fileName))
#         copy2(fileName, validDir)
#     else:
#         # print(str(fileName))
#         copy2(fileName, detectDir)
#     num += 1


add_province_car_number_info()