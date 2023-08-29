import os
from tqdm import tqdm
from PIL import Image

total_images_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\rec_car_brand_train_database_focus8\images'
total_labels_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_COCO_dataset\train\labels'

trainfiles = os.listdir(total_images_path)  #（图片文件夹）

number =0
for image_index in tqdm(range(len(trainfiles))):
    number+=1
    fileName = os.path.join(total_images_path, trainfiles[image_index])
    try:

        # test=os.path.join(total_images_path, r'0104275173611-93_259-224&464_379&531-379&531_230&513_224&464_375&474-0_0_3_25_33_29_29_29-75-6.jpg')
        # print(test)
        image_cur = Image.open(fileName)

    except Exception as e:
        labels_path = os.path.join(total_labels_path, trainfiles[image_index].replace('.jpg','.txt'))
        print('%s:\n this images is error,will been deleted!!!' % labels_path)
        os.remove(fileName) # 删除此图像
        if os.path.exists(labels_path):
            os.remove(labels_path) # 删除此标注信息
        else:
            continue

# 注：遍历整个标签文件，如果标签文件没有对应图像，则删除此标签文件
label_files = os.listdir(total_labels_path)  #（图片文件夹）
for single_label in label_files:
    number += 1
    print(number)
    test_image_name = single_label.replace('.txt','.jpg')
    if test_image_name  not in trainfiles:
        print(test_image_name)
        os.remove(os.path.join(total_labels_path, single_label))




