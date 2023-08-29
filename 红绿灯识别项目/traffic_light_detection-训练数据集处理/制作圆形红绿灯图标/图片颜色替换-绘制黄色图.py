from PIL import Image
import numpy as np
import os

red_image_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground\RC_a'
green_image_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground\GC_1'
save_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground\YC_a'


def creat_yellow_mask_image(green_image_path,red_image_path):
    for image_name in os.listdir(green_image_path):
        green_image = np.array(Image.open(os.path.join(green_image_path,image_name)).convert('RGB'))
        red_image = np.array(Image.open(os.path.join(red_image_path,image_name)).convert('RGB'))

        # square1 = np.array(Image.open(file_name1).convert('RGB'))
        # square2 = np.array(Image.open(file_name2).convert('RGB'))

        red_image[:,:,1]=green_image[:,:,1]
        # square3 = (square1+square2)/2
        # print(square3.shape)
        # image=Image.fromarray(np.uint8(square1))
        image=Image.fromarray(red_image)
        # image.show()
        save_image_path = os.path.join(save_path,image_name)
        image.save(save_image_path)


creat_yellow_mask_image(green_image_path,red_image_path)