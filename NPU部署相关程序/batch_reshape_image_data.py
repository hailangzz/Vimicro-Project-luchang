from PIL import Image
import os


def image_resize(image_path, new_path):  
    print('============>>修改图片尺寸')
    for img_name in os.listdir(image_path):
        img_path = image_path + "/" + img_name
        image = Image.open(img_path)
        image = image.resize((1024,576))
        # process the 1 channel image
        image.save(new_path + '/' + img_name)
    print("end the processing!")


if __name__ == '__main__':
    print("ready for ::::::::  ")
    ori_path = r"./car_brand/"
    new_path = r'./test_car_brand/'
    image_resize(ori_path, new_path)
