import os
import random
from shutil import copy2

origin_image_path =r"F:\AiTotalDatabase\ADAS_test_images\origin_images"
save_image_path = r"F:\AiTotalDatabase\ADAS_test_images\random_images"

def get_number_random_image(origin_image_path,save_image_path,image_number=800):
    total_image_list = os.listdir(origin_image_path)
    random_image_list = random.sample(total_image_list,image_number)

    for image_name in random_image_list:
        copy2(os.path.join(origin_image_path,image_name), save_image_path)

get_number_random_image(origin_image_path,save_image_path,image_number=800)