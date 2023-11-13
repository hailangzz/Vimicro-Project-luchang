import os
from shutil import copy2
from PIL import Image
import numpy as np
import cv2
import cv2
origin_image_path = r"F:\AiTotalDatabase\ADAS_test_images\origin_images"
save_resize_image_path = r"F:\AiTotalDatabase\ADAS_test_images\origin_images_640_640"

all_origin_image_name = os.listdir(origin_image_path)

datalist_cur = open("datalist.txt",'w')

image_num = 0
for single_image_name in all_origin_image_name:

    if '.jpg' in single_image_name or '.png' in single_image_name:
        single_image_path = os.path.join(origin_image_path,single_image_name)
        # origin_image =  Image.open(single_image_path)
        # resize_image = origin_image.resize((640, 640))

        # origin_image = cv2.imread(single_image_path.encode('utf-8'))
        origin_image = cv2.imdecode(np.fromfile(single_image_path, dtype=np.uint8), -1)
        resize_image = cv2.resize(origin_image, (640, 640))


        save_image_name = r"yolop_quant_image_"+str(image_num)+".jpg"
        datalist_cur.write(save_image_name)
        datalist_cur.write("\n")
        # resize_image.save(os.path.join(save_resize_image_path,save_image_name))
        cv2.imwrite(os.path.join(save_resize_image_path,save_image_name), resize_image)

        image_num+=1
        if image_num>10:
            break

datalist_cur.close()
