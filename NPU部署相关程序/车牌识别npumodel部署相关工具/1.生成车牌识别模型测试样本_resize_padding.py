import os
import cv2
import numpy as np
import random

class create_car_brand_rec_sample:

    def get_random_image_list(self,origin_image_path,num=10):
        total_image_list = os.listdir(origin_image_path)
        return random.sample(total_image_list,num)


    def letterbox(self, img, new_shape=(48, 120), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


if __name__ == '__main__':
    image_source_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\real_car_brand_test'
    save_padding_image_path = r'D:\\test_car_brand_rec_padding'
    create_rec_sample = create_car_brand_rec_sample()
    car_brand_images = create_rec_sample.get_random_image_list(image_source_path)

    for single_car_brand in car_brand_images:
        image_path = os.path.join(image_source_path,single_car_brand)
        # print(image_path)
        # image = cv2.imread(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

        image = create_rec_sample.letterbox(image)
        cv2.imwrite(os.path.join(save_padding_image_path,single_car_brand),image[0])

        # print(image_path)
        pass
