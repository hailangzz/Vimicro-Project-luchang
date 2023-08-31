import os
from PIL import Image
import numpy as np
import copy
import math



def bool_image_value(image_array,threshold_value = 125):
    bool_array = copy.deepcopy(image_array)

    # ͼ�����Ԫ����125��Ϊ���ֵ�����Ϊ0��1
    # ����������������Ԫ���ض���
    bool_array[image_array > threshold_value] = True
    bool_array[image_array <= threshold_value] = False

    # print(np.sum(bool_array))
    # # ʹ��np.unique����ͳ��Ԫ��Ƶ��
    # unique_elements, counts = np.unique(bool_array, return_counts=True)
    # # ��ӡÿ��Ԫ�ؼ���Ƶ��
    # for element, count in zip(unique_elements, counts):
    #     print(f"����ֵ {element} ��Ƶ��Ϊ {count}")
    # pass
    return bool_array

def get_single_iou_value(bool_array_left,bool_array_right):
    # ��������������Ԫ��Ϊ1����ͬλ�õĸ���
    count_and = np.sum(bool_array_left & bool_array_right)
    count_or = np.sum(bool_array_left | bool_array_right)

    single_iou = count_and/count_or
    return single_iou

def read_predict_image(image_path):
    image = Image.open(image_path)  # �� 'image.jpg' �滻Ϊ��Ҫ�򿪵�ͼ���ļ�·��
    # ��ͼ��ת��ΪNumPy����
    image_array = np.array(image)
    return image_array

def get_standard_and_predict_mask_file(standard_mask_image_path,predict_image_path):
    standard_and_predict_info = {'compare_mask_image_path':[]}
    stand_mask_image_list = os.listdir(standard_mask_image_path)
    predict_mask_image_list = os.listdir(predict_image_path)

    for mask_image_name in stand_mask_image_list:
        if mask_image_name in predict_mask_image_list:
            couple_image = [os.path.join(standard_mask_image_path,mask_image_name),os.path.join(predict_image_path,mask_image_name)]
            standard_and_predict_info['compare_mask_image_path'].append(couple_image)

    return standard_and_predict_info


def calculate_Iou_value(standard_and_predict_info):
    total_iou_value = 0
    total_couple_num = 0
    for couple_image in standard_and_predict_info['compare_mask_image_path']:

        try:

            image_array0 = read_predict_image(couple_image[0])
            bool_array0 = bool_image_value(image_array0)


            image_array1 = read_predict_image(couple_image[1])
            bool_array1 = bool_image_value(image_array1)

            single_iou = get_single_iou_value(bool_array0, bool_array1)

            if math.isnan(single_iou):
                total_iou_value += 0
            else:
                total_iou_value+=single_iou
            total_couple_num +=1
            print(total_couple_num,single_iou)

        except Exception as e:
            print(e)
            continue
    return total_iou_value/total_couple_num


standard_mask_image_path = r"F:\AiTotalDatabase\ADAS_test_images\standard_lane_mask"
predict_image_path = r"F:\AiTotalDatabase\ADAS_test_images\result"

standard_and_predict_info = get_standard_and_predict_mask_file(standard_mask_image_path,predict_image_path)
iou_value = calculate_Iou_value(standard_and_predict_info)
print(iou_value)