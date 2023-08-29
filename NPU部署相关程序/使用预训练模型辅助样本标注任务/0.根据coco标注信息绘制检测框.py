import os
import cv2
from PIL import Image, ImageDraw

''' # cv2 处理图像标注框box
def xywh2xyxy(coco_info=[0.333854,0.559259,0.0114583,0.0537037],image_shape=[1920,1080]):
    print(image_shape)
    w = image_shape[1]*coco_info[2]
    h = image_shape[0]*coco_info[3]

    x1 = image_shape[1]*coco_info[0] - w/2
    y1 = image_shape[0] * coco_info[1] - h/2
    x2 = image_shape[1] * coco_info[0] + w / 2
    y2 = image_shape[0] * coco_info[1] + h / 2

    return int(x1),int(y1),int(x2),int(y2)

def drawbox_in_image(image_path,box_info_list,save_write_box_path=r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\bdd_100k_model_draw_box_images'):
    image = cv2.imread(image_path)
    image_shape = image.shape
    for box_info in box_info_list:
        # 读取每个xywh标注的xyxy坐标
        x1,y1,x2,y2 = xywh2xyxy(box_info,image_shape)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    # print(image_path)
    save_write_box_image_path = os.path.join(save_write_box_path,image_path.split('\\')[-1])
    print(save_write_box_image_path)
    # print(image)

    # cv2.imshow('test',image)
    # cv2.waitKey(0)  # 等待按键
    cv2.imwrite(save_write_box_image_path,image)
'''

def xywh2xyxy(coco_info=[0.333854,0.559259,0.0114583,0.0537037],image_shape=[1920,1080]):
    w = image_shape[0]*coco_info[2]
    h = image_shape[1]*coco_info[3]

    x1 = image_shape[0]*coco_info[0] - w/2
    y1 = image_shape[1] * coco_info[1] - h/2
    x2 = image_shape[0] * coco_info[0] + w / 2
    y2 = image_shape[1] * coco_info[1] + h / 2

    return int(x1),int(y1),int(x2),int(y2)

def drawbox_in_image(image_path, box_info_list,
    save_write_box_path=r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\bdd_100k_model_draw_box_images'):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    image_shape = image.size

    for box_info in box_info_list:
        # 读取每个xywh标注的xyxy坐标
        x1, y1, x2, y2 = xywh2xyxy(box_info, image_shape)
        draw.rectangle([x1, y1, x2, y2],outline='red',width=2)
    save_write_box_image_path = os.path.join(save_write_box_path, image_path.split('\\')[-1])
    # print(save_write_box_image_path)
    image.save(save_write_box_image_path)

def get_image_box_info_list(single_image_label_path):
    box_info_list = []
    label_cur = open(single_image_label_path,'r')
    all_row_label_info = label_cur.readlines()
    for row_label in all_row_label_info:
        row_label = row_label.strip()
        row_label = [float(float_value) for float_value in row_label.split(' ')[1:]]
        box_info_list.append(row_label)
        # print(row_label)
    # print(box_info_list)
    return box_info_list


def get_total_image_info(origin_image_path=r'D:\Git_WareHouse\yolov5-master\traffic_light_test_DVR',origin_coco_label_path=r'D:\Git_WareHouse\yolov5-master\detect\exp\labels'):
    total_image_name_list = os.listdir(origin_coco_label_path)
    for single_image_label_name in total_image_name_list:
        single_image_name = single_image_label_name.replace('.txt','.jpg')
        single_image_path = os.path.join(origin_image_path,single_image_name)
        single_image_label_path = os.path.join(origin_coco_label_path, single_image_label_name)

        box_info_list = get_image_box_info_list(single_image_label_path)
        # 绘制图片的检测框图
        drawbox_in_image(single_image_path, box_info_list)


get_total_image_info()
