import os
import re
import cv2
root_path = r'F:\AiTotalDatabase\CCPD2020\ccpd_green\val\images\\'
file_name = os.listdir(root_path)


def convert_land(size, land_point):
    dw = 1. / size[0]
    dh = 1. / size[1]

    print(land_point)
    for index in range(len(land_point)):
        if index%2==0:
            land_point[index]=int(land_point[index])*dw
        else:
            land_point[index] = int(land_point[index]) * dh

    return land_point

def conver_box(size,box_info):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box_info[0] + box_info[1]) / 2.0
    y = (box_info[2] + box_info[3]) / 2.0
    w = box_info[1] - box_info[0]
    h = box_info[3] - box_info[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


for image_id in file_name:
    print(image_id)
    id = image_id.split('.')[0]
    out_file = open(r'F:\AiTotalDatabase\CCPD2020\ccpd_green\val\labels/%s.txt' % id,
                    'w')  # 需要保存的txt格式文件路径
    img = cv2.imread((root_path + image_id))
    height = img.shape[0]
    width = img.shape[1]
    point_info = image_id.split('.')[0].split('-')[2:4]
    # print(point_info)
    # Xmin = point.split('_')[2].split('&')[0]
    box_point = re.findall('\d+\d*', point_info[0])  # 正则表达式 从字符串中提取数值
    # print(box_point)
    Xmin = min(box_point[0::2])  # list[start:stop:step]
    Ymin = min(box_point[1::2])
    Xmax = max(box_point[0::2])
    Ymax = max(box_point[1::2])
    box_info = (float(Xmin), float(Xmax), float(Ymin),
         float(Ymax))
    # print(box_info)
    box_label = conver_box((width, height), box_info)

    land_point = re.findall('\d+\d*', point_info[1])  # 正则表达式 从字符串中提取数值

    land_point = convert_land((width, height), land_point)

    total_info = box_label+land_point
    out_file.write('0' + " " + " ".join([str(a) for a in total_info]) + '\n')
print('end')
