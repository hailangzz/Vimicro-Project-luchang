# import os
# from PIL import Image
#
# image_path = r'D:\PycharmProgram\sample_yolov5_calMAP\images\20221101_125117_NF_385.jpg'
# box_info = [451.38913, 422.20929, 496.110443, 437.040405]
#
# cut_info = [box_info[0],box_info[2],box_info[1],box_info[3],]
#
# image = Image.open(image_path)
# cropped = image.crop(cut_info)
# cropped.save("./leftlower_pil_cut.jpg")

bbox = [0.501823,0.78287,0.0713542,0.0472222]
size=[1080,1920]
def xywhn2xyxy(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    xmax = (bbox[0] + bbox[2] / 2.) * size[1]
    ymax = (bbox[1] + bbox[3] / 2.) * size[0]
    box = [xmin, ymin, xmax, ymax]
    return list(map(int, box))


def xywh2xyxy(coco_info=[0.333854,0.559259,0.0114583,0.0537037],image_shape=[1920,1080]):
    w = image_shape[0]*coco_info[2]
    h = image_shape[1]*coco_info[3]

    x1 = image_shape[0]*coco_info[0] - w/2
    y1 = image_shape[1] * coco_info[1] - h/2
    x2 = image_shape[0] * coco_info[0] + w / 2
    y2 = image_shape[1] * coco_info[1] + h / 2

    return int(x1),int(y1),int(x2),int(y2)


xyxy = xywhn2xyxy(bbox,size)
print(xyxy)

xyxy = xywh2xyxy(bbox,image_shape=[1920,1080])
print(xyxy)