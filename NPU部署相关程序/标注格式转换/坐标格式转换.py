
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

def xyxy2xywhn(object, width, height):
    cat_id = object[0]
    xn = object[1] / width
    yn = object[2] / height
    wn = object[3] / width
    hn = object[4] / height
    out = "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(cat_id, xn, yn, wn, hn)
    return out

origin_image_size=[1920,1080]
model_size=[1024,576]
def fours_coco_2_xyxy(bbox,model_size=[1024,576],origin_image_size=[1920,1080]): #开发板标注框转换函数
    x1=int(bbox[0]*origin_image_size[0]/model_size[0])
    y1 = int(bbox[1] * origin_image_size[1] / model_size[1])
    x2 = int(bbox[2] * origin_image_size[0] / model_size[0])
    y2 = int(bbox[3] * origin_image_size[1] / model_size[1])

    return x1,y1,x2,y2

xyxy = fours_coco_2_xyxy([477.858337,437.364990,549.640808,463.634705])
print(xyxy)