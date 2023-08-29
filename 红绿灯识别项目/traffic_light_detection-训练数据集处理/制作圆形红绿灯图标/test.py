import warnings
warnings.filterwarnings("ignore")

import os
import xml.etree.cElementTree as ET
from PIL import Image
import shutil
from tqdm import tqdm
import random
import copy


# 对 mask标记图：图片背景透明化
def transPNG(srcImageName):
    img = Image.open(srcImageName)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] > 220 and item[1] > 220 and item[2] > 220:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

# 图片融合
def mix(img1,img2,coordinator):
    im = img1
    mark = img2
    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    layer.paste(mark, coordinator)
    out = Image.composite(layer, im, layer)
    return out


new_mask_sample_path=r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground_mask\GAF_1\image.0040g.png'
image = transPNG(new_mask_sample_path)

back_image = Image.open(r'E:\paddle_训练数据集\ICDAR2019-LSVT\voc_dataset\JPEGImages\gt_29970.jpg')

back_image = mix(back_image,image, (0, 0))

# back_image.paste(image, [0,0])
back_image.show()