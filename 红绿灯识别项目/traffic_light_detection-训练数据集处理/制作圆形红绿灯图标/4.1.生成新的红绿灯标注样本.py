import warnings
warnings.filterwarnings("ignore")
from PIL import ImageDraw,Image, ImageFilter
import os
import xml.etree.cElementTree as ET
from PIL import Image
import shutil
from tqdm import tqdm
import random
import copy

class create_new_mask_sample:

    def __init__(self):
        self.new_mask_lable_classify_map={'GC_1':0, 'RC_1':1,'YC_1':2, 'GAF_1':3,'GAL_1':4,'GAR_1':5,'RAF_1':6,'RAL_1':7,'RAR_1':8}
        self.total_new_mask_sample_list=[]
        self.new_mask_sample_path=r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground_mask'
        print('hello world!',self.new_mask_sample_path)
        self.get_total_mask_sample()
        # self.create_new_coco_sample()
        pass

    # 读取全部的new mask 信息列表：
    def get_total_mask_sample(self,):
        single_mask_info={'classfiy_name':'','image_path':''}
        total_classify_name = os.listdir(self.new_mask_sample_path)
        for classify_name in total_classify_name:
            classify_image_path = os.path.join(self.new_mask_sample_path,classify_name)
            # print(classify_image_path)
            for mask_image_name in os.listdir(classify_image_path):
                single_mask_info['classfiy_name'] = classify_name
                single_mask_info['image_path'] = os.path.join(classify_image_path,mask_image_name)
                self.total_new_mask_sample_list.append(copy.deepcopy(single_mask_info))

    # 对新的标注区图进行resize操作
    def create_random_shape_mask_img(self,resize=(24,24)):
        choice_mask_info = random.choice(self.total_new_mask_sample_list)
        mask_img = Image.open(choice_mask_info['image_path'])
        # resize_mask = mask_img.resize(resize, Image.BILINEAR) # 双现行插值
        resize_mask = mask_img.resize(resize, Image.ANTIALIAS) # 高质量resize

        return choice_mask_info['classfiy_name'],resize_mask

        # resize_mask.save(os.path.join(outdir, os.path.basename(jpgfile)))

    # 将新模拟的标注图粘贴到原始图像区域上
    def create_new_coco_sample(self,origin_image_path='./00a0f008-3c67908e.jpg',past_point=(20, 20)):
        origin_image = Image.open(origin_image_path)
        resize_mask = self.create_random_shape_mask_img()
        origin_image.paste(resize_mask, past_point)
        origin_image.save('resize.jpg')



class Create_train_sample:
    def __init__(self):

        self.background_image_path = r'E:\paddle_训练数据集\ICDAR2019-LSVT\voc_dataset\JPEGImages'
        self.save_create_image_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\create_traffic_light_sample\images'
        self.save_create_label_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\create_traffic_light_sample\labels'
        self.background_image_list = [os.path.join(self.background_image_path,image_name) for image_name in os.listdir(self.background_image_path)]

    def get_paste_point(self,background_img_size,mask_img_size):#获取随机的目标图片粘贴位置
        paste_point=[0,0]
        paste_point[0] = random.randint(0,background_img_size[0]-mask_img_size[0])
        paste_point[1] = random.randint(0, background_img_size[1] - mask_img_size[1])

        return paste_point

    # 对 mask标记图：图片背景透明化
    def transPNG(self,srcImageName):
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

    # 图片融合：融合背景和标注图 back_image:为图像矩阵，mask_image:为背景透明化的图像矩阵
    def mix(self,back_image, mask_image, past_point):

        im = back_image
        mark = mask_image
        layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
        layer.paste(mark, past_point)
        out = Image.composite(layer, im, layer)
        return out

    def convert_box(self,size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return [x * dw, y * dh, w * dw, h * dh]

    def get_image_target_info(self,background_img_size, qr_code_img_size, paste_point):
        target_box_info = {'box_point': [0, 0, 0, 0], 'coco_point': [0, 0, 0, 0]}
        target_box_info['box_point'][0] = paste_point[0]
        target_box_info['box_point'][1] = paste_point[1]
        target_box_info['box_point'][2] = paste_point[0] + qr_code_img_size[0]
        target_box_info['box_point'][3] = paste_point[1] + qr_code_img_size[1]

        target_box_info['coco_point'] = self.convert_box(background_img_size,
                                                    [target_box_info['box_point'][0], target_box_info['box_point'][2],
                                                     target_box_info['box_point'][1], target_box_info['box_point'][3]])
        return target_box_info

    def create_train_images(self,mask_image_object,create_image_number=100):

        for index in tqdm(range(create_image_number)):

            try:
                lable_file_path = os.path.join(self.save_create_label_path,str(index)+'.txt')
                lable_cur = open(lable_file_path,'w')
                # 在背景主图上绘制随机数量的目标图:
                target_num = random.randint(1, 5)
                background_img = Image.open(random.choice(self.background_image_list)).convert("RGBA")  # 背景图
                for i in range(target_num):
                    mask_image_resize = random.randint(10,50)
                    mask_choice_info = random.choice(mask_image_object.total_new_mask_sample_list)
                    # mask_image = Image.open(random.choice(mask_image_object.total_new_mask_sample_list)['image_path']).convert("RGBA")  # 背景图
                    mask_image = self.transPNG(mask_choice_info['image_path'])
                    mask_image = mask_image.resize((mask_image_resize,mask_image_resize), Image.ANTIALIAS) #mask图片改变大小
                    mask_image = mask_image.filter(
                        ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1)))  # 图像增加高斯模糊···
                    paste_point = self.get_paste_point(background_img.size, mask_image.size)  # 随机获取目标图像粘贴位置point信息···
                    # 以下为mask与背景图像的融合
                    # print(paste_point)
                    layer = Image.new('RGBA', background_img.size, (0, 0, 0, 0))
                    layer.paste(mask_image, paste_point)
                    # result_image = self.mix(background_img,mask_image,paste_point)
                    background_img = Image.composite(layer, background_img, layer)
                    # background_img.paste(mask_img, paste_point)  # 粘贴目标图到背景图像上···
                    # 以下记录标注标签的位置信息
                    target_box_info = self.get_image_target_info(background_img.size, mask_image.size, paste_point)
                    mask_classify =mask_choice_info['classfiy_name']
                    classsify_flag = str(mask_image_object.new_mask_lable_classify_map[mask_classify]) # 实际的mask标注对应的类别
                    lable_cur.write(classsify_flag + ' ' + str(target_box_info['coco_point'][0]) + ' ' + str(
                        target_box_info['coco_point'][1]) + ' ' + str(target_box_info['coco_point'][2]) + ' ' + str(
                        target_box_info['coco_point'][3]) + '\n')
                save_path = os.path.join(self.save_create_image_path,str(index)+'.png')
                background_img.save(save_path)
                lable_cur.close()
            except Exception as e:
                print(e)
                continue



mask_image_object = create_new_mask_sample()

a = Create_train_sample()
a.create_train_images(mask_image_object,create_image_number=50000)
# print(create_new_sample.total_new_mask_sample_list[0])
# a.read_sample_info(create_new_sample)
