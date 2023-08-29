import warnings
warnings.filterwarnings("ignore")

import os
import xml.etree.cElementTree as ET
from PIL import Image
import shutil
from tqdm import tqdm
import random
import copy

class create_new_mask_sample:

    def __init__(self):
        # self.new_mask_lable_classify_map={'GC_1':0, 'RC_1':1,'YC_1':2, 'GAF_1':3,'GAL_1':4,'GAR_1':5,'RAL_1':6}
        # self.new_mask_sample_path=r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample'
        self.new_mask_lable_classify_map = {'GC_1':0, 'RC_1':1,'YC_1':2, 'GAF_1':3,'GAL_1':4,'GAR_1':5,'RAF_1':6,'RAL_1':7,'RAR_1':8}
        self.new_mask_sample_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground_mask'
        self.total_new_mask_sample_list=[]
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
        self.mask_lable_classify_map={'car':0, 'bus':1, 'truck':2}
        self.image_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\VOC2007\JPEGImages'
        self.lable_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\VOC2007\Annotations'
        self.total_image_list=[]
        self.total_lable_list = []
        self.total_image_list = os.listdir(self.image_path)
        self.total_lable_list = os.listdir(self.lable_path)

        self.save_cut_image_area_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\cut_mark_area'
        self.save_coco_dataset_path = r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\traffic_light_detection_coco_buff"

    def GetAnnotBoxLoc(self,AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        ObjBndBoxSet = {'file_name':AnotPath.split('\\')[-1],'box_info':{}}
        for Object in ObjectSet:
            ObjName = Object.find('name').text

            if ObjName not in ObjBndBoxSet['box_info']:
                ObjBndBoxSet['box_info'][ObjName]=[]
            BndBox = Object.find('bndbox')
            x1 = int(BndBox.find('xmin').text)
            y1 = int(BndBox.find('ymin').text)
            x2 = int(BndBox.find('xmax').text)
            y2 = int(BndBox.find('ymax').text)
            ObjBndBoxSet['box_info'][ObjName].append([x1,y1,x2,y2])

            # print(ObjName,[x1,y1,x2,y2])
        return ObjBndBoxSet

    def convert_box(self,box, size=[1280, 720]):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[2]) / 2.0 - 1, (box[1] + box[3]) / 2.0 - 1, box[2] - box[0], box[3] - box[1]
        return [x * dw, y * dh, w * dw, h * dh]

    def creat_coco_lable(self,train_label_info,sample_num,sample_type='train',create_new_sample = False):

        if not create_new_sample: # 如果不增加自定义模拟mask图像时，保存的训练样本及对应标签
            train_cur = open(os.path.join(self.save_coco_dataset_path,sample_type,'labels', train_label_info['file_name'].replace('.xml', '.txt')), 'w')
            # print(train_label_info)
            for mask_classify in train_label_info['box_info']:
                for single_box in train_label_info['box_info'][mask_classify]:
                    conver_box_info = str(self.convert_box(single_box))
                    train_cur.writelines(str(self.mask_lable_classify_map[mask_classify])+' '+conver_box_info.replace(', ',' ')[1:-1])
                    train_cur.write('\n')
            train_cur.close()

            # 将对应的样本图拷贝到对应目录中
            sample_image_name = train_label_info['file_name'].replace('.xml', '.jpg')
            shutil.copy(os.path.join(self.image_path,sample_image_name ),
                        os.path.join(self.save_coco_dataset_path,sample_type,'images', sample_image_name))

        else: #当需要产生自定义模拟训练图像时：
            # 将对应的样本图拷贝到对应目录中
            sample_image_name = train_label_info['file_name'].replace('.xml', '.jpg')
            origin_image =  Image.open(os.path.join(self.image_path,sample_image_name))
            # shutil.copy(os.path.join(self.image_path, sample_image_name),
            #             os.path.join(self.save_coco_dataset_path, sample_type, 'images', sample_image_name))

            train_cur = open(os.path.join(self.save_coco_dataset_path, sample_type, 'labels',
                                          train_label_info['file_name'].replace('.xml', '_new%s.txt'%sample_num)), 'w')
            # print(train_label_info)
            for mask_classify in train_label_info['box_info']:
                for single_box in train_label_info['box_info'][mask_classify]:
                    try:
                        wight=single_box[2]-single_box[0]
                        hight=single_box[3]-single_box[1]
                        new_resize = (int(wight*0.9),int(hight*0.9)) # 获取模拟mask image的shape

                        classfiy_name,new_resize_mask = create_new_sample.create_random_shape_mask_img(new_resize) # 获取模拟mask 图像
                        # print(classfiy_name,new_resize_mask)
                        origin_image.paste(new_resize_mask, (int(single_box[0]+wight*0.1),int(single_box[1]+hight*0.1)))

                        conver_box_info = str(self.convert_box(single_box))
                        train_cur.writelines(
                            str(create_new_sample.new_mask_lable_classify_map[classfiy_name]) + ' ' + conver_box_info.replace(', ',' ')[
                                                                                     1:-1])
                        train_cur.write('\n')
                    except:
                        print(sample_image_name,': this image is error!!!')
            train_cur.close()
            origin_image.save(os.path.join(self.save_coco_dataset_path,sample_type,'images', sample_image_name.replace('.jpg','_new%s.jpg'%sample_num)))




    def cut_mark_area(self,LableBoxInfo):
        image_path = os.path.join(self.image_path,LableBoxInfo['file_name'].split('.')[0]+'.jpg')
        img = Image.open(image_path)

        for lable_classify in LableBoxInfo['box_info']:


            if lable_classify not in self.mask_lable_classify_map:
                self.mask_lable_classify_map['lable_classify']=max(self.mask_lable_classify_map.values())+1
            for single_box_id in range(len(LableBoxInfo['box_info'][lable_classify])):
                wide = LableBoxInfo['box_info'][lable_classify][single_box_id][2] - \
                       LableBoxInfo['box_info'][lable_classify][single_box_id][0]
                high = LableBoxInfo['box_info'][lable_classify][single_box_id][3] - \
                       LableBoxInfo['box_info'][lable_classify][single_box_id][1]

                if lable_classify == 'car':
                    # print(LableBoxInfo)

                    if high<1.5*wide:
                        pass
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                    elif high>=1.5*wide and high<2.5*wide:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] = LableBoxInfo['box_info'][lable_classify][single_box_id][1]+high/2 - 2
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        # roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                        #     0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                    else:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] = \
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high/3*2 - 2
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        # roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                        #     0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')

                if lable_classify == 'bus':
                    # print(LableBoxInfo)

                    if high<1.5*wide:
                        pass
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                    elif high>=1.5*wide and high<2.5*wide:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = LableBoxInfo['box_info'][lable_classify][single_box_id][1]+high/2 + 2
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        # roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                        #     0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                    else:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = \
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high/3 + 2
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        # roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                        #     0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')

                if lable_classify == 'truck':
                    # print(LableBoxInfo)

                    if high<1.5*wide:
                        pass
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                    elif high>=1.5*wide and high<2.5*wide:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = LableBoxInfo['box_info'][lable_classify][single_box_id][1]+high/2 + 2
                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        # roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                        #     0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                    else:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = \
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high/3*2 + 2

                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] = \
                            LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high / 3 - 2

                        # roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        # roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                        #     0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                # print(lable_classify,single_box_id)

                # if high<1.5*wide:
                #     roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                # elif high>=1.5*wide and high<2.5*wide:
                #     roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                # # 保存截取的图片
                # print(self.save_cut_image_area_path+LableBoxInfo['file_name'].split('.')[0]+'_'+lable_classify+'_'+str(single_box_id)+'.jpg')
                # roi.save(self.save_cut_image_area_path+'\\'+LableBoxInfo['file_name'].split('.')[0]+'_'+lable_classify+'_'+str(single_box_id)+'.jpg')

        return LableBoxInfo

    def read_sample_info(self,create_new_sample=False):
        print('hello word!')
        # total_image_list=[os.path.join(self.image_path,image_name) for image_name in os.listdir(self.image_path)]
        # total_lable_list = [os.path.join(self.lable_path,lable_name) for lable_name in os.listdir(self.lable_path)]


        if not create_new_mask_sample: # 当不是自定义构造样本时
            sample_num = 0
            # 读取标注文件信息：
            for lable_xml_index in tqdm(range(len(self.total_lable_list[:30]))):

                lable_xml = self.total_lable_list[lable_xml_index]
                # print(lable_xml)
                LableBoxInfo = self.GetAnnotBoxLoc(os.path.join(self.lable_path,lable_xml))
                # origin_image_path = os.path.join(self.total_image_list,lable_xml.split('.')[0]+'jpg')
                train_label_info = self.cut_mark_area(LableBoxInfo)

                if sample_num<20:
                    self.creat_coco_lable(train_label_info,'train',create_new_sample)
                else:
                    self.creat_coco_lable(train_label_info,'val',create_new_sample)
                sample_num += 1

        else: # 自定义构造样本时
            sample_num = 0
            my_create_sample_number = 45
            for lable_xml_index in tqdm(range(my_create_sample_number)):

                # lable_xml = self.total_lable_list[lable_xml_index]
                lable_xml =random.choice(self.total_lable_list)
                # print(lable_xml)
                LableBoxInfo = self.GetAnnotBoxLoc(os.path.join(self.lable_path, lable_xml))
                # origin_image_path = os.path.join(self.total_image_list,lable_xml.split('.')[0]+'jpg')
                train_label_info = self.cut_mark_area(LableBoxInfo)

                if sample_num < 42:
                    self.creat_coco_lable(train_label_info,sample_num, 'train', create_new_sample)
                else:
                    self.creat_coco_lable(train_label_info,sample_num, 'val', create_new_sample)
                sample_num += 1


create_new_sample = create_new_mask_sample()

a = Create_train_sample()
a.read_sample_info(create_new_sample)
