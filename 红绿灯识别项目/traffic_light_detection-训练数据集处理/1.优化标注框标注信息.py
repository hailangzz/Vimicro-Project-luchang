import os
import xml.etree.cElementTree as ET
from PIL import Image
import shutil
from tqdm import tqdm

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
        self.save_coco_dataset_path = r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\traffic_light_detection_coco"

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

    def creat_coco_lable(self,train_label_info,sample_type='train'):

        # if sample_type=='train':
        train_cur = open(os.path.join(self.save_coco_dataset_path,sample_type,'labels', train_label_info['file_name'].replace('.xml', '.txt')), 'w')
        # print(train_label_info)
        for mask_classify in train_label_info['box_info']:
            for single_box in train_label_info['box_info'][mask_classify]:
                conver_box_info = str(self.convert_box(single_box))
                train_cur.writelines(str(self.mask_lable_classify_map[mask_classify])+' '+conver_box_info.replace(',',' ')[1:-1])
                train_cur.write('\n')
        train_cur.close()

        # 将对应的样本图拷贝到对应目录中
        sample_image_name = train_label_info['file_name'].replace('.xml', '.jpg')
        shutil.copy(os.path.join(self.image_path,sample_image_name ),
                    os.path.join(self.save_coco_dataset_path,sample_type,'images', sample_image_name))


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
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                    elif high>=1.5*wide and high<2.5*wide:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] = LableBoxInfo['box_info'][lable_classify][single_box_id][1]+high/2 - 2
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                            0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                    else:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] = \
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high/3*2 - 2
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                            0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')

                if lable_classify == 'bus':
                    # print(LableBoxInfo)

                    if high<1.5*wide:
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                    elif high>=1.5*wide and high<2.5*wide:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = LableBoxInfo['box_info'][lable_classify][single_box_id][1]+high/2 + 2
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                            0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                    else:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = \
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high/3 + 2
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                            0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')

                if lable_classify == 'truck':
                    # print(LableBoxInfo)

                    if high<1.5*wide:
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                    elif high>=1.5*wide and high<2.5*wide:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = LableBoxInfo['box_info'][lable_classify][single_box_id][1]+high/2 + 2
                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                            0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                    else:
                        LableBoxInfo['box_info'][lable_classify][single_box_id][3] = \
                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high/3*2 + 2

                        LableBoxInfo['box_info'][lable_classify][single_box_id][1] = \
                            LableBoxInfo['box_info'][lable_classify][single_box_id][1] + high / 3 - 2

                        roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                        roi.save(self.save_cut_image_area_path + '\\' + LableBoxInfo['file_name'].split('.')[
                            0] + '_' + lable_classify + '_' + str(single_box_id) + '.jpg')
                # print(lable_classify,single_box_id)

                # if high<1.5*wide:
                #     roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                # elif high>=1.5*wide and high<2.5*wide:
                #     roi = img.crop(LableBoxInfo['box_info'][lable_classify][single_box_id])
                # # 保存截取的图片
                # print(self.save_cut_image_area_path+LableBoxInfo['file_name'].split('.')[0]+'_'+lable_classify+'_'+str(single_box_id)+'.jpg')
                # roi.save(self.save_cut_image_area_path+'\\'+LableBoxInfo['file_name'].split('.')[0]+'_'+lable_classify+'_'+str(single_box_id)+'.jpg')

        return LableBoxInfo

    def read_sample_info(self,):
        sample_num = 0
        print('hello word!')
        # total_image_list=[os.path.join(self.image_path,image_name) for image_name in os.listdir(self.image_path)]
        # total_lable_list = [os.path.join(self.lable_path,lable_name) for lable_name in os.listdir(self.lable_path)]


        # 读取标注文件信息：
        for lable_xml_index in tqdm(range(len(self.total_lable_list[:]))):

            lable_xml = self.total_lable_list[lable_xml_index]
            # print(lable_xml)
            LableBoxInfo = self.GetAnnotBoxLoc(os.path.join(self.lable_path,lable_xml))
            # origin_image_path = os.path.join(self.total_image_list,lable_xml.split('.')[0]+'jpg')
            train_label_info = self.cut_mark_area(LableBoxInfo)

            if sample_num<40000:
                self.creat_coco_lable(train_label_info,'train')
            else:
                self.creat_coco_lable(train_label_info,'val')

            sample_num += 1



a = Create_train_sample()
a.read_sample_info()