
# -*-coding: GBK -*-
# ����ΪLinux�ϵı����ʽ
import os
from shutil import copy2


class create_car_person_plate_sample:

    def __init__(self,
                 origin_detect_labels_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify\labels",
                 origin_images_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify",
                 save_path = r"D:\Ѹ������\AI���ݼ�����\�˳������ݼ���Ŀ\person_car_dataset",
                 check_target_classify = {'0': 'person', '1': 'rider', '2': 'car', '3': 'bus', '4': 'truck'},
                 Bool_operate_set_dict = {"Is_copy_all_image":False,
                                          "Is_only_target_image":False,
                                          "Is_check_pixle": False,}
                 ):

        self.origin_detect_labels_path = origin_detect_labels_path
        self.origin_images_path = origin_images_path
        self.save_path = save_path

        self.all_images_name_list=[]
        self.get_all_images_name()
        self.init_train_folder()
        # ע����ǰֻ����ˡ����������࣬���ұ�עת��ʱ�����˹鲢```
        #self.check_target_classify={'person':'0','rider':'0','car':'1','bus':'1','truck':'1',}
        self.check_target_classify = check_target_classify
        self.Bool_operate_set_dict=Bool_operate_set_dict

    def init_train_folder(self):
        if "train" not in os.listdir(self.save_path):
            os.mkdir(os.path.join(self.save_path,"train"))
            os.mkdir(os.path.join(self.save_path, "train",'images'))
            os.mkdir(os.path.join(self.save_path, "train",'labels'))

    def get_all_images_name(self):
        total_name_list = os.listdir(self.origin_images_path)
        for single_name in total_name_list:
            if "jpg" in single_name or 'png' in single_name:
                self.all_images_name_list.append(single_name)

    def copy_image_to_dest_folder(self,image_name):
        fileName = os.path.join(self.origin_images_path,image_name)
        dst_path = os.path.join(self.save_path,'train','images')
        copy2(fileName, dst_path)

    def check_image_with_wh_pixel_rate(self,coco_box_info,w_rate=0.1):
        coco_box_W_rate = float(coco_box_info.split(' ')[2])
        if coco_box_W_rate>=w_rate:
            return True
        else:
            return False

    # ��һ���ֵ亯����ͳһ��һ����ע�ļ���������Ϣһ���Զ�ȡ���������ⰴ�ж�ȡʱ�ģ����ֲ����ĸ��š�
    def check_single_label_file_info(self,label_cur):
        total_label_info_dict = {'classify_num':[],'point_box':[]}
        all_label_file_info = label_cur.readlines()
        for row_label_info in all_label_file_info:
            classify_num,point_box=row_label_info.split(' ',1)
            if self.check_image_with_wh_pixel_rate(point_box): #�����ͨ��pixelɸѡ����д������
                total_label_info_dict['classify_num'].append(classify_num)
                total_label_info_dict['point_box'].append(point_box)
        label_cur.close()
        return total_label_info_dict


    def check_target_labels(self,max_images_number=200):
        get_images_num=0

        check_dest_classify_set = set(self.check_target_classify)
        all_labels_name_list = os.listdir(self.origin_detect_labels_path)
        for single_label_name in all_labels_name_list:

            now_lable_cur = open(os.path.join(self.origin_detect_labels_path,single_label_name),'r')

            # ����ͼ��ѵ�����ݼ�imagesĿ¼�£��ڴ˴�����ͼ���Ŀ�������Ӳ���Ŀ���������Ҳ����ȷ������Ŀ���ǿ���ͼƬ��if classify_num in check_dest_classify_set:��
            if self.Bool_operate_set_dict['Is_copy_all_image']: #ֻ�е���Ҫʱ���ſ�������ͼ��
                self.copy_image_to_dest_folder(single_label_name.replace('.txt','.jpg'))
            total_label_info_dict=self.check_single_label_file_info(now_lable_cur)



            if len(total_label_info_dict['point_box'])>0:
                Is_exist_target = False  # �Ƿ����Ŀ��ͼ
                for classify_num in self.check_target_classify:
                    if classify_num in total_label_info_dict['classify_num']:
                        Is_exist_target = True
                        # ����ͼ��ѵ�����ݼ�imagesĿ¼�£��ڴ˴�����ͼ���Ŀ�������Ӳ���Ŀ���������Ҳ����ȷ������Ŀ���ǿ���ͼƬ��if classify_num in check_dest_classify_set:��
                        if self.Bool_operate_set_dict['Is_only_target_image']:  # ֻ�е���Ҫʱ���ſ�������ͼ��
                            self.copy_image_to_dest_folder(single_label_name.replace('.txt', '.jpg'))
                        break

                if Is_exist_target:  # ��Ŀ�����ʱ��

                    get_images_num += 1
                    write_label_cur = open(os.path.join(self.save_path, "train", 'labels', single_label_name), 'w')
                    for index, point_box in enumerate(total_label_info_dict['point_box']):
                        # �˴������������������ɸѡ����
                        if self.Bool_operate_set_dict['Is_check_pixle']:

                            check_pixle_flag = self.check_image_with_wh_pixel_rate(point_box)
                            if check_pixle_flag:  # �����ص����W ��������ʱ��
                                if total_label_info_dict['classify_num'][index] in ['0']:  # preson���
                                    write_label_cur.write('0' + ' ' + point_box)


                        else:
                            if total_label_info_dict['classify_num'][index] in ['0']:  # preson���
                                write_label_cur.write('0' + ' ' + point_box)

                    write_label_cur.close()
                if get_images_num > max_images_number:
                    break



if __name__ == "__main__":
    # origin_detect_labels_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\face_detect\labels"
    # origin_images_path = r"D:\Git_WareHouse\yolov5-master\test_folder\test_face"
    # save_path = r"D:\Ѹ������\AI���ݼ�����\�˳������ݼ���Ŀ\person_face_dataset"
    # check_target_classify = {'0': 'person'}
    # Bool_operate_set_dict={"Is_copy_all_image":False,
    #                        "Is_only_target_image":True,
    #                        "Is_check_pixle": True,
    #                        }

    origin_detect_labels_path = r"/home/zhangzhuo/git_workspace/yolov5/detect_DVR_Vimicro_Image/shenzhen_select/labels"
    origin_images_path = r"/home/zhangzhuo/total_dataset/Vimicro_Image_dataset"
    save_path = r"/home/zhangzhuo/total_dataset/person_target_dataset"
    check_target_classify = {'0': 'person'}
    Bool_operate_set_dict = {"Is_copy_all_image": False,
                             "Is_only_target_image": True,
                             "Is_check_pixle": True,
                             }
     # �Ƿ�ͬʱ��������ͼƬ��ѵ���ļ�����

    create_sample = create_car_person_plate_sample(origin_detect_labels_path,origin_images_path,save_path,check_target_classify,Bool_operate_set_dict)
    create_sample.check_target_labels()
    # print(create_sample.all_images_name_list)