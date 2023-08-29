
# -*-coding: GBK -*-
# 以下为Linux上的编码格式
import os
from shutil import copy2


class create_car_person_plate_sample:

    def __init__(self,
                 origin_detect_labels_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify\labels",
                 origin_images_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify",
                 save_path = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\person_car_dataset",
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
        # 注：当前只检测人、汽车两大类，并且标注转换时进行了归并```
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

    # 做一个字典函数，统一将一个标注文件的所有信息一次性读取出来。避免按行读取时的，各种操作的干扰。
    def check_single_label_file_info(self,label_cur):
        total_label_info_dict = {'classify_num':[],'point_box':[]}
        all_label_file_info = label_cur.readlines()
        for row_label_info in all_label_file_info:
            classify_num,point_box=row_label_info.split(' ',1)
            if self.check_image_with_wh_pixel_rate(point_box): #如果能通过pixel筛选，则写入数据
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

            # 拷贝图像到训练数据集images目录下：在此处拷贝图像的目的是增加不含目标的样本，也可在确定包含目标是拷贝图片（if classify_num in check_dest_classify_set:）
            if self.Bool_operate_set_dict['Is_copy_all_image']: #只有当需要时，才拷贝样本图像
                self.copy_image_to_dest_folder(single_label_name.replace('.txt','.jpg'))
            total_label_info_dict=self.check_single_label_file_info(now_lable_cur)



            if len(total_label_info_dict['point_box'])>0:
                Is_exist_target = False  # 是否存在目标图
                for classify_num in self.check_target_classify:
                    if classify_num in total_label_info_dict['classify_num']:
                        Is_exist_target = True
                        # 拷贝图像到训练数据集images目录下：在此处拷贝图像的目的是增加不含目标的样本，也可在确定包含目标是拷贝图片（if classify_num in check_dest_classify_set:）
                        if self.Bool_operate_set_dict['Is_only_target_image']:  # 只有当需要时，才拷贝样本图像
                            self.copy_image_to_dest_folder(single_label_name.replace('.txt', '.jpg'))
                        break

                if Is_exist_target:  # 当目标存在时。

                    get_images_num += 1
                    write_label_cur = open(os.path.join(self.save_path, "train", 'labels', single_label_name), 'w')
                    for index, point_box in enumerate(total_label_info_dict['point_box']):
                        # 此处可以增加像检测框，像素筛选功能
                        if self.Bool_operate_set_dict['Is_check_pixle']:

                            check_pixle_flag = self.check_image_with_wh_pixel_rate(point_box)
                            if check_pixle_flag:  # 当像素点符合W 比例限制时。
                                if total_label_info_dict['classify_num'][index] in ['0']:  # preson类别
                                    write_label_cur.write('0' + ' ' + point_box)


                        else:
                            if total_label_info_dict['classify_num'][index] in ['0']:  # preson类别
                                write_label_cur.write('0' + ' ' + point_box)

                    write_label_cur.close()
                if get_images_num > max_images_number:
                    break



if __name__ == "__main__":
    # origin_detect_labels_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\face_detect\labels"
    # origin_images_path = r"D:\Git_WareHouse\yolov5-master\test_folder\test_face"
    # save_path = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\person_face_dataset"
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
     # 是否同时拷贝样本图片到训练文件夹中

    create_sample = create_car_person_plate_sample(origin_detect_labels_path,origin_images_path,save_path,check_target_classify,Bool_operate_set_dict)
    create_sample.check_target_labels()
    # print(create_sample.all_images_name_list)