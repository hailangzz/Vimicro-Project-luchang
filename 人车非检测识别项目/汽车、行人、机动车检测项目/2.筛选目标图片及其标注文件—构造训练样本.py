# coding=utf-8
import os
from shutil import copy2


class create_car_person_plate_sample:

    def __init__(self,
                 origin_detect_labels_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify\labels",
                 origin_images_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify",
                 save_path = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\person_car_dataset",
                 check_target_classify_map = {'0': 'person', '1': 'rider', '2': 'car', '3': 'bus', '4': 'truck'},
                 Is_copy_image = False
                 ):

        self.origin_detect_labels_path = origin_detect_labels_path
        self.origin_images_path = origin_images_path
        self.save_path = save_path

        self.all_images_name_list=[]
        self.get_all_images_name()
        self.init_train_folder()
        # 注：当前只检测人、汽车两大类，并且标注转换时进行了归并```
        #self.check_target_classify_map={'person':'0','rider':'0','car':'1','bus':'1','truck':'1',}
        self.check_target_classify_map = check_target_classify_map

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


    def check_target_labels(self):

        all_labels_name_list = os.listdir(self.origin_detect_labels_path)
        for single_label_name in all_labels_name_list:

            now_lable_cur = open(os.path.join(self.origin_detect_labels_path,single_label_name),'r')
            write_label_cur = open(os.path.join(self.save_path, "train", 'labels', single_label_name), 'w')
            total_label_info = now_lable_cur.readlines()

            # 拷贝图像到训练数据集images目录下：
            if Is_copy_image: #只有当需要时，才拷贝样本图像
                self.copy_image_to_dest_folder(single_label_name.replace('.txt','.jpg'))
            for row_label_info in total_label_info:
                classify_num,point_box=row_label_info.split(' ',1)
                if classify_num in self.check_target_classify_map:
                    #当目标标注存在时，写入样本标注信息:(注意类别归并)
                    # print(classify_num,self.check_target_classify_map[classify_num]['map_value'])
                    write_label_cur.write(self.check_target_classify_map[classify_num]['map_value']+' '+point_box)

            write_label_cur.close()
            now_lable_cur.close()



if __name__ == "__main__":
    # origin_detect_labels_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify\labels"
    # origin_images_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify"
    # save_path = r"D:\迅雷下载\AI数据集汇总\人车非数据及项目\person_car_dataset"

    origin_detect_labels_path = r"/home/zhangzhuo/git_workspace/yolov5/detect_DVR_Vimicro_Image/shanghai_select/labels"
    origin_images_path = r"/home/zhangzhuo/total_dataset/Vimicro_DVR_ShangHai/SH_1080p_jpg"
    save_path = r"/home/zhangzhuo/git_workspace/yolov5/datasets/person_car_dataset"
    check_target_classify_map = {'0':{'map_value':'0','name':'person'} ,
                                 '1':{'map_value':'0','name': 'rider'},
                                 '2':{'map_value':'1','name':  'car'},
                                 '3':{'map_value':'1','name':  'bus'},
                                 '4':{'map_value':'1','name':  'truck'}
                                 }
    Is_copy_image=True # 是否同时拷贝样本图片到训练文件夹中

    create_sample = create_car_person_plate_sample(origin_detect_labels_path,origin_images_path,save_path,check_target_classify_map,Is_copy_image)
    create_sample.check_target_labels()
    # print(create_sample.all_images_name_list)