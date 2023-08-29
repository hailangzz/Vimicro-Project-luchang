import os
from shutil import copy2

class combin_mult_train_datasets:
    def __init__(self,left_datasets_path,right_datasets_path,combin_save_datasets_path,part_datasets_search_map_info):
        self.combin_train_dataset_dict={} # 用来存储，整合好的训练样本字典信息。（其key为图片名称；value为coco标注信息数组，{'image_path':"","point_box":[]}）
        self.part_datasets_search_map_info = {"left_datasets_search_map":{"0":"1","1":"11"},
                                          "right_datasets_search_map":{"0":"10"}
                                          }
        self.left_datasets_path=left_datasets_path
        self.right_datasets_path = right_datasets_path
        self.combin_save_datasets_path = combin_save_datasets_path
        self.part_datasets_search_map_info = part_datasets_search_map_info

        self.calculate_combin_train_dataset_dict()
        self.final_create_combin_datasets()
        pass

    def calculate_combin_train_dataset_dict(self):
        self.read_part_datasets_info(self.left_datasets_path,
                                     self.part_datasets_search_map_info['left_datasets_search_map'])
        self.read_part_datasets_info(self.right_datasets_path,
                                     self.part_datasets_search_map_info['right_datasets_search_map'])


    def read_part_datasets_info(self,datasets_path,search_and_map_classify={"0":"10","1":"11"}):
        datasets_images_path = os.path.join(datasets_path, 'train', 'images')
        datasets_labels_path = os.path.join(datasets_path,'train','labels')
        total_labels_name_list = os.listdir(datasets_labels_path)

        for single_labal_name in total_labels_name_list:
            single_image_name = single_labal_name.replace('.txt','.jpg')
            single_label_cur = open(os.path.join(datasets_labels_path,single_labal_name))
            labels_info = single_label_cur.readlines()
            for row_label_info in labels_info:
                classify_num,point_box = row_label_info.split(' ',1)
                if classify_num in search_and_map_classify:

                    if single_image_name not in self.combin_train_dataset_dict:
                        self.combin_train_dataset_dict[single_image_name]={'image_path':"","point_box":[]}
                        self.combin_train_dataset_dict[single_image_name]['image_path'] = os.path.join(datasets_images_path,single_image_name)
                        self.combin_train_dataset_dict[single_image_name]["point_box"].append(search_and_map_classify[classify_num]+' '+point_box)
                    else:
                        self.combin_train_dataset_dict[single_image_name]["point_box"].append(
                            search_and_map_classify[classify_num] + ' ' + point_box)
            single_label_cur.close()


        pass

    def final_create_combin_datasets(self):
        save_combin_train_path = os.path.join(self.combin_save_datasets_path, "train")
        images_save_path = os.path.join(save_combin_train_path,'images')
        labels_save_path = os.path.join(save_combin_train_path, 'labels')
        if "train" not in os.listdir(self.combin_save_datasets_path):
            os.mkdir(save_combin_train_path)
            os.mkdir(images_save_path)
            os.mkdir(labels_save_path)
        # 拷贝图像、写入标签信息等：
        for image_name in self.combin_train_dataset_dict:
            origin_images_path = self.combin_train_dataset_dict[image_name]['image_path']

            copy2(origin_images_path,images_save_path)
            save_label_file_cur = open(os.path.join(labels_save_path,image_name.replace('.jpg','.txt')),'w')
            for row_label_info in self.combin_train_dataset_dict[image_name]['point_box']:
                save_label_file_cur.write(row_label_info)
            save_label_file_cur.close()

left_datasets_path=r"D:\中星微人工智能工作\ykang_plate_det_rec\ykang_plate_det_rec\PersonFaceDatasets_buffer\PersonFaceDatasets_buffer"
right_datasets_path=r"D:\中星微人工智能工作\ykang_plate_det_rec\ykang_plate_det_rec\part_wide_face_val"
combin_save_datasets_path=r"D:\中星微人工智能工作\ykang_plate_det_rec\ykang_plate_det_rec\test_combin_datasets"
part_datasets_search_map_info={"left_datasets_search_map":{"0":"1","1":"11"},
                               "right_datasets_search_map":{"0":"10"}
                               }

create_datasets = combin_mult_train_datasets(left_datasets_path,right_datasets_path,combin_save_datasets_path,part_datasets_search_map_info)
print(len(create_datasets.combin_train_dataset_dict))