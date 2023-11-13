import os
from shutil import copy2
# 原始样本集：F:\DMS交付\dms_datasets\dms_train_data 、原始样本标注：names: ['face', 'smoke', 'eyes', 'sunglasses', 'mouth', 'phone']  # class names
# 当前筛选，保留原始样本中的：'smoke'、'sunglasses'、'phone'

origin_DMS_sample_path = r"F:\DMS交付\dms_datasets\dms_train_data"
sample_type_dict={'train_type':['train','valid'],"file_type":['images','labels']}
save_check_sample_path = r"F:\AiTotalDatabase\DMS_train_sample"

def check_class_info(origin_DMS_sample_path,sample_type_dict,save_check_sample_path,check_class_id_list=['1','3','5']):
    for train_type in sample_type_dict['train_type']:
        if train_type == "train":
            second_sample_path = os.path.join(origin_DMS_sample_path,train_type)
            save_check_second_path = os.path.join(save_check_sample_path,train_type)
            save_check_second_path_images = os.path.join(save_check_second_path,"images")
            save_check_second_path_labels = os.path.join(save_check_second_path, "labels")

            if not os.path.exists(save_check_second_path):
                os.mkdir(save_check_second_path)
                os.mkdir(save_check_second_path_images)
                os.mkdir(save_check_second_path_labels)
            for file_type in sample_type_dict['file_type']:
                if file_type == "labels":
                    file_origin_path = os.path.join(second_sample_path,file_type)
                    save_check_third_path = os.path.join(save_check_second_path,file_type)


                    total_file_name_list = os.listdir(file_origin_path)
                    for file_name in total_file_name_list:
                        copy_origin_image_name=""
                        save_image_name = ""
                        save_label_name = ""
                        label_file_list=[]

                        all_row_file_info = open(os.path.join(file_origin_path,file_name),'r').readlines()
                        for single_row in all_row_file_info:

                            if single_row[0] in check_class_id_list:
                                if single_row[0] == '1':
                                    single_row ='0'+single_row[1:]
                                elif single_row[0] == '3':
                                    single_row ='1'+single_row[1:]
                                elif single_row[0] == '5':
                                    single_row ='2'+single_row[1:]

                                label_file_list.append(single_row)

                        if len(label_file_list)>0:
                            copy_origin_image_name = os.path.join(second_sample_path,'images',file_name.replace('.txt','.jpg'))
                            save_image_name = os.path.join(save_check_second_path_images,file_name.replace('.txt','.jpg'))
                            save_label_name = os.path.join(save_check_second_path_labels,file_name)
                            #copy2(copy_origin_image_name, save_image_name)

                            save_label_cur = open(save_label_name,'w')
                            for write_label in label_file_list:
                                save_label_cur.write(write_label)
                            save_label_cur.close()

check_class_info(origin_DMS_sample_path,sample_type_dict,save_check_sample_path,check_class_id_list=['1','3','5'])