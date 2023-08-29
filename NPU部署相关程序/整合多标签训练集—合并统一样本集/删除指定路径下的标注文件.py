#coding=utf-8

    # 注：由于在制作样本时，有的样本集中含有其他类的标注目标。但数量又较少，为了方便，我们直接将含有其他类的少量样本标签文件删除。这样更高效且准确


import os

delet_dest_label_path = r"D:\PycharmProgram\ZZ_Total_DeepLearning_Model\yolov7-face-main\detect_DVR_Vimicro_Image\shanghai_DVR_face_detect\labels"
origin_label_path = r"D:\PycharmProgram\ZZ_Total_DeepLearning_Model\yolov7-face-main\detect_DVR_Vimicro_Image\shanghai_DVR_face_detect\delect_path"

all_label_name_list = [txt_file_name for txt_file_name in os.listdir(origin_label_path)]  #锛堝浘鐗囨枃浠跺す锛?# print(all_image_name_list)
for label_name in all_label_name_list:
    delect_label_path = os.path.join(delet_dest_label_path,label_name)
    os.remove(delect_label_path)
