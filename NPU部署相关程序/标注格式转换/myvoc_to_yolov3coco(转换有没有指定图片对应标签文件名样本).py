import os
import xml.etree.ElementTree as ET
import shutil


train_object_id_map={}

def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def cover_annotations_to_yolov3_labels(use_type_info_dict,second_dir):

    label_name_map_dict={'green':'0','red':'1','yellow':'2','none':'5'}

    save_images_dir = os.path.join(second_dir,'images')
    save_labels_dir = os.path.join(second_dir,'labels')
    if not os.path.exists(save_images_dir):
        os.mkdir(save_images_dir)
        os.mkdir(save_labels_dir)

    for single_annotations_path_id in range(len(use_type_info_dict['annotations_path_list'])):
        #写入存储标签的句柄···
        # print(save_images_dir)
        shutil.copy(use_type_info_dict['images_path_list'][single_annotations_path_id], save_images_dir)
        write_label_cur=open(os.path.join(save_labels_dir,use_type_info_dict['images_name_list'][single_annotations_path_id]+'.txt'),'w')
        # print(write_label_cur)
        # print(use_type_info_dict['annotations_path_list'][single_annotations_path_id])

        tree = ET.parse(use_type_info_dict['annotations_path_list'][single_annotations_path_id])
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        # print(w,h)

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            # if int(obj.find('difficult').text) == 0: #当标注文件有 difficult项时，其用来判断标注的可识别程度

            # 注：以下为自增型标注类别定义；（自己也可指定标注类别关系map）
            # if len(train_object_id_map)==0:
            #     train_object_id_map[class_name]=0
            # else:
            #     if class_name not in train_object_id_map:
            #         train_object_id_map[class_name]=max(train_object_id_map.values())+1

            train_object_id_map[class_name]=label_name_map_dict[class_name] #自定义标注类别映射


            xmlbox = obj.find('bndbox')
            x_d, y_d, w_d, h_d = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            # print(train_object_id_map[class_name],x_d, y_d, w_d, h_d)
            # class_name_id = yaml['names'].index(class_name)  # class id
            # out_file.write(" ".join([str(a) for a in (class_name_id, *bb)]) + '\n')
            write_label_cur.write(str(train_object_id_map[class_name])+' '+str(x_d)+' '+str(y_d)+' '+str(w_d)+' '+str(h_d)+'\n')

        write_label_cur.close()

def get_voc_data(voc_data_dir,save_yolov3_coco_direct_name):

    use_type_info_dict={'images_path_list':[],'annotations_path_list':[],'images_name_list':[]}

    second_dir = os.path.join(voc_data_dir,save_yolov3_coco_direct_name) #次级存储txt标注文件的存储路径

    if not os.path.exists(second_dir):
        os.mkdir(second_dir)

    # 接下来读取标注文件及对应的图像文件路径：
    image_folder_name = 'daytime'
    xml_label_folder_name = 'xml_'
    total_voc_xml_namelist = os.listdir(os.path.join(voc_data_dir,xml_label_folder_name)) #xml_ 为xml标注文件夹名称
    # print(total_voc_xml_namelist)

    for xml_file_name in total_voc_xml_namelist:
        use_type_info_dict['images_path_list'].append(
            os.path.join(voc_data_dir,image_folder_name,xml_file_name.replace('.xml','.jpg')))  # 图片样本路径列表
        use_type_info_dict['annotations_path_list'].append(
            os.path.join(voc_data_dir, xml_label_folder_name, xml_file_name))  # xml标注路径列表
        use_type_info_dict['images_name_list'].append(xml_file_name.split('.')[0])  # 图像样本名称列表road839
    # print(use_type_info_dict)

    cover_annotations_to_yolov3_labels(use_type_info_dict,second_dir)






if __name__ == '__main__':
    voc_data_dir = r'D:\迅雷下载\AI数据集汇总\for_zihui'

    save_yolov3_coco_direct_name = "yolov3_coco_data"
    get_voc_data(voc_data_dir, save_yolov3_coco_direct_name)

    # print(train_object_id_map)