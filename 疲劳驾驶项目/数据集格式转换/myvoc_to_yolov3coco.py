import os
import xml.etree.ElementTree as ET
import shutil


train_object_id_map={}

def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def cover_annotations_to_yolov3_labels(use_type_info_dict,second_dir):

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
            if int(obj.find('difficult').text) == 0:
                if len(train_object_id_map)==0:
                    train_object_id_map[class_name]=0
                else:
                    if class_name not in train_object_id_map:
                        train_object_id_map[class_name]=max(train_object_id_map.values())+1

                xmlbox = obj.find('bndbox')
                x_d, y_d, w_d, h_d = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                # print(train_object_id_map[class_name],x_d, y_d, w_d, h_d)
                # class_name_id = yaml['names'].index(class_name)  # class id
                # out_file.write(" ".join([str(a) for a in (class_name_id, *bb)]) + '\n')
            write_label_cur.write(str(train_object_id_map[class_name])+' '+str(x_d)+' '+str(y_d)+' '+str(w_d)+' '+str(h_d)+'\n')

        write_label_cur.close()

def get_voc_data(train_and_valid_list,voc_data_dir,save_yolov3_coco_data_dir):
    for use_type in train_and_valid_list:
        use_type_info_dict={'images_path_list':[],'annotations_path_list':[],'images_name_list':[]}

        second_dir = os.path.join(save_yolov3_coco_data_dir,use_type.split('.')[0])

        if not os.path.exists(second_dir):
            os.mkdir(second_dir)
        use_type_list_info = open(os.path.join(voc_data_dir,use_type),'r').readlines()

        for single_image_info in use_type_list_info:
            single_image_info_list = single_image_info.strip().replace('./',"").split(' ')
            single_image_info_list[0]=os.path.join(voc_data_dir,single_image_info_list[0])
            single_image_info_list[1] = os.path.join(voc_data_dir, single_image_info_list[1])

            use_type_info_dict['images_path_list'].append(single_image_info_list[0])
            use_type_info_dict['annotations_path_list'].append(single_image_info_list[1])
            use_type_info_dict['images_name_list'].append(single_image_info_list[1].split('/')[-1].split('.')[0])
            # print("images_name:",single_image_info_list[1].split('/')[-1].split('.')[0])

        cover_annotations_to_yolov3_labels(use_type_info_dict,second_dir)






if __name__ == '__main__':
    voc_data_dir = r'D://PycharmProgram//PaddleDetection//dataset//roadsign_voc/'
    save_yolov3_coco_data_dir = r"./yolov3_coco_data/"
    if not os.path.exists(save_yolov3_coco_data_dir):
        os.mkdir(save_yolov3_coco_data_dir)
    train_and_valid_list = ['train.txt','valid.txt']
    get_voc_data(train_and_valid_list, voc_data_dir, save_yolov3_coco_data_dir)

    # print(train_object_id_map)