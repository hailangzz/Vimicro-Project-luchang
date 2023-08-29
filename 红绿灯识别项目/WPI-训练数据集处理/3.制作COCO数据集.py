import os
import shutil


origin_WPI_dataset_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\WPI Traffic Light Dataset\VOC2007'
label_path = os.path.join(origin_WPI_dataset_path,'annotation.txt')

#box=[single_image_info['xmin'],single_image_info['xmax'],single_image_info['ymin'],single_image_info['ymax'],]
def convert_box(box,size=[1920,1080]):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return [x * dw, y * dh, w * dw, h * dh]

def read_sample_label(label_path):

    label_dict={
                'train_label':{},
                'test_label': {},
                }
    txt_cur=open(label_path,'r')
    label_info_list = txt_cur.readlines()
    print(label_info_list)
    for single_row_label in label_info_list:
        buf_info = single_row_label.strip().split(' ')
        if 'test' in buf_info[0]:
            label_dict['test_label'][buf_info[0]]=[]
            for voc_info in buf_info[1:]:
                voc_info = voc_info.split(',')
                coco_label = [int(voc_info[-1])]
                box_info = convert_box([int(voc_info[0]),int(voc_info[2]),int(voc_info[1]),int(voc_info[3])])
                coco_label.extend(box_info)
                label_dict['test_label'][buf_info[0]].append(coco_label)
        else:
            label_dict['train_label'][buf_info[0]] = []
            for voc_info in buf_info[1:]:
                voc_info = voc_info.split(',')
                coco_label = [int(voc_info[-1])]
                box_info = convert_box([int(voc_info[0]), int(voc_info[2]), int(voc_info[1]), int(voc_info[3])])
                coco_label.extend(box_info)
                label_dict['train_label'][buf_info[0]].append(coco_label)
    return label_dict



def create_coco_dataset(label_dict):

    COCO_dataset_name = 'WPI_Traffic_Light_Dataset_COCO'
    print(os.listdir(origin_WPI_dataset_path))
    if COCO_dataset_name not in os.listdir(origin_WPI_dataset_path):
        print(COCO_dataset_name)
        os.mkdir(os.path.join(origin_WPI_dataset_path,COCO_dataset_name))

    if 'train' not in os.listdir(os.path.join(origin_WPI_dataset_path,COCO_dataset_name)):
        os.mkdir(os.path.join(origin_WPI_dataset_path, COCO_dataset_name,'train'))
        os.mkdir(os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'train','labels'))
        os.mkdir(os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'train', 'images'))
    if 'val' not in os.listdir(os.path.join(origin_WPI_dataset_path,COCO_dataset_name)):
        os.mkdir(os.path.join(origin_WPI_dataset_path, COCO_dataset_name,'val'))
        os.mkdir(os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'val', 'labels'))
        os.mkdir(os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'val', 'images'))

    train_label_save_path = os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'train','labels')
    val_label_save_path = os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'val', 'labels')

    train_image_save_path = os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'train', 'images')
    val_image_save_path = os.path.join(origin_WPI_dataset_path, COCO_dataset_name, 'val', 'images')
    for key_info in label_dict:
        if 'test' in key_info:
            for train_sample_name_key in label_dict[key_info]:
                #创建训练样本的标签信息
                train_cur = open(os.path.join(val_label_save_path,train_sample_name_key.replace('.jpg','.txt')),'w')
                for row_coco_info in label_dict[key_info][train_sample_name_key]:
                    train_cur.writelines(str(row_coco_info).replace(',','')[1:-1])
                    train_cur.write('\n')
                train_cur.close()

                # 将对应的样本图拷贝到对应目录中
                shutil.copy(os.path.join(origin_WPI_dataset_path,'JPEGImages',train_sample_name_key), os.path.join(val_image_save_path,train_sample_name_key))

        else:
            for train_sample_name_key in label_dict[key_info]:
                #创建训练样本的标签信息
                train_cur = open(os.path.join(train_label_save_path,train_sample_name_key.replace('.jpg','.txt')),'w')
                for row_coco_info in label_dict[key_info][train_sample_name_key]:
                    train_cur.writelines(str(row_coco_info).replace(',','')[1:-1])
                    train_cur.write('\n')
                train_cur.close()

                # 将对应的样本图拷贝到对应目录中
                shutil.copy(os.path.join(origin_WPI_dataset_path,'JPEGImages',train_sample_name_key), os.path.join(train_image_save_path,train_sample_name_key))



    # pass

label_dict = read_sample_label(label_path)
create_coco_dataset(label_dict)







