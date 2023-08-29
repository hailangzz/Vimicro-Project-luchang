import os

def cover_label_function(cover_label_maps={},origin_coco_label_path=r'D:\Git_WareHouse\yolov5-master\detect\exp\labels',save_label_path=r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\model_conver_save_labels\bdd_100k_model_draw_box_images'):
    total_image_name_list = os.listdir(origin_coco_label_path)
    for single_image_label_name in total_image_name_list:
        single_image_label_path = os.path.join(origin_coco_label_path, single_image_label_name)
        # print(single_image_label_path)
        single_label_cur=open(single_image_label_path,'r')
        cover_save_label_cur = open(os.path.join(save_label_path,single_image_label_name),'w')

        origin_all_label_rows = single_label_cur.readlines()
        for row_info in origin_all_label_rows:
            row_info = row_info.split(' ')
            row_info[0]=cover_label_maps[row_info[0]]
            row_info = ' '.join(row_info)
            cover_save_label_cur.write(row_info)
            # print(row_info)
        cover_save_label_cur.close()


cover_label_maps={'7':'0','8':'1','9':'2','10':'5'}

cover_label_function(cover_label_maps)