import os
import json


json_dir = r'F:\AiTotalDatabase\aluminum\annotations\train.json'  # json文件路径
out_dir = r'F:\AiTotalDatabase\aluminum\annotations\train'  # 输出的 txt 文件路径

def read_josn_keys_info(content):

    database_categories = {}
    database_annotations = {}
    database_images = {}

    for categories_info in content['categories']:
        if categories_info['id'] not in database_categories:
            database_categories[categories_info['id']]=categories_info['name']
    for annotations_info in content['annotations']:
        if annotations_info['image_id'] not in database_annotations:
            database_annotations[annotations_info['image_id']]={}
            if annotations_info['category_id'] not in database_annotations[annotations_info['image_id']]:
                database_annotations[annotations_info['image_id']][annotations_info['category_id']]=[]
                database_annotations[annotations_info['image_id']][annotations_info['category_id']].append(annotations_info['bbox'])
            else:
                database_annotations[annotations_info['image_id']][annotations_info['category_id']].append(annotations_info['bbox'])
        else:
            if annotations_info['category_id'] not in database_annotations[annotations_info['image_id']]:
                database_annotations[annotations_info['image_id']][annotations_info['category_id']]=[]
                database_annotations[annotations_info['image_id']][annotations_info['category_id']].append(annotations_info['bbox'])
            else:
                database_annotations[annotations_info['image_id']][annotations_info['category_id']].append(annotations_info['bbox'])
    for images_info in content['images']:
        if images_info['id'] not in database_images:
            database_images[images_info['id']]={'image_name':images_info['file_name'].split('/')[-1],'height':images_info['height'],'width':images_info['width']}

    return database_categories,database_annotations,database_images

def main(save_yolo_labels_path):
    # 读取 json 文件数据
    with open(json_dir, 'r') as load_f:
        content = json.load(load_f)

    database_categories,database_annotations,database_images=read_josn_keys_info(content)
    print(database_categories)
    print(database_annotations)
    print(database_images)
    # 循环处理
    for image_id in database_images:
        image_name = database_images[image_id]['image_name']
        image_box_info = database_annotations[image_id]
        print(image_id)
        label_txt_cur=open(os.path.join(save_yolo_labels_path,image_name.replace('.jpg','.txt')),'w')
        for categories_key in image_box_info:
            for box_info in image_box_info[categories_key]:
                x = (box_info[0] + box_info[2] / 2) / database_images[image_id]['width']
                y = (box_info[1] + box_info[3] / 2) / database_images[image_id]['height']
                w = (box_info[2]) / database_images[image_id]['width']
                h = (box_info[3]) / database_images[image_id]['height']

                label_txt_cur.write(str(categories_key-1)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')
        label_txt_cur.close()


        # print(content.keys())
        # print(len(content['info']), content['info'])
        # print(len(content['categories']), content['categories'])
        # print(len(content['annotations']), content['annotations'])
        # print(len(content['images']),content['images'])
        # tmp = t['name'].split('.')
        # filename = out_dir + tmp[0] + '.txt'
        #
        # if os.path.exists(filename):
        #     # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值
        #     x = (t['bbox'][0] + t['bbox'][2]) / 2 / t['image_width']
        #     y = (t['bbox'][1] + t['bbox'][3]) / 2 / t['image_height']
        #     w = (t['bbox'][2] - t['bbox'][0]) / t['image_width']
        #     h = (t['bbox'][3] - t['bbox'][1]) / t['image_height']
        #     fp = open(filename, mode="r+", encoding="utf-8")
        #     file_str = str(t['category']) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
        #                ' ' + str(round(h, 6))
        #     line_data = fp.readlines()
        #
        #     if len(line_data) != 0:
        #         fp.write('\n' + file_str)
        #     else:
        #         fp.write(file_str)
        #     fp.close()
        #
        # # 不存在则创建文件
        # else:
        #     fp = open(filename, mode="w", encoding="utf-8")
        #     fp.close()


if __name__ == '__main__':
    save_yolo_labels_path = r'F:\AiTotalDatabase\aluminum\train\labels'
    main(save_yolo_labels_path)