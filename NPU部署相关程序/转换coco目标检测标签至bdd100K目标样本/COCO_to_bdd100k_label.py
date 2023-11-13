import os
import json

sample_bdd_100k_label_path = r"E:\Pycharm_Workspace\LUChang_TotalProgram\YOLOP_ChangeNet\database\bdd\bdd100k\labels\train\0000f77c-62c2a288.json"
sample_bdd_100k_label_cur =open(sample_bdd_100k_label_path,'r')
sample_bdd_100k_label_info = json.load(sample_bdd_100k_label_cur)
print(sample_bdd_100k_label_info)

sample_bdd_100k_label_path = r"F:\AiTotalDatabase\ADAS_test_images\json_label\000-GBT19056_粤B88888_F_20230822155502_25.json"
sample_bdd_100k_label_cur =open(sample_bdd_100k_label_path,'r')
sample_bdd_100k_label_info = json.load(sample_bdd_100k_label_cur)
print(sample_bdd_100k_label_info)

# 以下读取本地标注图像信息
local_label_path=r"F:\AiTotalDatabase\ADAS_test_images\detect_label_result\000-GBT19056_粤B88888_F_20230822155502_25"

def cover_coco_label_to_bdd100k(local_label_path):
    bdd100k_info_dict={}
    bdd100k_info_dict['name']=local_label_path.split('\\')[-1].split('.')[0]
    bdd100k_info_dict['frames'] = [{'objects': []}]

    id_index = 0

    local_label_info=open(local_label_path,'r').readlines()
    for single_row in local_label_info:
        [x1,y1,x2,y2] = map(int,(single_row.strip().split(' ')[-4:]))
        single_bdd_info={'category':'car','id': id_index,'box2d': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}}
        # print(x1,y1,x2,y2)
        id_index+=1
        bdd100k_info_dict['frames'][0]['objects'].append(single_bdd_info)
    return bdd100k_info_dict

# 批量数据转换
def batch_coco_label_cover_bdd100k(total_coco_label_path,save_bdd100k_path):
    total_coco_label_name = os.listdir(total_coco_label_path)
    for single_coco_label_name in total_coco_label_name:

        single_coco_label_path = os.path.join(total_coco_label_path,single_coco_label_name)
        bdd100k_info_dict = cover_coco_label_to_bdd100k(single_coco_label_path)
        save_db100k_json_path = os.path.join(save_bdd100k_path,bdd100k_info_dict['name']+'.json')
        # 将数据保存到JSON文件
        with open(save_db100k_json_path, "w") as json_file:
            json.dump(bdd100k_info_dict, json_file)

# bdd100k_info_dict = cover_coco_label_to_bdd100k(local_label_path)
batch_coco_label_cover_bdd100k("F:\AiTotalDatabase\ADAS_test_images\detect_label_result","F:\AiTotalDatabase\ADAS_test_images\json_label")

