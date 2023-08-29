import os
from shutil import copy2

oringin_path=r"D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\DMS效果测试事件分段"

save_test_part_image_path=r"D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\DMS效果测试事件分段_order"

def copy_order_test_image(oringin_path,save_test_part_image_path):
    all_classify_list=os.listdir(oringin_path)
    record_test_image_info={}
    total_image_id = 1
    for classify_name in all_classify_list:
        second_direct = os.path.join(oringin_path,classify_name)
        record_test_image_info[classify_name]={'image_start_id':0,'image_end_id':0}
        second_image_name_list = os.listdir(second_direct)
        for second_image_index in range(len(second_image_name_list)):
            if second_image_index==0:
                record_test_image_info[classify_name]['image_start_id'] = total_image_id
            record_test_image_info[classify_name]['image_end_id'] = total_image_id
            image_path_origin=os.path.join(second_direct,second_image_name_list[second_image_index])
            save_image_path = os.path.join(save_test_part_image_path,str(total_image_id)+'.jpg')
            copy2(image_path_origin, save_image_path)
            total_image_id+=1

    # 将标注信息存储起来：
    part_vidoe_labels_cur = open(os.path.join(save_test_part_image_path,'test_part_image_label_info.txt'),'w')
    for key,values in record_test_image_info.items():
        part_vidoe_labels_cur.write(key+': '+str(values)+'\n')
    part_vidoe_labels_cur.close()

copy_order_test_image(oringin_path, save_test_part_image_path)
