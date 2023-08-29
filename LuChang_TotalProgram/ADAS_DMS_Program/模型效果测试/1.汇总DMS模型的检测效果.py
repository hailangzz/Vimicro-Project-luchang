import os
from shutil import copy2,move

def DMS_model_detect_file_info(DMS_detect_result_file):
    detect_info_dict={'this images is readding finish':[],'event_calling':[],'event_eyeclose':[],'event_glass':[],'event_mask':[],'event_smoke':[],'event_yawn':[]}
    file_cur = open(DMS_detect_result_file,'r')
    all_detect_info_lines=file_cur.readlines()

    for single_detect_info in all_detect_info_lines:
        for keys in detect_info_dict:
            if keys in single_detect_info:
                detect_info_dict[keys].append(single_detect_info.split('  ')[0])
    print(len(detect_info_dict['this images is readding finish']))
    return detect_info_dict




def cheak_save_test_DMS_folder(detect_info_dict,save_father_path=r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\DMS_test_save'):
    for key in detect_info_dict:
        if key =='this images is readding finish' and not os.path.exists(os.path.join(save_father_path,'normal_behavior')):
            os.mkdir(os.path.join(save_father_path,'normal_behavior'))
        elif key !='this images is readding finish' and not os.path.exists(os.path.join(save_father_path, key)):
            os.mkdir(os.path.join(save_father_path, key))

def statistics_DMS_detect_info(origin_total_images_path,detect_info_dict,save_father_path):
    cheak_save_test_DMS_folder(detect_info_dict,save_father_path)
    origin_total_images_name_list = os.listdir(origin_total_images_path)

    detect_test_have_actions_image_list = []

    for key in detect_info_dict:
        if key !='this images is readding finish':
            for image_name in detect_info_dict[key]:
                detect_test_have_actions_image_list.append(image_name)



    for image_name in detect_info_dict['this images is readding finish']:
        origin_image_path = os.path.join(origin_total_images_path,image_name)

        if image_name not in detect_test_have_actions_image_list:
            save_image_path = os.path.join(save_father_path,'normal_behavior',image_name)
            copy2(origin_image_path, save_image_path)
        else:
            for keys in detect_info_dict:
                if image_name in detect_info_dict[keys] and keys!='this images is readding finish':
                    save_image_path = os.path.join(save_father_path, keys, image_name)
                    copy2(origin_image_path, save_image_path)


origin_total_images_path = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\LuChang_part'
DMS_detect_result_file = r'D:\LuChang_Program_Total\ADAS_DMS项目\DMS_Result.txt'
save_father_path = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\DMS_test_save'
detect_info_dict=DMS_model_detect_file_info(DMS_detect_result_file)
statistics_DMS_detect_info(origin_total_images_path,detect_info_dict,save_father_path)
print(detect_info_dict)