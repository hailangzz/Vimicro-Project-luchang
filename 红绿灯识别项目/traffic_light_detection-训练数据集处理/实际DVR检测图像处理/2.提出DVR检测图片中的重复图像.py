import os
import re
from shutil import copy2

oringin_DVR_path = r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\extract_detect_traffic_light'
save_search_DVR_path = r"D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\extract_detect_traffic_light\search_extract_detect_traffic_light3"
total_image_name_list = [image_name for image_name in os.listdir(oringin_DVR_path)]
print(total_image_name_list)

# 创建image名切分字典结构，将帧信息切分出来
def cut_image_namg_info(total_image_name_list):

    cut_image_info_dict = {}
    for image_name in total_image_name_list:
        data_flag = re.findall('.*(?=NF_)', image_name)[0]+'NF_'
        frame_info = int(re.findall('(?<=NF_).*', image_name)[0].split('.')[0])
        if data_flag not in cut_image_info_dict:
            cut_image_info_dict[data_flag]={'frame_list':[],'frame_min':0,'frame_save_list':[]}
            cut_image_info_dict[data_flag]['frame_list'].append(frame_info)
        else:
            cut_image_info_dict[data_flag]['frame_list'].append(frame_info)

    print(cut_image_info_dict)
    print(len(cut_image_info_dict))
    return cut_image_info_dict

def order_image_frame_info(cut_image_info_dict): #对图像帧进行排序，获取最小帧数值

    for key in cut_image_info_dict:
        cut_image_info_dict[key]['frame_min'] = min(cut_image_info_dict[key]['frame_list'])
        list.sort(cut_image_info_dict[key]['frame_list'])

    print(cut_image_info_dict)
    return cut_image_info_dict


def dizeng(list,diff_number = 7):
    l1 = []
    l2 = []
    for i in range(0,len(list)-1):
        if list[i]+diff_number==list[i+1]:
            l2.append(list[i])
            l2.append(list[i+1])
            if i==len(list)-2:
                l1.append(l2)
        else:
            l1.append(l2)
            l2=[]
            continue

    l1_1 = [i for i in l1 if i]
    l1_2 = []
    for a in l1_1:
        list2 = []
        [list2.append(i) for i in a if not i in list2]
        l1_2.append(list2)
    return(l1_2)


# 根据帧数间隔，去点接近的图像帧，减少图片样本量
def reduce_image_number(order_cut_image_info_dict,frame_interval=30):
    total_image_number = 0
    for key in order_cut_image_info_dict:
        '''
        # # 策略一：以恒定间隔帧 900来提取目标图像。
        # for frame_value in order_cut_image_info_dict[key]['frame_list']:
        #     real_differ = frame_value-order_cut_image_info_dict[key]['frame_min']
        #     if real_differ>frame_interval or real_differ==0:
        #         order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
        #         total_image_number += 1
        # 策略二：只提取帧的前后20张图像
        if len(order_cut_image_info_dict[key]['frame_list'])<20:
            for frame_value in order_cut_image_info_dict[key]['frame_list']:
                order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
                total_image_number += 1
        else:
            for frame_value in order_cut_image_info_dict[key]['frame_list'][:10]:
                order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
                total_image_number += 1
            for frame_value in order_cut_image_info_dict[key]['frame_list'][-10:]:
                order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
                total_image_number += 1
        '''
        # 策略三：考虑到一段视频中，有效视频其为连续的帧差值为7，则添加这一特性，来整理帧信息；
        continue_frame_list = dizeng(order_cut_image_info_dict[key]['frame_list'])
        total_continue_frame =  [item for sublist in continue_frame_list for item in sublist]


        # for frame_value in order_cut_image_info_dict[key]['frame_list']:
        #     if frame_value not in total_continue_frame:
        #         order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
        #         total_image_number += 1
        for continue_frame in continue_frame_list:
            if len(continue_frame) < 10:
                # pass
                if len(continue_frame)<5:
                    for frame_value in continue_frame:
                        order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
                        total_image_number += 1
            else:
                for frame_value in continue_frame[:5]:
                    order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
                    total_image_number += 1
                for frame_value in continue_frame[-5:]:
                    order_cut_image_info_dict[key]['frame_save_list'].append(frame_value)
                    total_image_number += 1




    print(total_image_number)
    print(order_cut_image_info_dict)
    return order_cut_image_info_dict

def save_search_frame_image(reduce_image_info_dict):
    for key in reduce_image_info_dict:
        for search_frame_value in reduce_image_info_dict[key]['frame_save_list']:
            image_path = os.path.join(oringin_DVR_path,key+str(search_frame_value)+'.jpg')
            copy2(image_path, save_search_DVR_path)




cut_image_info_dict = cut_image_namg_info(total_image_name_list)
order_cut_image_info_dict = order_image_frame_info(cut_image_info_dict)
reduce_image_info_dict = reduce_image_number(order_cut_image_info_dict,frame_interval=900)
save_search_frame_image(reduce_image_info_dict)







