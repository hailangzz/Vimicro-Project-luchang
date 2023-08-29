import os

save_deal_rec_fill_path=r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\all_detect_crops_database_labels_file\deal_detect_SZ_DVR_car_brand_crops_focus_8_pt_labels.txt'
save_rec_cur = open(save_deal_rec_fill_path,'w',encoding='utf-8')

rec_file_path = r'D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\detect_car_brand_crops\all_detect_crops_database_labels_file\detect_SZ_DVR_car_brand_crops_focus_8_pt_labels_origin.txt'
rec_file_cur = open(rec_file_path,encoding='utf-8')
all_rec_info = rec_file_cur.readlines()

car_brand_search_char_list = [char.strip() for char in open(r'./car_brand_char_label.txt','r',encoding='utf-8').readlines()][-39:]
# print(car_brand_search_char_list)

total_deal_rec_info = []
for row_rec_info in all_rec_info:
    single_rec_info = []
    now_row_rec_info_list = row_rec_info.split('\t')
    if now_row_rec_info_list[-1]!="('', 0.0)\n": #排除识别为空的样本
        # print(now_row_rec_info_list)
        image_path = now_row_rec_info_list[0]
        rec_label_info = now_row_rec_info_list[1][2:-2].replace('"','\'').split('\', ')  #切分标注信息，且对干扰字符“进行替换操作
        if float(rec_label_info[-1])>=0.94 and rec_label_info[0][0] in car_brand_search_char_list and len(rec_label_info[0]) in (7,8):
            # print(image_path)
            # print(rec_label_info[1])
            save_rec_cur.write(image_path+'\t'+rec_label_info[0]+'\n')
save_rec_cur.close()

        # print(rec_label_info[0])