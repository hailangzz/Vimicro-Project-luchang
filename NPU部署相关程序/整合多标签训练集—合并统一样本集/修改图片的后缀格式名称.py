import os

origin_file_path = r'D:\PycharmProgram\ZZ_Total_DeepLearning_Model\yolov7-face-main\detect_DVR_Vimicro_Image\shanghai_DVR_face_detect'

total_image_name = [image_name for image_name in os.listdir(origin_file_path)]
for image_name in total_image_name:
    split_image_name = image_name.split('.')
    if split_image_name[-1]!='.jpg':
        os.rename(os.path.join(origin_file_path,image_name), os.path.join(origin_file_path,split_image_name[0]+'.jpg'))

