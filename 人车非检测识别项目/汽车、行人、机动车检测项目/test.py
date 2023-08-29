import os

origin_detect_labels_path = r"D:\Git_WareHouse\yolov5-master\runs\detect\test_DVR_all_classify\labels"
total_origin_labels = os.listdir(origin_detect_labels_path)

cheack_label_path = r"/home/zhangzhuo/git_workspace/yolov5/datasets/person_car_dataset/train/labels"
cheack_labels = os.listdir(cheack_label_path)

for single_label_name in total_origin_labels:
    if single_label_name not in cheack_labels:
        print(single_label_name)