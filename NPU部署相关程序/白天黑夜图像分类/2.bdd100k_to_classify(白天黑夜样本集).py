#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright @ 2021 zuosi <807269961@qq.com>
# Distributed under terms of the MIT license
import re
import os
import json
from shutil import copy2,move

def search_file(data_dir, pattern=r'\.jpg$'):
    root_dir = os.path.abspath(data_dir)
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if re.search(pattern, f, re.I):
                abs_path = os.path.join(root, f)
                # print('new file %s' % absfn)
                yield abs_path


class Bdd2yolov5:
    def __init__(self):
        self.bdd100k_width = 1280
        self.bdd100k_height = 720
        self.select_categorys = ["person", "car", "bus", "truck"] #此处指只检测"person", "car", "bus", "truck"这几类

        # 此处指将car、bus、truck合并为同一类
        self.cat2id = {
            "person": 0,
            "car": 1,
            "bus": 1,
            "truck": 1
                      }
        self.select_all_classify_info()

    # 表示选取所有标签种类标注信息
    def select_all_classify_info(self,):
        self.select_categorys = self.all_categorys
        self.cat2id={}
        for classify_id in range(len(self.select_categorys)):
            self.cat2id[self.select_categorys[classify_id]]= classify_id

    @property
    def all_categorys(self):
        return ["person", "rider", "car", "bus", "truck", "bike",
                "motor", "traffic light", "traffic sign", "train"]

    def _filter_by_attr(self, attr=None):
        if attr is None: # 判断是否什么都没有
            return False
            # 过滤掉晚上的图片
        if attr['timeofday'] == 'night':  # 确定图片是否是晚上
            return True
        return False

    def _filter_by_box(self, w, h):
        # size ratio
        # 过滤到过于小的小目标
        threshold = 0.001
        if float(w * h) / (self.bdd100k_width * self.bdd100k_height) < threshold:  # 确定标注框是否过小
            return True
        return False

    def bdd2yolov5(self, path,search_file):
        lines = ""
        with open(path) as fp:
            j = json.load(fp)

            # if self._filter_by_attr(j['attributes']): # 去掉夜间场景的标注图像信息
            #     return
            # print(j[10])
            # print(j[10].keys())
            # for keyname in j[10].keys():
            #     print(keyname,j[10][keyname])
            #
            # print(j[10]['labels'][0].keys())
            # for label in j[10]['labels']:
            #     print(label.keys())

            # return

            # 开始循环每张图像标注信息列表：
            sample_daytime_path = os.path.join(search_file,'daytime')
            sample_night_path = os.path.join(search_file, 'night')

            if not os.path.exists(os.path.join(search_file,'daytime')):
                os.mkdir(os.path.join(search_file,'daytime'))
            if not os.path.exists(os.path.join(search_file,'night')):
                os.mkdir(os.path.join(search_file,'night'))



            for single_image_info in j:
                origin_image_path = r'D:\迅雷下载\AI数据集汇总\自动驾驶数据集\Obeject Detect\BDD100K\bdd100k_images_100k\bdd100k\images\100k\val'
                origin_image_path = os.path.join(origin_image_path, single_image_info['name'])
                if single_image_info['attributes']['timeofday'] == 'night':

                    dst_image_path = os.path.join(sample_night_path, single_image_info['name'])
                    copy2(origin_image_path, dst_image_path)

                elif single_image_info['attributes']['timeofday']!='night':
                    dst_image_path = os.path.join(sample_daytime_path,single_image_info['name'])
                    print(origin_image_path)
                    print(dst_image_path)
                    copy2(origin_image_path, dst_image_path)



                # if single_image_info['attributes']['timeofday'] not in ['daytime','night']:
                #     print(single_image_info['name'],': is night')



if __name__ == "__main__":
    bdd_label_dir = r"D:\迅雷下载\AI数据集汇总\自动驾驶数据集\Obeject Detect\BDD100K\bdd100k_labels_release\val"
    cvt = Bdd2yolov5()
    for path in search_file(bdd_label_dir, r"\.json$"):

        cvt.bdd2yolov5(path,bdd_label_dir)


