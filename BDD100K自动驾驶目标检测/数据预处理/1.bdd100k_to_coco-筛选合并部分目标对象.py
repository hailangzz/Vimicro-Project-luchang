#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright @ 2021 zuosi <807269961@qq.com>
# Distributed under terms of the MIT license
import re
import os
import json


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

    def bdd2yolov5(self, path):
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
            for single_image_info in j:
                # print(single_image_info)
                # 循环遍历单张图片标注信息labels列表。
                dw = 1.0 / self.bdd100k_width
                dh = 1.0 / self.bdd100k_height
                for row_label in single_image_info['labels']:
                    # 此条标注的类别为 self.select_categorys 时，提取标注信息。
                    if row_label["category"] in self.select_categorys:
                        # print(row_label["category"])
                        idx = self.cat2id[row_label["category"]]
                        cx = (row_label["box2d"]["x1"] + row_label["box2d"]["x2"]) / 2.0
                        cy = (row_label["box2d"]["y1"] + row_label["box2d"]["y2"]) / 2.0
                        w = row_label["box2d"]["x2"] - row_label["box2d"]["x1"]
                        h = row_label["box2d"]["y2"] - row_label["box2d"]["y1"]
                        if w <= 0 or h <= 0:
                            continue
                        if self._filter_by_box(w, h):
                            continue
                        # 根据图片尺寸进行归一化
                        cx, cy, w, h = cx * dw, cy * dh, w * dw, h * dh
                        line = f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                        lines += line

                if len(lines) != 0:
                    # 转换后的以*.txt结尾的标注文件我就直接和*.json放一具目录了
                    # yolov5中用到的时候稍微挪一下就行了
                    yolo_txt = path.replace(".json", ".txt")
                    with open(yolo_txt, 'w') as fp2:
                        fp2.writelines(lines)
                # print("%s has been dealt!" % path)
            # # return
            # for fr in j["frames"]:
            #     dw = 1.0 / self.bdd100k_width
            #     dh = 1.0 / self.bdd100k_height
            #     for obj in fr["objects"]:
            #         if obj["category"] in self.select_categorys:
            #             idx = self.cat2id[obj["category"]]
            #             cx = (obj["box2d"]["x1"] + obj["box2d"]["x2"]) / 2.0
            #             cy = (obj["box2d"]["y1"] + obj["box2d"]["y2"]) / 2.0
            #             w = obj["box2d"]["x2"] - obj["box2d"]["x1"]
            #             h = obj["box2d"]["y2"] - obj["box2d"]["y1"]
            #             if w <= 0 or h <= 0:
            #                 continue
            #             if self._filter_by_box(w, h):
            #                 continue
            #             # 根据图片尺寸进行归一化
            #             cx, cy, w, h = cx * dw, cy * dh, w * dw, h * dh
            #             line = f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            #             lines += line
            #     if len(lines) != 0:
            #         # 转换后的以*.txt结尾的标注文件我就直接和*.json放一具目录了
            #         # yolov5中用到的时候稍微挪一下就行了
            #         yolo_txt = path.replace(".json", ".txt")
            #         with open(yolo_txt, 'w') as fp2:
            #             fp2.writelines(lines)
            #     # print("%s has been dealt!" % path)


if __name__ == "__main__":
    bdd_label_dir = r"D:\迅雷下载\AI数据集汇总\自动驾驶数据集\Obeject Detect\BDD100K\bdd100k_labels_release\val"
    cvt = Bdd2yolov5()
    for path in search_file(bdd_label_dir, r"\.json$"):

        cvt.bdd2yolov5(path)


