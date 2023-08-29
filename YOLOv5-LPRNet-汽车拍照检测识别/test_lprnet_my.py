# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''
from torch.utils.data import DataLoader


from PIL import Image, ImageDraw, ImageFont

# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import argparse
import torch
import time
import cv2
import os

from models.LPRNet import CHARS, LPRNet
from utils.load_lpr_data import LPRDataLoader





provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']



# LPR预处理
def LPR_transform(img):
    img = np.asarray(img)  # 转成array,变成24*94*3
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)  # 转成tensor类型
    img = img.unsqueeze(0)  # 添加batch维度 1*24*94*3
    return img

def test():
    print('hello!!!')

    total = 0
    tp = 0

    filename = r'./demo/rec_test/A0K986_176.jpg'
    img_o=cv2.imread(filename)
    # img_o = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)  # 防止无法读取汉字路径
    # img_o = cv2.cvtColor(Image)

    print('origin img:',img_o[0:21,0:94,:],img_o[0:21,0:94,:].shape)

    img_o=img_o[0:21,0:94,:]
    img_crop = cv2.resize(img_o, (94, 24))
    print('img_crop_resize:',img_crop,img_crop.shape)
    img_crop = LPR_transform(img_crop)  # LPR预处理
    print('img_transform:',img_crop)



    #########################################################
    lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(torch.device("cpu"))
    lprnet.load_state_dict(torch.load(r'D:\中星微人工智能工作\Total_Models\yolov5-LPRNet-车牌检测识别/lprnet_best.pth', map_location=torch.device('cpu')))
    lprnet.to(torch.device("cpu")).eval()
    print("load rec pretrained model successful!")
    print('torch.Tensor(img_crop):',torch.Tensor(img_crop),torch.Tensor(img_crop).shape)

    print(lprnet,torch.device("cpu"),type(torch.device("cpu")))
    np.random.seed(1)
    # img_crop = np.random.uniform(0, 1, size=[1, 3, 24, 94])
    # print('input_random ims:', img_crop)
    preds = lprnet(torch.Tensor(img_crop).to(torch.device("cpu")))  # classifier prediction

    prebs = preds.cpu().detach().numpy()
    print('prebs: ', prebs, prebs.shape)
    # lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS),
    #                       dropout_rate=1)
    # LPR_device = torch.device("cpu")
    # lprnet.to(LPR_device)  # 加载LPR网络
    # print("Successful to build lprnetwork!")
    #
    # # load pretrained model 加载训练好的权重
    #
    # lprnet.load_state_dict(torch.load(r'D:\中星微人工智能工作\Total_Models\yolov5-LPRNet-车牌检测识别/lprnet_best.pth',map_location=torch.device('cpu')))


    # img_crop = Variable(img_crop)
    # # forward
    # prebs = lprnet(img_crop)
    # prebs = prebs.cpu().detach().numpy()
    preb = prebs[0, :, :] # preb为68*18的矩阵，车牌有68个可选字符
    print(preb)
    preb_label = list()
    for j in range(preb.shape[1]):
        preb_label.append(np.argmax(preb[:, j], axis=0))  # argmax返回最大值的索引值
    no_repeat_blank_label = list()  # 存最后8个字符label的list，18个字符要去掉部分重复的和‘-’
    pre_c = preb_label[0]  # preb_label为1*18
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in preb_label:  # dropout repeate label and blank label
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c

    print(preb_label)



if __name__ == "__main__":
    test()