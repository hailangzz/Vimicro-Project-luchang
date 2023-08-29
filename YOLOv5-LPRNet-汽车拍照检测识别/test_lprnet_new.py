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
import numpy as np
import argparse
import torch
import time
import cv2
import os

from models.LPRNet import CHARS, LPRNet
from utils.load_lpr_data import LPRDataLoader


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--test_img_dirs', default=r"./demo/rec_test", help='the test images path')
    parser.add_argument('--dropout_rate', default=1, help='dropout rate.') #神经元屏蔽比例：
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default=r'D:\中星微人工智能工作\Total_Models\yolov5-LPRNet-车牌检测识别/runsLPRNet__iteration_1000.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    args = get_parser()

    lprnet = LPRNet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device).eval()  # 模型测试的时候使用的参数

    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model,map_location=torch.device('cpu')))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    print('test_img_dirs',test_img_dirs)

    image_list = os.listdir(test_img_dirs)

    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len) #对输入样本进行预处理的类
    print('test_dataset: ',test_dataset)
    print('load test image number is: ',len(test_dataset))
    try:
        Greedy_Decode_Eval(lprnet, test_dataset,image_list, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets,image_list, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size

    print('epoch_size::::',epoch_size,len(datasets))
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, label,lengths = next(batch_iterator)

        # start = 0
        # targets = []
        # for length in lengths:
        #     label = labels[start:start+length]
        #     targets.append(label)
        #     start += length
        # targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        # images: [bs, 3, 24, 94]
        # prebs:  [bs, 68, 18]
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        # print('prebs:  !!!',prebs,prebs.shape)
        # print(prebs[0,:,0])
        # print(prebs[0, :, 1])
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]  # 对每张图片 [68, 18]

            # print('rec preb info:',preb,preb.shape)
            preb_label = list()
            for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
                preb_label.append(np.argmax(preb[:, j], axis=0))
            # print('preb_label: ',preb_label,len(preb_label))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:  # 记录重复字符
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # 去除重复字符和空白字符'-'
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)  # 得到最终的无重复字符和无空白字符的序列
            print('preb_labels: ',preb_labels,len(preb_labels))

            word_list = []
            for word_index in range(len(preb_labels[0])):
                word_list.append(CHARS[preb_labels[0][word_index]])
                # print(CHARS[preb_labels[0][word_index]])
            print(str(word_list))


if __name__ == "__main__":
    test()
