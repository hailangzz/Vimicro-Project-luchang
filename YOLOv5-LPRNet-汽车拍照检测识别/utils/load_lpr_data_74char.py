from imutils import paths
import numpy as np
import random
import cv2
import os

from torch.utils.data import Dataset

CHARS = ['0','1','2','3','4','5','6','7','8','9',
         'A','B','C','D','E','F','G','H','J','K',
         'L','M','N','P','Q','R','S','T','U','V',
         'W','X','Y','Z',
         '京','津','冀','蒙','青','陕','甘','宁',
         '晋','鲁','豫','皖','鄂','云','贵','桂',
         '川','渝','新','藏','黑','吉','辽','浙',
         '苏','沪','湘','闽','赣','琼',
         '粤','港','澳','学','挂','警','使','应','急', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}   # {'京': 0, '沪': 1, '津': 2, '渝': 3, '冀': 4, '晋':

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        # print(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        # print(filename)
        # Image = cv2.imread(filename)
        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1) # 防止无法读取汉字路径
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        height, width, _ = Image.shape
        # print(self.img_size)
        if height != self.img_size[1] or width != self.img_size[0]:
            self.resize = cv2.resize(Image, self.img_size)
            Image = self.resize
        # print('Image.shape: ',Image.shape,Image)
        Image = self.PreprocFun(Image) #图形标准化转换
        # print(Image.shape,Image)
        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        # print('imgname, suffix: ',imgname, suffix)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])
        # print('label:',label)

        # if len(label) == 8:
        #     # 注不检测电动车牌的有效性
        #     # if self.check(label) == False:
        #     #     print(imgname)
        #     #     assert 0, "Error label ^~^!!!"
        #     pass
        # # print(imgname,label)
        return Image, label, len(label)

    def transform(self, img): #标注化转换
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1)) #将输入图像转为（channle,x,y）

        # print('img transform:',img)

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True



