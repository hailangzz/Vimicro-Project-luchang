import time
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# 绘图函数
def show_images(imgs, num_rows, num_cols, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)



test_images='./00a0f008-3c67908e.jpg'
def test1(img=Image.open(test_images)):
    """图像左右翻转测试"""
    apply(img, torchvision.transforms.RandomHorizontalFlip())


def test2(img=Image.open(test_images)):
    """图像上下翻转测试"""
    apply(img, torchvision.transforms.RandomVerticalFlip())

def test3(img=Image.open(test_images)):
    #图像裁剪
    # 参数依次为：
    # 200 --> 将高和宽缩放到200像素
    # (0.1, 1）--> 裁剪区域的面积占原始图像面积的比例
    # (0.5, 2) --> 宽和高比例的取值范围
    shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
    apply(img, shape_aug)

def test4(img=Image.open(test_images)):
    #     亮度(brightness)；
    #   2）对比度(contrast)；
    #   3）饱和度(saturation)；
    #   4）色调(hue)。
    color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
    )
    apply(img, color_aug)


if __name__ == '__main__':
    test4()

if __name__ == '__main__':
    test3()

if __name__ == '__main__':
    test2()

if __name__ == '__main__':
    test1()


