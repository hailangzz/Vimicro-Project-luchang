import struct
import os
import cv2 as cv
import numpy as np

dest_image_size={"image_weight":120,"image_height":48}
if __name__ == '__main__':
    filepath = './attr_48_120_10.bin'
    binfile = open(filepath, 'rb')  # 打开二进制文件
    # filepath = './attr_48_120_10_i.bin'
    # binfile = open(filepath, 'rb')
    size = os.path.getsize(filepath)  # 获得文件大小
    print(size)

    Data = []
    i = 1
    while (i < size + 1):
        data = binfile.read(1)  # 每次输出一个字节
        # print(data)
        num = struct.unpack('B', data)
        if (i % 4 != 0):
            Data.append(num[0])  # 每四位是一个标志位，将四的倍数以外的有效数据保存
        i = i + 1
    # print(len(Data))
    R_start = 0
    R_end = dest_image_size['image_weight']
    G_start = dest_image_size['image_weight']
    G_end = dest_image_size['image_weight']*2
    B_start = dest_image_size['image_weight']*2
    B_end = dest_image_size['image_weight']*3
    red_start = dest_image_size['image_weight']*3
    red_end = dest_image_size['image_weight']*4
    R = []
    G = []
    B = []
    ir = []
    # 提取红色分量的像素值（过滤标识位）
    while R_end <= len(Data) - dest_image_size['image_weight']*3:
        for j in Data[R_start:R_end]:
            R.append(j)
        R_start = R_start + dest_image_size['image_weight']*4
        R_end = R_end + dest_image_size['image_weight']*4

    while G_end <= len(Data) - dest_image_size['image_weight']*2:
        for k in Data[G_start:G_end]:
            G.append(k)
        G_start = G_start + dest_image_size['image_weight']*4
        G_end = G_end + dest_image_size['image_weight']*4

    while B_end <= len(Data) - dest_image_size['image_weight']:
        for m in Data[B_start:B_end]:
            B.append(m)
        B_start = B_start + dest_image_size['image_weight']*4
        B_end = B_end + dest_image_size['image_weight']*4

    while red_end <= len(Data):
        for n in Data[red_start:red_end]:
            ir.append(n)
        red_start = red_start + dest_image_size['image_weight']*4
        red_end = red_end + dest_image_size['image_weight']*4

    # print(len(R),len(G),len(B),len(ir))
    # 将list转化为np.array
    IR = np.array(ir)
    IR = IR.reshape(dest_image_size['image_height'], dest_image_size['image_weight'])
    # cv.imshow('ir', IR)
    cv.imwrite("ir.bmp", IR)

    BGR = []
    for p in range(len(R)):
        BGR.append(B[p])
        BGR.append(G[p])
        BGR.append(R[p])
    # print(len(BGR))

    BGR = np.array(BGR)
    BGRimage = BGR.reshape(dest_image_size['image_height'], dest_image_size['image_weight'], 3)
    cv.imwrite("BGR.bmp", BGRimage)
    # cv.waitKey()
    binfile.close()