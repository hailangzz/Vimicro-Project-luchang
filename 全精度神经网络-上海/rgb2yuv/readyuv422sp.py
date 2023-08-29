import numpy as np
import cv2
import os


def read_yuv422(image_path, height, width):
    """
    :param image_path: 待转换的.yuv图像文件路径
    :param rows: 图像行数
    :param cols: 图像列数
    :return: y,u,v分量
    """
    fp = open(image_path, 'rb')#打开二进制文件
    # size = height * width * 3
    # size = os.path.getsize(image_path) #获得文件大小
    # for i in range(size):
        # #print(i)
        # data = fp.read(4) #每次输出一个字节
        # print(data)
    
    # h_new = height*width*2
    # raw=fp.read(h_new)
    # print(len(raw))
    # yuv = np.frombuffer(raw, dtype=np.uint8)
    # print(yuv)
    # # framesize = height * width * 2
    # # h_h = height // 2
    # # h_w = width // 2
    yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
    uv = np.zeros(shape=(height, width), dtype='uint8', order='C')
    # # vt = np.zeros(shape=(height, h_w), dtype='uint8', order='C')
    for m in range(height):
        for n in range(width):
            yt[m, n] = ord(fp.read(1))
            #print(yt[m, n])
            
    print('---------done----------')
    for m in range(height):
        for n in range(width):
            uv[m, n] = ord(fp.read(1))
            #print(uv[m, n])
    # for m in range(height):
        # for n in range(h_w):
            # vt[m, n] = ord(fp.read(1))
            
    #uv = np.concatenate((ut.reshape(-1), vt.reshape(-1)), axis = 2)
    #image = np.concatenate((yt.reshape(-1), uv.reshape(-1)), axis = 0)
    #img = image.reshape((height * 2, width)).astype('uint8') # YUV422sp 的存储格式为：NV16（YY UV）
    yt = yt[np.newaxis,:,:]
    uv = uv[np.newaxis,:,:]
    print(yt.shape)
    print(uv.shape)
    img = np.concatenate((yt, uv), axis = 0)
    print(img.shape)
    #return img
    #print(vt)
    #img = cv2.merge(yt, uv)
    #return yt
    
in_path = './yuv422sp_data/' 
out_path = './yuv422sp_data_out/'  
imagelist = os.listdir(in_path)
if not os.path.exists(out_path):
    os.mkdir(out_path)
for i in range(len(imagelist)):
    in_name = os.path.join(in_path, imagelist[i])
    #print(in_name)
    image_name, exten = os.path.splitext(imagelist[i])
    out_name = os.path.join(out_path + image_name + '.jpg')
    image = read_yuv422(in_name, 441, 358)
    #print(image.shape)
    #bgr_img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV16)
    #cv2.imwrite(out_name, bgr_img)
    
    
    
