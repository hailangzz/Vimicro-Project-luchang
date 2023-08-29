import numpy as np
import cv2
import os


def read_yuv420p(image_path, height, width):
    """
    :param image_path: 待转换的.yuv图像文件路径
    :param rows: 图像行数
    :param cols: 图像列数
    :return: y,u,v分量
    """
    fp = open(image_path, 'rb')#打开二进制文件
    yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
    u = np.zeros(shape=(height//2, width//2), dtype='int8', order='C')
    v = np.zeros(shape=(height//2, width//2), dtype='int8', order='C')
    
    for m in range(height):
        for n in range(width):
            yt[m, n] = ord(fp.read(1))
            #print('YYYYYY',yt[m, n])
            
    print('---------done----------')
    
    for m in range(height//2):
        for n in range(width//2):
            u[m, n] = ord(fp.read(1))
            #print('UUU',u[m, n])
    for m in range(height//2):
        for n in range(width//2):
            v[m, n] = ord(fp.read(1))
            #print('VVV',v[m, n])

    
    
    # yt = yt[np.newaxis,:,:]
   
    print(yt.reshape(-1).shape)
    print(u.reshape(-1).shape)
    print(v.reshape(-1).shape)
    img = np.concatenate((yt.reshape(-1), u.reshape(-1),v.reshape(-1)))
    print(img.shape)
    img = img.reshape((height*3 // 2, width)).astype('uint8') 
    print(img.shape)
   
    return img
    
in_path = './test_out/' 
out_path = './'  
imagelist = os.listdir(in_path)
if not os.path.exists(out_path):
    os.mkdir(out_path)
for i in range(len(imagelist)):
    in_name = os.path.join(in_path, imagelist[i])
   
    image_name, exten = os.path.splitext(imagelist[i])
    out_name = os.path.join(out_path + image_name + '.jpg')
    image = read_yuv420p(in_name, 418, 632)
    bgr_img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    cv2.imwrite(out_name, bgr_img)
    
    
    
