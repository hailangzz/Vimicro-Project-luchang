import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import struct
import torch
from RGB2YUV_conv_RGB2YUV420p_ori import Net
def RGB2YUV(img_rgb, mode = 'YUV422S'):
    #         R            G            B 
    params = [[0.299,      0.587,       0.114],             # Y
              [-0.14713,   -0.28886,    0.436],             # U
              [0.615,      -0.51499,    -0.10001]           # V
              ]             
    fc_RGB2YUV = np.array(params, dtype=np.float32)
    fc_RGB2YUV = np.transpose(fc_RGB2YUV,(1,0))
    YUV = np.matmul(img_rgb, fc_RGB2YUV)
    YUV = np.transpose(YUV,(2,0,1))
    Y = YUV[0,:,:]
    U = YUV[1,:,:]
    V = YUV[2,:,:]

    if 'YUV444' in mode:
        UV = np.concatenate([V[np.newaxis,:,:],U[np.newaxis,:,:]],axis = 0)
        off_UV = np.ones_like(UV)*128.0
        UV = UV + off_UV
        Y = Y.astype(np.uint8)
        UV = UV.astype(np.int8)
    if 'YUV420' in mode or 'YUV422' in mode:#== 'YUV422S' or mode == 'YUV420S' or mode == 'YUV422P' or mode == 'YUV420P':
        U_half = U.reshape(U.shape[0],-1,2)[:,:,0]
        V_half = V.reshape(V.shape[0],-1,2)[:,:,1]
        if 'S' in mode:#mode == 'YUV422S' or mode == 'YUV420S':
            UV  = np.concatenate([U_half[:,:,np.newaxis],V_half[:,:,np.newaxis]], axis = 2)
            UV = UV.reshape(UV.shape[0],-1)
            UV = UV[np.newaxis,:,:]
        if 'P' in mode:# == 'YUV422P'or mode == 'YUV420P':
            UV  = np.concatenate([U_half[np.newaxis,:,:],V_half[np.newaxis,:,:]], axis = 0)
            #print(UV.shape)
        if '420' in mode:# == 'YUV420S' or mode == 'YUV420P':
            UV = UV[:,::2,:]
        off_UV = np.ones_like(UV)*128.0
        UV = UV + off_UV
        Y = Y.astype(np.uint8)
        UV = UV.astype(np.int8)
    return Y,UV

def SaveYUVBinary(YUV, file_name):
    Y, UV = YUV
    #print(Y.shape)
    fw = open(file_name , 'wb')

    height, width = Y.shape
    for h in range(height):
        
        buff = struct.pack('{}B'.format(width), *(Y[h,:].tolist()))
        fw.write(buff)
    channels, height, width = UV.shape
    for c in range(channels):
        for h in range(height):
            
            buff = struct.pack('{}b'.format(width), *(UV[c,h,:].tolist()))
            fw.write(buff)
    fw.close()

device = torch.device("cuda:3")
model_rgb2yuv = Net().to(device)
model_rgb2yuv.load_state_dict(torch.load('RGB2YUV420p_1.pth',map_location=torch.device('cuda:3')))#
model_rgb2yuv = model_rgb2yuv.to(device)
model_rgb2yuv.eval()

filepath = '/home/database/FACE_QUALITY/'  # 数据集目录

pathDir = os.listdir(filepath)
 
Y_channel = 0
U_channel = 0
V_channel = 0
for f in pathDir:
    if 'jan' in f or 'YUV' in f:
        continue
    path=filepath+f+'/images_768/train/'
    jpg_list=os.listdir(path)
    for jpg in jpg_list:
        filename = jpg
        img = cv2.imread(os.path.join(path, filename))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #HWC
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        img = img.unsqueeze(0).to(device)
        img_y,img_uv = model_rgb2yuv(img)
        img_y,img_uv = np.array(img_y.cpu().detach()[0]),np.array(img_uv.cpu().detach()[0])
        
        Y_channel = Y_channel + np.sum(img_y/255.0)
        U_channel = U_channel + np.sum(img_uv[0,:, :]/255.0)
        V_channel = V_channel + np.sum(img_uv[1,:, :]/255.0)
 
num = len(pathDir) * 1280 * 720  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
Y_mean = Y_channel / num
U_mean = U_channel / num/2
V_mean = V_channel / num/4
# Y_mean = 0.429479
# U_mean = 0.001265
# V_mean = -0.005162
print("R_mean is %f, G_mean is %f, B_mean is %f" % (Y_mean, U_mean, V_mean))
Y_channel = 0
U_channel = 0
V_channel = 0
for f in pathDir:
    if 'jan' in f or 'YUV' in f:
        continue
    path=filepath+f+'/images_768/train/'
    jpg_list=os.listdir(path)
    for jpg in jpg_list:
        filename = jpg
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #HWC
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        img = img.unsqueeze(0).to(device)
        
        img_y,img_uv = model_rgb2yuv(img)
        img_y,img_uv = np.array(img_y.cpu().detach()[0]),np.array(img_uv.cpu().detach()[0])
        
        Y_channel = Y_channel + np.sum((img_y/255.0 - Y_mean) ** 2)
        U_channel = U_channel + np.sum((img_uv[0,:, :]/255.0 - U_mean) ** 2)
        V_channel = V_channel + np.sum((img_uv[1,:, :]/255.0 - V_mean) ** 2)
 
R_var = np.sqrt(Y_channel / num)
G_var = np.sqrt(U_channel / num/4)
B_var = np.sqrt(V_channel / num/4)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (Y_mean, U_mean, V_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))

# R_mean is 0.429479, G_mean is 0.001265, B_mean is -0.005162
# R_var is 0.252774, G_var is 0.006632, B_var is 0.006628
