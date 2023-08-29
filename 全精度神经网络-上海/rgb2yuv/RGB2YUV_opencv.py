import numpy as np
import cv2
import struct
import os

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
        print('YYYYYY',Y[h,:])
        buff = struct.pack('{}B'.format(width), *(Y[h,:].tolist()))
        fw.write(buff)
    channels, height, width = UV.shape
    for c in range(channels):
        for h in range(height):
            print('UVUVUV',UV[c,h,:])
            buff = struct.pack('{}b'.format(width), *(UV[c,h,:].tolist()))
            fw.write(buff)
    fw.close()

def YUV422SP2RGB():
    pass

in_path = '../figure/' 
out_path = 'test_out/'  
imagelist = os.listdir(in_path)
if not os.path.exists(out_path):
    os.mkdir(out_path)
for i in range(len(imagelist)):
    in_name = os.path.join(in_path, imagelist[i])
    print(in_name)
    image_name, exten = os.path.splitext(imagelist[i])
    out_name = os.path.join(out_path + image_name + '.yuv')
    #out_name = os.path.join(in_path, imagelist[i])
    img = cv2.imread(in_name)
    print(img.shape)
    h,w,_=img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
    cv2.imwrite('test.png',img_yuv)
    
    Y=img_yuv[:h,:]
    U=img_yuv[h:h*5//4,:w]
    V=img_yuv[h*5//4:img_yuv.shape[0],:w]
    print(Y.shape,U.shape,V.shape)
    for i in range(h):
        print('*****',Y[i,:])
    for i in range(h//4):
        print('UUUUUUUUUU',U[i,:])
    for i in range(h//4):
        print('VVVVVVVVVV',V[i,:])
    #img_yuv = RGB2YUV(img,mode = 'YUV420P')
    #print(img_yuv.shape)
    #SaveYUVBinary(img_yuv, out_name)

