import os
import cv2
import numpy as np

img_dir = r"D:\Git_WareHouse\yolov5\test_car_brand"
save_dir = r'D:\Git_WareHouse\yolov5\test_car_brand_reshape'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
 
TARGET_WIDTH = 1024
TARGET_HEIGHT = 576
 
f = open('imglist.txt', 'w')


img_list = os.listdir(img_dir)
for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    img_bgr = cv2.imread(img_path) 
    src_h,src_w,src_c=img_bgr.shape
    sx = float(TARGET_WIDTH)/src_w
    sy = float(TARGET_HEIGHT)/src_h
    r=min(sx,sy)
    pad_im=np.zeros((TARGET_HEIGHT,TARGET_WIDTH,3),dtype=np.uint8)
    resize_im=cv2.resize(img_bgr,None,None,fx=r,fy=r)
    off_x=(pad_im.shape[1]-resize_im.shape[1])//2
    off_y=(pad_im.shape[0]-resize_im.shape[0])//2
    pad_im[off_y:off_y+resize_im.shape[0],off_x:off_x+resize_im.shape[1],:]=resize_im[:,:,:]
    
    
    
    f.write(img_name + '\n')
    save_path = os.path.join(save_dir, img_name)
    print(save_path)
    cv2.imwrite(save_path, pad_im)
    

    
    #r=TARGET_WIDTH/max(src_h,src_w)
    #resize_im=cv2.resize(rgb_im,None,None,fx=r,fy=r)
    #pad_im=np.zeros((TARGET_HEIGHT,TARGET_WIDTH,3)).astype(np.uint8)
    #pad_im[0:resize_im.shape[0],0:resize_im.shape[1],:]=resize_im[:,:,:] 
    
    #src_h,src_w,src_c=im.shape
    # sx = float(TARGET_WIDTH)/src_w
    # sy = float(TARGET_HEIGHT)/src_h
    # r=min(sx,sy)
    # pad_im=np.zeros((TARGET_HEIGHT,TARGET_WIDTH,3),dtype=np.uint8)
    # resize_im=cv2.resize(im,None,None,fx=r,fy=r)
    #
    # off_x=(pad_im.shape[1]-resize_im.shape[1])//2
    # off_y=(pad_im.shape[0]-resize_im.shape[0])//2
    #
    # pad_im[off_y:off_y+resize_im.shape[0],off_x:off_x+resize_im.shape[1],:]=resize_im[:,:,:]