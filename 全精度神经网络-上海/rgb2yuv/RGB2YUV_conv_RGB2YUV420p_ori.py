#coding=utf-8

from tokenize import group
import cv2
import numpy as np
import struct
import glob
import os

import torch
import torch.nn as nn


class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      self.conv_rgbToyuv444 = nn.Conv2d(3, 3, 1, 1, 0, bias=True) # in_channels, out_channels, kernel_size, stride, padding         
      self.conv_420Y = nn.Conv2d(3, 1, 1, 1, 0, bias=False)
      self.conv_420UV = nn.Conv2d(3, 2, 2, 2, 0, bias=False)
   def forward(self, img):
      out = self.conv_rgbToyuv444(img)
      #print(out.size())
      Y = self.conv_420Y(out)
      UV = self.conv_420UV(out)
      #print(UV.size())
      return Y, UV

device = torch.device("cuda:0")
net = Net()
net = net.to(device)

def save_model(model, filename):
   state = model.state_dict()
   for key in state: state[key] = state[key].clone().cpu()
   torch.save(state, filename)

def SaveYUVBinary(Y, UV, file_name):
   fw = open(file_name , 'wb')
   channel, height, width = Y.shape[0], Y.shape[1], Y.shape[2]
   for h in range(height):
      for w in range(width):
         buff = struct.pack('B', Y[0,h,w])
         fw.write(buff)
   channel, height, width = UV.shape[0], UV.shape[1], UV.shape[2]
   for c in range(channel):
        for h in range(height):
            for w in range(width):
               buff = struct.pack('b',UV[c,h,w])
               fw.write(buff)
   fw.close()



pre_dict = net.state_dict()
for k, v in pre_dict.items():
   #print(k, v.shape)
   if k == "conv_rgbToyuv444.weight":
      pre_dict[k][0,0,0,0] =  0.299    # Y
      pre_dict[k][0,1,0,0] =  0.587    # Y
      pre_dict[k][0,2,0,0] =  0.114    # Y
      pre_dict[k][1,0,0,0] = -0.147    # U
      pre_dict[k][1,1,0,0] = -0.289    # U
      pre_dict[k][1,2,0,0] =  0.436    # U
      pre_dict[k][2,0,0,0] =  0.615    # V
      pre_dict[k][2,1,0,0] = -0.515    # V
      pre_dict[k][2,2,0,0] = -0.100    # V
   if k == "conv_rgbToyuv444.bias":
      pre_dict[k][0] = 0      # Y
      pre_dict[k][1] = 128    # U
      pre_dict[k][2] = 128    # V
   if k == "conv_420Y.weight":  #1311
      pre_dict[k][0,0,0,0] = 1    # Y
      pre_dict[k][0,1,0,0] = 0    # Y
      pre_dict[k][0,2,0,0] = 0    # Y
   if k == "conv_420UV.weight":  #2322
      pre_dict[k][0,0,0,0] = 0    # Y
      pre_dict[k][0,0,0,1] = 0    # Y
      pre_dict[k][0,0,1,0] = 0    # Y
      pre_dict[k][0,0,1,1] = 0    # Y

      pre_dict[k][0,1,0,0] = 1    # U
      pre_dict[k][0,1,0,1] = 0    # U
      pre_dict[k][0,1,1,0] = 0    # U
      pre_dict[k][0,1,1,1] = 0    # U

      pre_dict[k][0,2,0,0] = 0    # V
      pre_dict[k][0,2,0,1] = 0    # V
      pre_dict[k][0,2,1,0] = 0    # V
      pre_dict[k][0,2,1,1] = 0    # V

      pre_dict[k][1,0,0,0] = 0    # Y
      pre_dict[k][1,0,0,1] = 0    # Y
      pre_dict[k][1,0,1,0] = 0    # Y
      pre_dict[k][1,0,1,1] = 0    # Y

      pre_dict[k][1,1,0,0] = 0    # U
      pre_dict[k][1,1,0,1] = 0    # U
      pre_dict[k][1,1,1,0] = 0    # U
      pre_dict[k][1,1,1,1] = 0    # U

      pre_dict[k][1,2,0,0] = 1    # V
      pre_dict[k][1,2,0,1] = 0    # V
      pre_dict[k][1,2,1,0] = 0    # V
      pre_dict[k][1,2,1,1] = 0    # V


# net.load_state_dict(pre_dict)
# save_model(net, 'RGB2YUV420p_1.pth')

# in_file = '../figure/pose.png'
# img = cv2.imread(in_file)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #HWC
# img = img.transpose((2,0,1))
# img = torch.from_numpy(img)
# img = img.type(torch.FloatTensor)
# img = img.unsqueeze(0).to(device)
# print(img.size())
# pred_Y, pred_UV = net(img)
# pred_Y = pred_Y.cpu().detach()
# pred_Y = np.array(pred_Y[0])
# pred_UV = pred_UV.cpu().detach()
# pred_UV = np.array(pred_UV[0])
# print("**********pred_Y************")
# print(pred_Y)
# print("**********pred_UV************")
# print(pred_UV.shape)
# Y = pred_Y.astype(np.uint8)
# UV = pred_UV.astype(np.int8)
# print("**********Y************")
# print(Y)
# print("**********UV************")
# print(UV)
# yuv_file = 'test_yuv420p_1.yuv'
# SaveYUVBinary(Y, UV, yuv_file)