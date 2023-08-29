# _*_ coding:utf-8 _*_
import time
import cv2
import os


import cv2
import os

def getFrame(videoPath, svPath):
    #读取视频
    cap = cv2.VideoCapture(videoPath)
    print(type(cap))
    numFrame = 0
    fps = cap.get(cv2.CAP_PROP_FPS) #获取视频的帧率
    print(fps)
    while True:
        if cap.grab(): # 用来指向下一帧
            '''
            flag:按帧读取视频，返回值ret是布尔型，正确读取则返回True
            frame:为每一帧的图像
            '''
            flag, frame = cap.retrieve()  #解码,并返回捕获的视频帧    
            if not flag:
                continue
            else:
                # cv2.imshow('video', frame)
                numFrame += 1
                #拼接图片保存路径
                newPath = svPath + "\\图片" + str(numFrame) + ".jpg"
                #将图片按照设置格式，保存到文件
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
                if numFrame == 1000:
                    #只保存10张图片，程序结束
                    cap.release() #释放读取画面状态
                    break
                else:
                    pass

        # # 注：等待接收键盘值并退出
        # if cv2.waitKey(10) == 27:
        #     break

if __name__ == '__main__':
    videopath = r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\Vimicro_Vidoe_dataset\20221108_080724_NF.mp4'  #自行更改路径
    svpath   = r'D:\迅雷下载\AI数据集汇总\Vimicro_DVR_Dataset\Vimicro_Image_dataset'                       #图片保存路径
    if os.path.exists(svpath):
        pass
    else:
        os.mkdir(svpath)   
    getFrame(videopath,svpath)