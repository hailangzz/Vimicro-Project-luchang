import os
import cv2
from tqdm import tqdm

class cut_image_from_vidoe():

    def __init__(self):
        self.origin_vidoe_path = r"F:\DMS_test_vidoe_MP4\mp4"
        self.total_vidoe_path_list = os.listdir(self.origin_vidoe_path)
        self.save_path = r'F:\DMS_test_vidoe_MP4\frame_image_DMS384'
        # print(self.total_vidoe_path_list)


    def getFrame(self,frame_interval):
        print(self.origin_vidoe_path)
        print(self.total_vidoe_path_list)
        # for single_vidoe  in self.total_vidoe_path_list:
        for vidoe_index in tqdm(range(len(self.total_vidoe_path_list))):

            single_vidoe = self.total_vidoe_path_list[vidoe_index]
            print(single_vidoe)
            # 读取视频
            cap = cv2.VideoCapture(os.path.join(self.origin_vidoe_path,single_vidoe))
            numFrame = 0
            fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
            # frame_interval = int(fps/2) # 根据视频帧计算每秒取两针
            frame_interval = 1  # 逐帧切分
            # # print(fps) # 30，为每秒30帧:

            images_number=0
            while True:
                if cap.grab(): # 用来指向下一帧
                    '''
                    flag:按帧读取视频，返回值ret是布尔型，正确读取则返回True
                    frame:为每一帧的图像
                    '''
                    numFrame += 1
                    # print(numFrame)
                    if numFrame%frame_interval==0: # 当达到间隔时取图像帧
                        images_number += 1
                        flag, frame = cap.retrieve()  # 解码,并返回捕获的视频帧
                        if not flag:
                            continue
                        else:
                            # cv2.imshow('video', frame) # 显示视频帧图像

                            # 拼接图片保存路径
                            # newPath = self.save_path + "\\" +single_vidoe.split('.TS')[0]+ "_"+ str(numFrame) + ".jpg"
                            newPath = self.save_path + "\\" + str(images_number) + ".jpg"

                            # frame = frame.resize((384, 384))  # 设置需要转换的帧图片的大小
                            dim = (384, 384)
                            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                            # 将图片按照设置格式，保存到文件
                            cv2.imencode('.jpg', resized)[1].tofile(newPath)

                # cap.release()  # 释放读取画面状态
                else:
                    break


if __name__ == '__main__':
    cut_image = cut_image_from_vidoe()

    cut_image.getFrame(15)
