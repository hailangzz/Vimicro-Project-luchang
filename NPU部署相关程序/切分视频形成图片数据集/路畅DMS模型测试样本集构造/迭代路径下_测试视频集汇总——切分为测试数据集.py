import os
import cv2
from tqdm import tqdm

def search_dir(path):

    files= os.listdir(path) # 得到文件夹下的所有文件名称
    # print(files)
    for file in files: # 遍历该文件夹
        if os.path.isdir(path+"\\"+file): # 是子文件夹
            search_dir(path+"\\"+file)
        else: # 是文件
            single_video_path=path+"\\"+file
            total_video_path.append(single_video_path)

            # print(single_video_path)
    # print(total_video_path)
    # return total_video_path


class cut_image_from_vidoe():

    def __init__(self,total_vidoe_path_list,save_path = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\ADASTestDatabase\夜间ADAS测试样本\xi_an'):

        self.total_vidoe_path_list = total_vidoe_path_list
        self.save_path = save_path
        # print(self.total_vidoe_path_list)


    def getFrame(self,frame_interval):

        print(self.total_vidoe_path_list)
        # for single_vidoe  in self.total_vidoe_path_list:
        images_number = 0
        for vidoe_index in tqdm(range(len(self.total_vidoe_path_list))):

            single_vidoe = self.total_vidoe_path_list[vidoe_index]
            print(single_vidoe)
            # 读取视频
            cap = cv2.VideoCapture(single_vidoe)
            numFrame = 0
            fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
            frame_interval = int(fps/3) # 根据视频帧计算每秒取三针
            # # print(fps) # 30，为每秒30帧:


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
                            # 将图片按照设置格式，保存到文件
                            cv2.imencode('.jpg', frame)[1].tofile(newPath)

                # cap.release()  # 释放读取画面状态
                else:
                    break


if __name__ == '__main__':
    path = r"G:\路畅_DMS_ADAS数据集\20230525"  # 文件夹目录
    total_video_path = []
    search_dir(path) #递归读取路径下视频测试文件
    # print(len(total_video_path))
    save_path = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\LuChang'

    cut_image = cut_image_from_vidoe(total_video_path,save_path)
    cut_image.getFrame(15)





