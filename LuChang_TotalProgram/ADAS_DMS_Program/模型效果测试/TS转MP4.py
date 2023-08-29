import os


def merge_ts_video(ts_path, ts_path_):
    all_ts = os.listdir(ts_path)
    # 最好是对all_ts 进行排序处理一下
    # 我这里获取到后直接对ts视频文件进行了排序处理,所以没有加排序操作
    for file in all_ts:
        with open(ts_path + file, 'rb') as f1:  # 读取视频以二进制进行处理
            with open(ts_path_ + "DMS_vidoe.mp4", 'ab') as f2:  # 存储到指定位置,VideoName为变量值
                f2.write(f1.read())
        # os.remove(os.path.join(ts_path, file))  # 将每次处理后的ts视频文件进行删除


merge_ts_video(r"F:\DMS_test_vidoe\\", r"F:\DMS_test_vidoe_MP4\\")
# 函数调用：merge_ts_video
# 参数值：
#       参数1 存放 ts 的路径 VideoPreliminaryStorage
#       参数2 存放 mp4 的路径 VideoFinalStorage
