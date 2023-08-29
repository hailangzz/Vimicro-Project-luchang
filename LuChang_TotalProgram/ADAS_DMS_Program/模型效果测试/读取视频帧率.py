import cv2

# 打开视频文件
cap = cv2.VideoCapture(r"F:\DMS_test_vidoe_MP4\DMS_vidoe.mp4")

# 获取视频的帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 输出视频的帧数
print(f"The video contains {frame_count} frames.")

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
# 输出视频的帧率
print(f"The video contains {fps} Hz.")

# 释放视频对象
cap.release()



