import cv2
import numpy as np

rgb_image = cv2.imread(r"C:\Users\zhangzuo\Pictures\20230606142212.jpg")

red_channel = rgb_image[:, :, 2]  # 提取红色通道
blue_channel = rgb_image[:, :, 0]  # 提取蓝色通道

infrared_image = np.zeros_like(rgb_image)  # 创建与原始图像相同大小的空白图像
infrared_image[:, :, 2] = red_channel  # 将红色通道赋值给红外图像的红色通道
infrared_image[:, :, 0] = blue_channel  # 将蓝色通道赋值给红外图像的蓝色通道


# 调整亮度和对比度
alpha = 1.2  # 亮度调整因子
beta = 30  # 对比度调整因子
infrared_image = cv2.convertScaleAbs(infrared_image, alpha=alpha, beta=beta)
# 转换为灰度图像
gray_image = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2GRAY)
# 调整亮度和对比度
alpha = 0.8  # 亮度调整因子
beta = 3  # 对比度调整因子
adjusted_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
# cv2.imshow("Infrared Image", infrared_image)
# cv2.waitKey(0)
# cv2.imwrite("infrared_image.jpg", infrared_image)
cv2.imshow("Infrared Image", adjusted_image)
cv2.waitKey(0)
cv2.imwrite("infrared_image.jpg", adjusted_image)
