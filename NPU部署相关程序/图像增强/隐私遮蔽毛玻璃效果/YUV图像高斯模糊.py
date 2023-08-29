import cv2

def skin_smoothing(image):
    image = cv2.imread(image)
    # 把目标转化为YUV格式
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # 把亮度分量进行高斯模糊
    blur_image = cv2.GaussianBlur(yuv_image[:, :, :], (7, 7), 5)
	# # 亮度分量梯度、边缘图像
    # sobel_image = cv2.Sobel(blur_image, cv2.CV_8U, 1, 0, ksize=3)
    # # 加、平权
    # skin_image = cv2.addWeighted(yuv_image[:, :, 0], 1.5, sobel_image, -0.5, 0)
    # yuv_image[:, :, 0] = skin_image
    # 将图像转换为BGR格式
    bgr_image = cv2.cvtColor(blur_image, cv2.COLOR_YUV2BGR)
    return bgr_image

# img = cv2.imread("face.jgp",1)  #读取一张图片，彩色

bgr_image = skin_smoothing("20230518110405.png")
cv2.imshow('dst',bgr_image)
cv2.waitKey()