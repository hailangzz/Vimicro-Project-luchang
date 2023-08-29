import cv2

# 读取RGB图像
rgb_image = cv2.imread(r"C:\Users\zhangzuo\Pictures\ribendongjingjietou_11101835.jpg")

# 转换为灰度图像
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# 调整亮度和对比度
alpha = 0.2 # 亮度调整因子
beta = 2  # 对比度调整因子
adjusted_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

# 显示并保存DMS图像
cv2.imshow("DMS Image", adjusted_image)
cv2.waitKey(0)
cv2.imwrite("dms_image.jpg", adjusted_image)
