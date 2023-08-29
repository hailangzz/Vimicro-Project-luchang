import cv2

image = cv2.imread(r'C:\Users\zhangzuo\Pictures\20230606135741.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresholded = cv2.threshold(gray, 127, 255, 0)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空白图像作为绘制背景
drawn_image = image.copy()
cv2.drawContours(drawn_image, contours, -1, (0, 255, 0), 2)

cv2.imshow('Contours', drawn_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
