import cv2

# 读取图像

img = cv2.imread(r'C:\Users\zhangzuo\Pictures\zhedang20.png')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 找到轮廓
# contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# ret, thresholded = cv2.threshold(gray, 50, 50, 0)
# ret, thresholded = cv2.threshold(gray, 60, 60, 0)
# contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ret, thresholded = cv2.threshold(gray, 25, 255, 0)
contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算轮廓的总面积
total_area = 0
total_point = 0
for c in contours:
    print(c.shape)
    total_area += cv2.contourArea(c)
    total_point+=c.shape[0]
# 计算图像的总面积

img_area = img.shape[0] * img.shape[1]

# 计算被遮挡的比例

percentage = total_area / img_area

print("图像遮挡百分比：",percentage)
# 如果超过50%，则排除掉

if percentage > 0.5:
    print('图像被遮挡，排除掉')

print("total_point:",total_point)
print("total_point rate:",total_point/(img.shape[0]+img.shape[1])/2)
# 绘制轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




