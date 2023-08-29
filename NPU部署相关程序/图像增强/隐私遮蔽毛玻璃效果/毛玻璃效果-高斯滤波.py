# import cv2
# img = cv2.imread(r'20230518110405.png')
# dst=cv2.medianBlur(img,5)  # 中值滤波
# cv2.imshow("img", dst)
# cv2.waitKey (0)
# cv2.destroyAllWindows()

import cv2
img = cv2.imread(r'20230518110405.png')
dst=cv2.GaussianBlur(img,(7,7),90) # 高斯滤波
cv2.imshow("img", dst)
cv2.waitKey (0)
cv2.destroyAllWindows()