from PIL import Image, ImageEnhance,ImageFilter

# 打开图像文件
img = Image.open(r'C:\Users\34426\Pictures\q.jpeg')

# 对比度增强
enhancer = ImageEnhance.Contrast(img)
img_contrast = enhancer.enhance(2) # 增强对比度1.5倍

# 亮度增强
enhancer = ImageEnhance.Brightness(img)
img_brightness = enhancer.enhance(2) # 增强亮度1.5倍

# 锐化处理
enhancer = ImageEnhance.Sharpness(img)
img_sharpness = enhancer.enhance(4) # 增强锐度1.5倍

# 高斯模糊
img_gaussian = img.filter(ImageFilter.GaussianBlur(radius=2))

# 保存处理后的图像
img_gaussian.save('example_gaussian.jpg')
# 保存处理后的图像
img_contrast.save(r'C:\Users\34426\Pictures\example_contrast.jpg')
img_brightness.save(r'C:\Users\34426\Pictures\example_brightness.jpg')
img_sharpness.save(r'C:\Users\34426\Pictures\example_sharpness.jpg')
# 保存处理后的图像
img_gaussian.save(r'C:\Users\34426\Pictures\example_gaussian.jpg')