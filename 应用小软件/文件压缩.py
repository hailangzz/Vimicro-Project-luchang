import os

password = "123"  # 压缩文件密码
input_file_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_images"  # 待压缩的文件的路径
output_zip_file_path = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\CCPD_rec_images.zip"  # 压缩文件的输出路径
cmd = r'D:\WinRAR\WinRAR.exe a -p%s -ep1 %s %s' % (password, output_zip_file_path, input_file_path)  # password为压缩密码
os.system(cmd)
