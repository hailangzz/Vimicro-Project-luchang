"""
实战场景: 如何压缩一个文件夹
"""

# 导入系统包
import platform
import os
import zipfile

def do_zip_compress(dirpath):
    print("原始文件夹路径：" + dirpath)
    output_name = f"{dirpath}.zip"
    parent_name = os.path.dirname(dirpath)
    print("压缩文件夹目录：", parent_name)
    zip = zipfile.ZipFile(output_name, "w", zipfile.ZIP_DEFLATED)
    # 多层级压缩
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if str(file).startswith("~$"):
                continue
            filepath = os.path.join(root, file)
            print("压缩文件路径：" + filepath)
            writepath = os.path.relpath(filepath, parent_name)
            zip.write(filepath, writepath)
    zip.close()


# 需要先创建文件夹resources
dirpath = r"D:\迅雷下载\AI数据集汇总\汽车拍照检测识别\rec_car_brand_train_database_focus8"
# 压缩文件夹
do_zip_compress(dirpath)

