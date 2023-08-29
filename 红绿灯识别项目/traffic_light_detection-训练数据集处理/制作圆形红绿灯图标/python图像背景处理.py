import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 源目录： 初始图片
project_dir = os.path.dirname(os.path.abspath(r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample\\z_test"))
input = os.path.join(project_dir, 'z_test')

# 输出目录 ： 比例缩小图片
output = os.path.join(project_dir, 'z_save')


def modify():
    # 切换目录
    os.chdir(input)

    # 遍历目录下所有的文件
    for image_name in os.listdir(os.getcwd()):
        # 打印图片名字
        # print(image_name)
        im = Image.open(os.path.join(input, image_name))
        # 需要设置rgb 并且将处理过的图片存储在别的变量下
        im = im.convert('RGB')
        # 重新设置大小（可根据需求转换）
        rem = im.resize((320, 320))
        # 对处理完的正方形图片进行保存
        rem.save(os.path.join(output, image_name))


if __name__ == '__main__':
    modify()
    print("\n图片已经改为正方形\n")


def circle(img_path, times):
    path_name = os.path.dirname(img_path)
    cir_file_name = 'cir_img.png'
    cir_path = path_name + '/' + cir_file_name
    ima = Image.open(img_path).convert("RGBA")
    size = ima.size
    # 要使用圆形，所以使用刚才处理好的正方形的图片
    r2 = min(size[0], size[1])
    if size[0] != size[1]:
        ima = ima.resize((r2, r2), Image.ANTIALIAS)
    # 最后生成圆的半径
    r3 = int(r2 / 2)
    imb = Image.new('RGBA', (r3 * 2, r3 * 2), (255, 255, 255, 0))
    pima = ima.load()  # 像素的访问对象
    pimb = imb.load()
    r = float(r2 / 2)  # 圆心横坐标

    for i in range(r2):
        for j in range(r2):
            lx = abs(i - r)  # 到圆心距离的横坐标
            ly = abs(j - r)  # 到圆心距离的纵坐标
            l = (pow(lx, 2) + pow(ly, 2)) ** 0.5  # 三角函数 半径
            if l < r3:
                pimb[i - (r - r3), j - (r - r3)] = pima[i, j]

    cir_file_name = times + '.png'  # 修改为自己需要的命名格式

    #  输出路径
    out_put_path = r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample\\z_out"
    cir_path = out_put_path + '/' + cir_file_name
    imb.save(cir_path)
    return


#  针对需要处理的图片文件夹 进行批量处理
for root, dirs, files in os.walk(r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample\\z_save"):  # 修改为图片路径
    for file in files:
        # 获取文件所属目录
        # 获取文件路径
        # print(os.path.join(root,file))
        circle(os.path.join(root, file), file)
print("----------------完成圆形切割-------------------")

# read_path(r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample\test")



