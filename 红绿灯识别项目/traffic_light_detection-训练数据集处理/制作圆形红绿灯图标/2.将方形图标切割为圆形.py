import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# SquareBackground_output_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\SquareBackground'
# CircularBackground_output_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground'
SquareBackground_output_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground\add_test'
CircularBackground_output_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\CircularBackground\buff'




def circle(img_path, times,save_Circular_path):

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

    # cir_file_name = times.replace('.jpg','') + '.png'  # 修改为自己需要的命名格式
    cir_file_name = times
    # cir_file_name = times.replace('.png', '_a.png')   # 修改为自己需要的命名格式

    #  输出路径
    out_put_path = save_Circular_path
    cir_path = out_put_path + '/' + cir_file_name
    imb.save(cir_path)
    return


total_classify_direct_name = os.listdir(SquareBackground_output_path)
print(total_classify_direct_name)

for single_classify_name in total_classify_direct_name:
    #针对需要处理的图片文件夹 进行批量处理
    for root, dirs, files in os.walk(os.path.join(SquareBackground_output_path,single_classify_name)):  # 修改为图片路径

        save_Circular_path = os.path.join(CircularBackground_output_path, single_classify_name) #保存对应圆形图像的路径
        if single_classify_name not in os.listdir(CircularBackground_output_path):
            os.mkdir(save_Circular_path)
        for file in files:
            # 获取文件所属目录
            # 获取文件路径
            # print(os.path.join(root,file))
            circle(os.path.join(root, file), file,save_Circular_path)
# print("----------------完成圆形切割-------------------")