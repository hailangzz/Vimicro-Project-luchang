import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 源目录： 初始图片
project_dir = os.path.abspath(r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample+plus")
list = os.listdir(project_dir)
input_directory_list = [os.path.join(project_dir, child_direct_name) for child_direct_name in os.listdir(project_dir) ]

# 输出目录 ： 比例缩小图片
SquareBackground_output_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\SquareBackground'



def modify(input_directory_list,SquareBackground_output_path):
    print(input_directory_list)
    # # 切换目录
    # os.chdir(input)

    # # 遍历目录下所有的文件
    # for image_name in os.listdir(os.getcwd()):
    #     # 打印图片名字
    #     # print(image_name)
    #     im = Image.open(os.path.join(input, image_name))
    #     # 需要设置rgb 并且将处理过的图片存储在别的变量下
    #     im = im.convert('RGB')
    #     # 重新设置大小（可根据需求转换）
    #     rem = im.resize((320, 320))
    #     # 对处理完的正方形图片进行保存
    #     rem.save(os.path.join(output, image_name))

    for single_classify_direct in input_directory_list:
        print(single_classify_direct)
        # 此时创建输出保存正方形图像需要的对应路径
        output_path = SquareBackground_output_path+'\\'+single_classify_direct.split('\\')[-1]
        if single_classify_direct.split('\\')[-1] not in os.listdir(SquareBackground_output_path):
            os.mkdir(output_path)

        for image_name in os.listdir(single_classify_direct):
            im = Image.open(os.path.join(single_classify_direct, image_name))
            # 需要设置rgb 并且将处理过的图片存储在别的变量下
            im = im.convert('RGB')
            # 重新设置大小（可根据需求转换）
            rem = im.resize((320, 320))
            # 对处理完的正方形图片进行保存
            rem.save(os.path.join(output_path, image_name))
            # print(image_name)



if __name__ == '__main__':
    modify(input_directory_list,SquareBackground_output_path)
    print("\n图片已经改为正方形\n")