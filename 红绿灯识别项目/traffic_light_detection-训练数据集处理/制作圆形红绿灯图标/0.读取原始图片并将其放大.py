#将图片按修改时间排序(这样才能与图片的描述一致)，将路径存入列表，以便后面逐个插入图片时调用
import  os

# 放大目标检测标注图片，能更好的挑选图片
project_dir = os.path.abspath(r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample+")
output_dir = os.path.abspath(r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample+plus")

def Amplify_mask_image(project_dir,output_dir):

    for single_classify_direct_name in os.listdir(project_dir):
        save_image_path = os.path.join(output_dir,single_classify_direct_name)
        if single_classify_direct_name not in os.listdir(output_dir):
            os.mkdir(save_image_path)

        path = os.path.join(project_dir,single_classify_direct_name)
        list_p = [path+"\\"+i for i in os.listdir(path)] #获取图片的文件名,并拼接完整路径
        list_p.sort(key=lambda path: os.path.getmtime(path)) #将列表中的文件按其修改时间排序，os.path.getmtime() 函数是获取文件最后修改时间

        #按比例缩小图片尺寸
        from PIL import Image
        for infile in list_p:
            im = Image.open(infile)
            (x,y) = im.size #读取图片尺寸（像素）
            x_s = 320 #定义缩小后的标准宽度
            y_s = int(y * x_s / x) #基于标准宽度计算缩小后的高度
            out = im.resize((x_s,y_s),Image.ANTIALIAS) #改变尺寸，保持图片高品质
            out.save(save_image_path+r'\\{}'.format(infile.split("\\")[-1]))

Amplify_mask_image(project_dir,output_dir)