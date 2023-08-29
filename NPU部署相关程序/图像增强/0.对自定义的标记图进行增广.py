import warnings
warnings.filterwarnings("ignore")
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import random
import os
import copy

# 自定义随机生成、扩展新的标注覆盖 mask image
class generate_more_type_image:

    def __init__(self):
        self.new_mask_lable_classify_map={'GC_1':0, 'RC_1':1,'YC_1':2, 'GAF_1':3,'GAL_1':4,'GAR_1':5,'RAL_1':6}
        self.total_new_mask_sample_list=[]
        self.new_mask_sample_path=r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample'
        print('hello world!',self.new_mask_sample_path)
        self.get_total_mask_sample()
        # self.create_new_coco_sample()

        self.save_generate_new_mask_path=r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample+'
        pass

    # 读取全部的new mask 信息列表：
    def get_total_mask_sample(self,):
        single_mask_info={'classfiy_name':'','image_path':''}
        total_classify_name = os.listdir(self.new_mask_sample_path)
        for classify_name in total_classify_name:
            classify_image_path = os.path.join(self.new_mask_sample_path,classify_name)
            # print(classify_image_path)
            for mask_image_name in os.listdir(classify_image_path):
                single_mask_info['classfiy_name'] = classify_name
                single_mask_info['image_path'] = os.path.join(classify_image_path,mask_image_name)
                self.total_new_mask_sample_list.append(copy.deepcopy(single_mask_info))

    def random_generate(self,input_img):

        GaussianBlur_value = random.uniform(0,1.5)
        GaussianNoise_value = (random.randint(0,25),random.randint(0,25))
        Multiply_value = (random.uniform(0.95,1),random.uniform(1,1.05))
        GammaContrast_value = (random.uniform(0.95,1),random.uniform(1,1.05))

        seq = iaa.Sequential([
            # iaa.Crop(px=(0, 16)),  # 从每侧裁剪图像0到16px（随机选择）
            # iaa.Fliplr(0.5),  # 水平翻转图像
            iaa.GaussianBlur(sigma=(0, GaussianBlur_value)),  # 使用0到3.0的sigma模糊图像
            iaa.AdditiveGaussianNoise(GaussianNoise_value[0], GaussianNoise_value[1]), # 10~40的高斯噪点
            # # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # 锐化每个图像，使用介于0（无锐化）和1（完全锐化效果）之间的alpha将结果与原始图像覆盖
            #
            iaa.Multiply(Multiply_value, per_channel=0.9), # 更改图像亮度（原始值的50-150%）。
            iaa.GammaContrast(GammaContrast_value, per_channel=0.9),# 改善或恶化图像的对比度。
        ])

        images_aug = seq.augment_images(input_img)
        image = Image.fromarray(images_aug)
        return image
        # image.save('3.jpg')

    def create_generate_image(self,):
        # for mask_classify_label in self.new_mask_lable_classify_map:
        for single_info in self.total_new_mask_sample_list:
            save_generate_path = os.path.join(self.save_generate_new_mask_path, single_info['classfiy_name'])
            if not os.path.exists(save_generate_path):
                os.mkdir(save_generate_path)

            input_image = single_info['image_path']
            generate_image = self.random_generate(imageio.imread(input_image))
            generate_image.save(os.path.join(save_generate_path,single_info['image_path'].split('\\')[-1].replace('.jpg','g.jpg')))




a = generate_more_type_image()
print(len(a.total_new_mask_sample_list))
a.create_generate_image()

# test_images='./00a0f008-3c67908e.jpg'
# input_img = imageio.imread(test_images)
#
# a=generate_more_type_image()
# a.random_generate(input_img)
#
#
#
# #images_aug = seq.augment_images(imglist)
#
# test_images='./00a0f008-3c67908e.jpg'
# input_img = imageio.imread(test_images)
# #图像增加噪点
# noise=iaa.AdditiveGaussianNoise(10,40)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test10-40.jpg')
#
# noise=iaa.AdditiveGaussianNoise(20,40)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test20-40.jpg')
#
# noise=iaa.AdditiveGaussianNoise(10,10)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test10-10.jpg')
#
# noise=iaa.AdditiveGaussianNoise(10,20)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test10-20.jpg')
#
#
# noise=iaa.AdditiveGaussianNoise(0,0)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test0-0.jpg')
#
# noise=iaa.AdditiveGaussianNoise(100,200)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test100-200.jpg')
#
# noise=iaa.AdditiveGaussianNoise(0.1,0.2)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test0.1-0.2.jpg')
#
# noise=iaa.AdditiveGaussianNoise(30,30)
# input_noise=noise.augment_image(input_img)
#
# image= Image.fromarray(input_noise)
# image.save('test30-30.jpg')
#
