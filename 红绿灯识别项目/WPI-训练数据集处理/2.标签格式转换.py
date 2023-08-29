"""
1.JPEGImages文件夹:改名
2.Annotations文件夹:xml文件 ok!
(1)mat=>txt
(2)txt=>xml
3.ImageSets.Main文件夹:数据集分割成trainval,test
"""
"""
the second step!
"""
import os
from scipy import io
from to_JPEGImages import home
from PIL import Image
import glob  # 文件搜索匹配

TXTPath = home
txt_splits = ['test', 'Frames_GT_wHolder']
WPI_CLASSES = (  # always index 0
    'GA_left', 'GA_right', 'GA_up','GA_up_left','GA_up_right','GC_GA_left', 'GC_GA_right', 'GC', 'RA_left','RC')
# 以下两个参数需根据文件路径修改
ImagePath = r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\WPI Traffic Light Dataset\VOC2007\JPEGImages"   # VOC数据集JPEGImages文件夹路径
XMLPath = r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\WPI Traffic Light Dataset\VOC2007\Annotations"    # VOC数据集Annotations文件夹路径


def read_mat(path, split):
    """
    读取mat文件数据
    :param path:
    :param split:
    :return:返回一个字典,内容为{seq:data}
    """
    matpath = os.path.join(path, split, 'labels')

    temp_file = os.listdir(matpath)

    matfile = []  # 存储mat文件名
    gt_dic = {}
    for file in temp_file:
        if file.endswith(".mat"):
            matfile.append(file)
    # print(matfile)
    for mat in matfile:
        print(mat)
        # if 'RC' in mat[:10]:
        # if 'Frames_GT_wHolder' in
        # print('mat path :',os.path.join(matpath, mat))
        mat_dict = io.loadmat(os.path.join(matpath, mat))  # 加载mat文件 返回字典
        print('mat_dict[GroundTruth].shape:',mat_dict['GroundTruth'].shape)

        gt_data = mat_dict['GroundTruth']
        # print(gt_data)
        gt_data = gt_data.squeeze()  # 去掉维度为1的维度 得到一个np数组列表
        # print(gt_data[0].shape)
        # print(gt_data[0][:4])
        # [object_num, frame_num, 7] => [frame_num, object_num, 7] 把相同图片的目标合并
        object_num = len(gt_data)
        # print(object_num)

        frame_num = 0
        for i in range(object_num):  # 得到图片数量
            temp_num = gt_data[i][-1][-3]
            print('temp_num:',temp_num)
            print('gt_data[i].shape[0]:',gt_data[i].shape[0])

            if temp_num > frame_num:
                frame_num = temp_num
        gt = [[] for i in range(frame_num)]
        for obejct in gt_data:
            for frame in obejct:
                frame = list(frame)
                # print(frame)
                # print(frame[-3] - 1)
                # print(len(gt))
                # gt[frame[-3] - 1].append(frame)
        gt_dic[mat] = gt
    return gt_dic


def create_txt(gt_dic, split, imagepath):
    """
    将字典的数据写入txt文件
    :param gt_dic: 从mat读取的数据
    :param split:
    :param imagepath: 图片存放路径
    :return:
    """
    file_path = os.path.join(home, 'annotation.txt')
    gtdata_txt = open(file_path, 'a')  # 若文件已存在,则将数据写在后面而不是覆盖 --'a'
    for seq, data in gt_dic.items():
        print(seq[:-4])
        for frame_id, objects in enumerate(data):
            gtdata_txt.write('%s_%s_%s' % (split, seq[:-4], str(data[frame_id][0][-3]).rjust(4, '0')))
            for x, y, w, h, _, _, label in objects:
                coordinate = change_coordinate(x, y, w, h)
                label = label - 1
                gtdata_txt.write(" " + ",".join([str(a) for a in coordinate]) + ',' + str(label))
            gtdata_txt.write('\n')
    gtdata_txt.close()


def creat_annotation():
    for split in txt_splits:
        gt_dic = read_mat(home, split)
        # create_txt(gt_dic, split, ImagePath)


def change_coordinate(x, y, w, h):
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def txt2xml(txtpath, xmlpath, imagepath):
    """
    txt => xml
    :param txtpath: annotation.txt路径
    :param xmlpath: xml保存路径
    :param imagepath: 图片路径
    :return:
    """
    # 打开txt文件
    lines = open(txtpath + '/' + 'annotation.txt').read().splitlines()
    gts = {}  # name: ['ob1', 'ob2', ...]
    for line in lines:

        gt = line.split(' ')
        key = gt[0]
        gt.pop(0)
        gts[key] = gt
    # 获得图片名字
    ImagePathList = glob.glob(imagepath + '/*.jpg')  # 字符串列表
    ImageBaseNames = []
    for image in ImagePathList:
        ImageBaseNames.append(os.path.basename(image))
    ImageNames = []  # 无后缀
    for image in ImageBaseNames:
        name, _ = os.path.splitext(image)  # 分开名字和后缀
        ImageNames.append(name)
    for name in ImageNames:
        img = Image.open(imagepath + '/' + name + '.jpg')
        width, height = img.size
        # 打开xml文件
        xml_file = open((xmlpath + '/' + name + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + name + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        gts_data = gts[name]
        for ob_id in range(len(gts_data)):
            gt_data = gts_data[ob_id].split(',')
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + WPI_CLASSES[int(gt_data[-1])] + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + gt_data[0] + '</xmin>\n')
            xml_file.write('            <ymin>' + gt_data[1] + '</ymin>\n')
            xml_file.write('            <xmax>' + gt_data[2] + '</xmax>\n')
            xml_file.write('            <ymax>' + gt_data[3] + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')


if __name__ == '__main__':
    creat_annotation()
    # txt2xml(TXTPath, XMLPath, ImagePath)