import os
import shutil


trainval_seqs = ['GA_left_IMG_2940_3086_3096',
                 'GA_right_2942_2951',
                 'GA_up_IMG_2951_3087_3096',
                 'GA_up_left_IMG_2940_3086_2942',
                 'GA_up_right_IMG_2940_2949_2951_3086',
                 'GC_GA_left_IMG_2951_3096',
                 'GC_GA_right_IMG_2951',
                 'GC_IMG_2949_2940',
                 'RA_left_IMG_3096',
                 'RC_IMG_2949'
                 ]
test_seqs = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8', 'seq9', 'seq10',
             'seq11', 'seq12', 'seq13', 'seq14', 'seq15', 'seq16', 'seq17']
splits = {'test': test_seqs,
          'Frames_GT_wHolder': trainval_seqs
          }

# 以下两个参数需根据文件路径修改
newpath = r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\WPI Traffic Light Dataset\VOC2007\JPEGImages"  # VOC数据集JPEGImages文件夹路径
home = r"D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\WPI Traffic Light Dataset"  # wpi数据集路径


def file_rename(path, file_name, new_name):
    """
    修改文件名字
    :param path: 文件所在文件夹路径
    :param file_name: 文件名
    :param new_name: 修改后的文件名
    :return:
    """
    file = os.path.join(path, file_name)
    dirname, filename = os.path.split(file)  # 分离文件路径和文件名(不含后缀)
    new_file = os.path.join(dirname, new_name)
    os.rename(file, new_file)


def file_move(path, file_name, new_path):
    """
    移动文件到指定文件夹
    :param path: 文件路径
    :param file_name: 文件名
    :param new_path: 指定文件夹路径
    :return:
    """
    file = os.path.join(path, file_name)
    shutil.move(file, new_path)


def all_rename_and_move():
    for split, seqs in splits.items():
        print(split,seqs)
        if split == 'Frames_GT_wHolder':
            # print(seqs)
            for seq in seqs:
                path = os.path.join(home, split, seq)  # 文件夹路径
                temp_file = os.listdir(path)  # 文件名列表
                total_frame = []  # 取出其中jpg文件
                for file in temp_file:
                    if file.endswith(".jpg"):
                        total_frame.append(file)

                for file in total_frame:
                    file_rename(path, file, '%s_%s_%s' % (split, seq, file[-8:]))
                    file_move(path, '%s_%s_%s' % (split, seq, file[-8:]), newpath)
        else:
            for seq in seqs:
                path = os.path.join(home, split, seq)  # 文件夹路径
                temp_file = os.listdir(path)  # 文件名列表
                total_frame = []  # 取出其中jpg文件
                for file in temp_file:
                    if file.endswith(".jpg"):
                        total_frame.append(file)

                for file in total_frame:
                    file_rename(path, file, '%s_%s_%s' % (split, seq, file[-8:]))
                    file_move(path, '%s_%s_%s' % (split, seq, file[-8:]), newpath)


if __name__ == '__main__':
    all_rename_and_move()
