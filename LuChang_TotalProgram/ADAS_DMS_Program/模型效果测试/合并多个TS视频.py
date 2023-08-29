# 一句话搞定
#ffmpeg -i "concat:F:\DMS_test_vidoe/1.ts|F:\DMS_test_vidoe/2.ts|F:\DMS_test_vidoe/3.ts|F:\DMS_test_vidoe/4.ts|F:\DMS_test_vidoe/5.ts|F:\DMS_test_vidoe/6.ts|F:\DMS_test_vidoe/7.ts|F:\DMS_test_vidoe/8.ts|F:\DMS_test_vidoe/9.ts|F:\DMS_test_vidoe/10.ts|F:\DMS_test_vidoe/11.ts|F:\DMS_test_vidoe/12.ts|F:\DMS_test_vidoe/13.ts|F:\DMS_test_vidoe/14.ts|" -c copy -y F:\DMS_test_vidoe/output.mp4


import os
import datetime

def test(path, save_path):
    file_names = os.listdir(path)
    if 'file_list.txt' in file_names:
        os.remove(path + 'file_list.txt')
    out_file_name = 'output2.mp4'
    while out_file_name in os.listdir(save_path):
        out_file_name = '新' + out_file_name
    f = open(path + 'file_list.txt', 'w+')

    concat_vidoe_file_name_string=""
    for one in file_names:
        f.write("file '" + one + "'\n")
        one_str=path+one+"|"
        concat_vidoe_file_name_string+=one_str
    f.close()
    # print("生成txt文件成功!")
    start = datetime.datetime.now()
    # print('开始合成，初始时间为:', datetime.datetime.now())

    # print(concat_vidoe_file_name_string)
    save_names=save_path+out_file_name
    conver_command="ffmpeg -i \"concat:"+concat_vidoe_file_name_string+"\" -c copy -y "+save_names
    print(conver_command)
    os.system(conver_command)

    # print('合成后的当前时间为：', datetime.datetime.now())
    # print('合成视频完成！用时：' + str(datetime.datetime.now() - start))

path=r"F:\DMS_test_vidoe\\"
save_path=r"F:\DMS_test_vidoe_MP4\\"
test(path, save_path)