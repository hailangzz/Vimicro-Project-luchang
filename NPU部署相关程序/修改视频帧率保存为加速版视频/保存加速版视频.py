from moviepy.editor import *
# intp_name = r'G:\\DL\\unet-keras\\unet-keras-master\\video_test\\stone_ture.mp4'
# outp_name= r'G:\\DL\\unet-keras\\unet-keras-master\\video_test\\stone_ture15.mp4'
intp_name = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\夜间驾驶样本\xi_an\videoplayback (1).mp4'
outp_name = r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\夜间驾驶样本\xi_an\videoplayback (1)_quick.mp4'
play_speed = 4 #速率(播放时长变为原来的二分之一)
au = VideoFileClip(intp_name)
new_au = au.fl_time(lambda t:  play_speed*t, apply_to=['mask', 'audio'])
new_au = new_au.set_duration(au.duration/play_speed)
new_au.write_videofile(outp_name)

