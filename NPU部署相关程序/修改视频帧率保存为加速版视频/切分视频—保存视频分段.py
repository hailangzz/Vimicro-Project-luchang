from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# 输入视频文件名和输出文件名
input_file = r"F:\DMS_MNN项目\data\video\035_19700101083658-60.mp4"
output_file = r"F:\DMS_MNN项目\data\video\035_19700101083658.mp4"

# 定义要保留的起始时间和结束时间
start_time = 50  # 开始时间，以秒为单位
end_time = 60  # 结束时间，以秒为单位

# 使用ffmpeg_extract_subclip函数切分视频
ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

print("Video has been trimmed and saved as", output_file)
