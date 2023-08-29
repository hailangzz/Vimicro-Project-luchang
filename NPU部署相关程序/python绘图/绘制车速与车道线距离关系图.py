from numpy import sin
import matplotlib.pyplot as plt
# 配置字体路径
plt.rcParams['font.sans-serif'] = ['SimHei']

speed_rate = 1000/3600*sin(0.08727)/3
# 示例数据
x = [0, 60, 70, 80 ,90,140]  # x轴数据
y = [60*speed_rate, 60*speed_rate, 70*speed_rate, 80*speed_rate,90*speed_rate,90*speed_rate]  # y轴数据

# 创建折线图
plt.plot(x, y, marker='o', linestyle='-', color='k')

# 添加标题和轴标签
plt.title('偏离临界线位置')
plt.xlabel('V（汽车行驶速度：km/h）')
plt.ylabel('D（报警临界距离：m）')

# # 设置坐标轴比例为相等
# plt.axis('equal')
# 设置Y轴的显示范围
plt.ylim(0.4, 0.8)  # 设置Y轴的上下限
print(y)
print(16.67 * sin(0.08727))
print(speed_rate)
# 添加网格
plt.grid(True)

# 显示图形
plt.show()
