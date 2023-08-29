import matplotlib.pyplot as plt
# 配置字体路径
plt.rcParams['font.sans-serif'] = ['SimHei']

# 示例数据
x = [0, 0.5, 0.85, 1.2]  # x轴数据
y = [0.5, 0.5, 0.85, 0.85]  # y轴数据

# 创建折线图
plt.plot(x, y, marker='o', linestyle='-', color='k')

# 添加标题和轴标签
plt.title('偏离临界线位置')
plt.xlabel('V（偏离速度）')
plt.ylabel('D（报警临界距离）')

# # 设置坐标轴比例为相等
# plt.axis('equal')
# 设置Y轴的显示范围
plt.ylim(0.3, 1)  # 设置Y轴的上下限

# 添加网格
plt.grid(True)

# 显示图形
plt.show()
