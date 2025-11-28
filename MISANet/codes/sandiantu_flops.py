import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
# 我们的模型数据
our_model_params = 3.49
our_model_f1 = 0.8835
our_model_marker = '*'  # 我们的模型使用三角形表示
our_model_color = 'red'

# 其他模型数据
other_models_data = {
    'FC-EF': (3.57, 0.7199, 'o', 'blue'),
    'FC-Siam-Diff': (4.72, 0.6818, 's', 'green'),
    'FC-Siam-Conc': (5.32, 0.7119, 'D', 'orange'),
    'ChangeNet': (10.91, 0.8572, 'p', 'purple'),
    'DSIFN': (82.26, 0.8149, 'H', 'brown'),
    'BIT': (10.63, 0.8435, 'v', 'pink'),
    'SNUNet': (54.82, 0.8685, '^', 'gray'),
    'ICIFNet': (24.51, 0.8456, 'X', 'cyan'),
    'DMINet': (14.55, 0.8470, 'd', 'magenta'),
    'SAGNet': (12.25, 0.8630, 'P', 'olive'),
    'BASNet': (4.70, 0.8707, 'x', 'black'),
    # 'SFBI-Net': (35.24, 0.8636, 'x', 'black'),
    'ABMFNet': (66.17, 0.8142, '8', 'gold'),
    # 'SEIFNet': (8.37, 0.8665, '8', 'gold'),
}

# 提取其他模型的参数量和F1分数数据
other_params = [data[0] for data in other_models_data.values()]
other_f1_scores = [data[1] for data in other_models_data.values()]
other_markers = [data[2] for data in other_models_data.values()]
other_colors = [data[3] for data in other_models_data.values()]
other_model_names = list(other_models_data.keys())

# 创建散点图
# plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制其他模型数据
for i in range(len(other_params)):
    plt.scatter(other_params[i], other_f1_scores[i], label=other_model_names[i], marker=other_markers[i],
                color=other_colors[i], s=105)

# 绘制我们的模型数据
plt.scatter(our_model_params, our_model_f1, label='Ours', marker=our_model_marker, color=our_model_color, s=200)

# 设置横轴和纵轴刻度范围和间隔
# plt.xticks(range(0, 51, 10))
# plt.xticks([i for i in range(0, 51, 10)])
# plt.yticks([i/100 for i in range(80, 95, 2)])  # 因为F1分数是在0.8到1之间的小数，所以将刻度值除以50以得到正确的刻度
ax.set_xticks([i for i in range(0, 101, 20)])
ax.set_yticks([i/100 for i in range(62, 93, 4)])
ax.set_ylim(0.62, 0.92)
ax.set_xlim(0, 100)
# 添加标签和标题
ax.tick_params(axis='both', labelsize=16)  # Adjust tick label font size
ax.set_xlabel('FLOPs(G)', fontsize=16)
ax.set_ylabel('Accuracy(F1)', fontsize=16)

# Add legend
# 添加图例，并设置为横向结构
plt.legend(fontsize=14, ncol=2, loc='lower right')
# Show grid
plt.grid(True)
# plt.savefig(r'C:\Users\Hongjin Ren\OneDrive\文档\paper2\参数量对比散点图\SYSU.pdf', format='pdf', bbox_inches='tight')
plt.savefig(r'E:\yucetu\keshihua\GZCD_daxiu.png', format='png', bbox_inches='tight')

plt.show()
#


##############################################################
# 学习单位: 南京信息工程大学
# 学   生: 尹皓宇
# 开发时间: 2023/12/26 13:22
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
# 我们的模型数据
our_model_params = 3.49
our_model_f1 = 0.9119
our_model_marker = '*'  # 我们的模型使用三角形表示
our_model_color = 'red'

# 其他模型数据
other_models_data = {
    'FC-EF': (3.57, 0.8317, 'o', 'blue'),
    'FC-Siam-Diff': (4.72, 0.8485, 's', 'green'),
    'FC-Siam-Conc': (5.32, 0.8629, 'D', 'orange'),
    'ChangeNet': (10.91, 0.8919, 'p', 'purple'),
    'DSIFN': (82.26, 0.8834, 'H', 'brown'),
    'BIT': (10.63, 0.8986, 'v', 'pink'),
    'SNUNet': (54.82, 0.8998, '^', 'gray'),
    'ICIFNet': (24.51, 0.8918, 'X', 'cyan'),
    'DMINet': (14.55, 0.8985, 'd', 'magenta'),
    'SAGNet': (12.25, 0.9010, 'P', 'olive'),
    'BASNet':(4.70,0.9069, 'x','black'),
    'ABMFNet': (66.17, 0.8157, '8', 'gold'),
}

# 提取其他模型的参数量和F1分数数据
other_params = [data[0] for data in other_models_data.values()]
other_f1_scores = [data[1] for data in other_models_data.values()]
other_markers = [data[2] for data in other_models_data.values()]
other_colors = [data[3] for data in other_models_data.values()]
other_model_names = list(other_models_data.keys())

# 创建散点图
# plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制其他模型数据
for i in range(len(other_params)):
    plt.scatter(other_params[i], other_f1_scores[i], label=other_model_names[i], marker=other_markers[i],
                color=other_colors[i], s=105)

# 绘制我们的模型数据
plt.scatter(our_model_params, our_model_f1, label='Ours', marker=our_model_marker, color=our_model_color, s=200)

# 设置横轴和纵轴刻度范围和间隔
# plt.xticks(range(0, 51, 10))
# plt.xticks([i for i in range(0, 51, 10)])
# plt.yticks([i/100 for i in range(80, 95, 2)])  # 因为F1分数是在0.8到1之间的小数，所以将刻度值除以50以得到正确的刻度
ax.set_xticks([i for i in range(0, 101, 20)])
ax.set_yticks([i/100 for i in range(78, 94, 2)])
ax.set_ylim(0.78, 0.93)
ax.set_xlim(0, 100)
# 添加标签和标题
ax.tick_params(axis='both', labelsize=16)  # Adjust tick label font size
ax.set_xlabel('FLOPs(G)', fontsize=16)
ax.set_ylabel('Accuracy(F1)', fontsize=16)

# Add legend
# 添加图例，并设置为横向结构
plt.legend(fontsize=14, ncol=2, loc='lower right')
# Show grid
plt.grid(True)
# plt.savefig(r'C:\Users\Hongjin Ren\OneDrive\文档\paper2\参数量对比散点图\SYSU.pdf', format='pdf', bbox_inches='tight')
plt.savefig(r'E:\yucetu\keshihua\GZCD_daxiu.png', format='png', bbox_inches='tight')

plt.show()
##############################################################



import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
# 我们的模型数据
our_model_params = 3.49
our_model_f1 = 0.8225
our_model_marker = '*'  # 我们的模型使用三角形表示
our_model_color = 'red'

# 其他模型数据
other_models_data = {
    'FC-EF': (3.57, 0.7772, 'o', 'blue'),
    'FC-Siam-Diff': (4.72, 0.7106, 's', 'green'),
    'FC-Siam-Conc': (5.32, 0.7818, 'D', 'orange'),
    'ChangeNet': (10.91, 0.7525, 'p', 'purple'),
    'DSIFN': (82.26, 0.8004, 'H', 'brown'),
    'BIT': (10.63, 0.7737, 'v', 'pink'),
    'SNUNet': (54.82, 0.7888, '^', 'gray'),
    'ICIFNet': (24.51, 0.7625, 'X', 'cyan'),
    'DMINet': (14.55, 0.8028, 'd', 'magenta'),
    'SAGNet': (12.25, 0.8187, 'P', 'olive'),
    'BASNet':(4.70,0.8012, 'x','black'),
    'ABMFNet': (66.17, 0.7331, '8', 'gold'),


}

# 提取其他模型的参数量和F1分数数据
other_params = [data[0] for data in other_models_data.values()]
other_f1_scores = [data[1] for data in other_models_data.values()]
other_markers = [data[2] for data in other_models_data.values()]
other_colors = [data[3] for data in other_models_data.values()]
other_model_names = list(other_models_data.keys())

# 创建散点图
# plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制其他模型数据
for i in range(len(other_params)):
    plt.scatter(other_params[i], other_f1_scores[i], label=other_model_names[i], marker=other_markers[i],
                color=other_colors[i], s=105)

# 绘制我们的模型数据
plt.scatter(our_model_params, our_model_f1, label='Ours', marker=our_model_marker, color=our_model_color, s=200)

# 设置横轴和纵轴刻度范围和间隔
# plt.xticks(range(0, 51, 10))
# plt.xticks([i for i in range(0, 51, 10)])
# plt.yticks([i/100 for i in range(80, 95, 2)])  # 因为F1分数是在0.8到1之间的小数，所以将刻度值除以50以得到正确的刻度
ax.set_xticks([i for i in range(0, 101, 20)])
ax.set_yticks([i/100 for i in range(62, 86, 3)])
ax.set_ylim(0.62, 0.85)
ax.set_xlim(0, 100)
# 添加标签和标题
ax.tick_params(axis='both', labelsize=16)  # Adjust tick label font size
ax.set_xlabel('FLOPs(G)', fontsize=16)
ax.set_ylabel('Accuracy(F1)', fontsize=16)

# Add legend
# 添加图例，并设置为横向结构
plt.legend(fontsize=14, ncol=2, loc='lower right')
# Show grid
plt.grid(True)
plt.savefig(r'E:\yucetu\keshihua\GZCD_daxiu.png', format='png', bbox_inches='tight')

plt.show()
#
