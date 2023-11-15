import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df1 = pd.read_csv('Results/CD/results_lr0.01.csv')
df2 = pd.read_csv('Results/CD/results_lr0.001.csv')
df3 = pd.read_csv('Results/CD/results_lr0.05.csv')
df4 = pd.read_csv('Results/CD/results_lr0.005.csv')

# 创建一个2x2的图形网格
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# 在第一个子图上绘制 train_loss
axs[0, 0].plot(df1['epoch'], df1['train_loss'], label='1e-2')
axs[0, 0].plot(df2['epoch'], df2['train_loss'], label='1e-3')
axs[0, 0].plot(df3['epoch'], df3['train_loss'], label='5e-2')
axs[0, 0].plot(df4['epoch'], df4['train_loss'], label='5e-3')
axs[0, 0].set_title('Train Loss')
axs[0, 0].legend()

# 在第二个子图上绘制 train_acc
axs[0, 1].plot(df1['epoch'], df1['train_acc'], label='1e-2')
axs[0, 1].plot(df2['epoch'], df2['train_acc'], label='1e-3')
axs[0, 1].plot(df3['epoch'], df3['train_acc'], label='5e-2')
axs[0, 1].plot(df4['epoch'], df4['train_acc'], label='5e-3')
axs[0, 1].set_title('Train Accuracy')
axs[0, 1].legend()

# 在第三个子图上绘制 test_loss
axs[1, 0].plot(df1['epoch'], df1['test_loss'], label='1e-2')
axs[1, 0].plot(df2['epoch'], df2['test_loss'], label='1e-3')
axs[1, 0].plot(df3['epoch'], df3['test_loss'], label='5e-2')
axs[1, 0].plot(df4['epoch'], df4['test_loss'], label='5e-3')
axs[1, 0].set_title('Test Loss')
axs[1, 0].legend()

# 在第四个子图上绘制 test_acc
axs[1, 1].plot(df1['epoch'], df1['test_acc'], label='1e-2')
axs[1, 1].plot(df2['epoch'], df2['test_acc'], label='1e-3')
axs[1, 1].plot(df3['epoch'], df3['test_acc'], label='5e-2')
axs[1, 1].plot(df4['epoch'], df4['test_acc'], label='5e-3')
axs[1, 1].set_title('Test Accuracy')
axs[1, 1].legend()

# 添加X轴和Y轴的标签
for ax in axs.flat:
    ax.set(xlabel='epoch', ylabel='value')

# 保存图片
plt.savefig('Results/CD/figure.png')

# 显示图形
plt.tight_layout()
plt.show()