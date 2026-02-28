import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
# 加载数据
df = pd.read_csv('大唐天桥山电场齿轮箱数据.csv')

# 提取两段数据
seg1 = df['label_1'].values[0:1024]
seg2 = df['label_1'].values[1024:2048] # 对应 1025 到 2048 点

t1 = np.arange(0, 1024)
t2 = np.arange(1024, 2048)

# CWT 设置
widths = np.arange(1, 128)

# 计算两段的小波变换
cwt1 = sig.cwt(seg1, sig.ricker, widths)
cwt2 = sig.cwt(seg2, sig.ricker, widths)

# 绘图设置
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 200
})

fig, axes = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 2, 1, 2]})

# 第一段 - 时域
axes[0].plot(t1, seg1, color='#1f77b4', linewidth=0.8)
axes[0].set_title("Segment 1: Samples 0 - 1024 (Time Domain)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, linestyle=':', alpha=0.7)

# 第一段 - 小波时频图
im1 = axes[1].imshow(np.abs(cwt1), extent=[0, 1024, 1, 128], cmap='jet', aspect='auto', interpolation='bilinear')
axes[1].set_title("Segment 1: Wavelet Time-Frequency Representation")
axes[1].set_ylabel("Scale")
fig.colorbar(im1, ax=axes[1], label='Magnitude')

# 第二段 - 时域
axes[2].plot(t2, seg2, color='#d62728', linewidth=0.8)
axes[2].set_title("Segment 2: Samples 1025 - 2048 (Time Domain)")
axes[2].set_ylabel("Amplitude")
axes[2].grid(True, linestyle=':', alpha=0.7)

# 第二段 - 小波时频图
im2 = axes[3].imshow(np.abs(cwt2), extent=[1024, 2048, 1, 128], cmap='jet', aspect='auto', interpolation='bilinear')
axes[3].set_title("Segment 2: Wavelet Time-Frequency Representation")
axes[3].set_xlabel("Sample Index")
axes[3].set_ylabel("Scale")
fig.colorbar(im2, ax=axes[3], label='Magnitude')

plt.tight_layout()
plt.show()
# 1. 横轴（X轴）：时间 (Time)
# 含义：代表原始振动信号采集的先后顺序。
#
# 在论文中的作用：TFT 模型将横轴切分为多个 Patch（小块）。每个切片代表轴承在某一特定时刻的“状态快照”。通过观察横轴，你可以看到故障冲击随时间周期性出现的规律。
#
# 2. 纵轴（Y轴）：频率 (Frequency)
# 含义：代表信号中所包含的频率成分，单位通常是赫兹 (Hz)。
#
# 在论文中的作用：
#
# 低频部分通常反映轴承的转速信息。
#
# 高频部分通常包含故障引起的冲击特征（如轴承内圈或外圈受损时发出的高频振荡）。
#
# 纵轴的分辨率越高，意味着模型能越清晰地分辨出故障特征频率及其倍频。
#
# 3. 颜色（颜色深浅/亮度）：能量/幅值 (Magnitude / Energy)
# 含义：代表该时刻、该频率下的信号强度（能量）。
#
# 颜色对应关系（以常见的 'jet' 或 'viridis' 色谱为例）：
#
# 深蓝色/冷色调：代表该频段能量很弱，属于背景噪声或无信号区。
#
# 红色、黄色/暖色调：代表该频段能量很强。