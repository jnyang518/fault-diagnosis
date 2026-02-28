import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取数据
file_path = 'gearset20_0.csv'
df = pd.read_csv(file_path)
columns = df.columns

# 2. 设置绘图参数
fs = 1.0  # 如果已知采样频率，请修改此处
N = len(df)

# 定义颜色
time_color = 'tab:blue'  # 时域图颜色
freq_color = 'tab:orange'  # 频域图颜色

# 创建画布
fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(15, 4 * len(columns)))
fig.suptitle('Time Domain & Frequency Domain Analysis', fontsize=16)

# 3. 循环绘制每一列
for i, col in enumerate(columns):
    data = df[col].values

    # --- 绘制时域图 (Time Domain) ---
    axes[i, 0].plot(data[:5000], color=time_color)  # 设置颜色
    axes[i, 0].set_title(f'Time Domain: {col} (First 5000 samples)')
    axes[i, 0].set_xlabel('Sample Index')
    axes[i, 0].set_ylabel('Amplitude')
    axes[i, 0].grid(True)

    # --- 绘制频域图 (Frequency Domain) ---
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, d=1 / fs)

    half_N = N // 2
    magnitude = 2.0 / N * np.abs(yf[:half_N])
    freqs = xf[:half_N]

    axes[i, 1].plot(freqs, magnitude, color=freq_color)  # 设置颜色
    axes[i, 1].set_title(f'Frequency Domain: {col}')

    if fs == 1.0:
        axes[i, 1].set_xlabel('Frequency (Normalized)')
    else:
        axes[i, 1].set_xlabel('Frequency (Hz)')

    axes[i, 1].set_ylabel('Magnitude')
    axes[i, 1].grid(True)

# 4. 调整布局并显示
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# plt.savefig('time_freq_analysis_colored.png')