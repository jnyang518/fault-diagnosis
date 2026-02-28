import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pywt  # 引入小波变换库
import cv2  # 用于图像缩放 (可选，安装 opencv-python)

# ================= 配置参数 =================
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'
TOTAL_LEN = 10240 # 一个完整样本的总长度
NUM_SEGMENTS = 10  # 将总长度切分为 10 段 (对应 10 张图)
SEGMENT_LEN = 1024  # 每段长度 (10240 / 10)
IMG_SIZE = (64, 64)  # 小波变换后的图像压缩尺寸 (Height, Width)
BATCH_SIZE = 16  # 批次大小 (图像显存占用大，建议调小)
EPOCHS = 20
LR = 0.0005
D_MODEL = 128  # CNN 提取出的特征维度，也是 Transformer 的输入维度
NUM_CLASSES = 4
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 小波变换函数 (CWT)
def signal_to_cwt_image(signal_segment):
    """
    将 1D 信号段转换为 2D 时频图 (CWT)
    """
    # 选取小波基 (例如莫尔小波 'morl' 或 'cmor')
    wavelet = 'morl'
    # 定义尺度 (决定了图像的高度，这里生成 64 个频带)
    scales = np.arange(1, 65)

    # 进行连续小波变换
    # coefs 形状: (scales, signal_length) -> (64, 1024)
    coefs, freqs = pywt.cwt(signal_segment, scales, wavelet)

    # 取绝对值作为幅值
    cwt_img = np.abs(coefs)

    # [重要] 归一化到 0-1 之间，方便神经网络训练
    cwt_img = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-6)

    # [可选] 缩放图像以减少计算量
    # 将 (64, 1024) 缩放到 (64, 64)
    cwt_img = cv2.resize(cwt_img, IMG_SIZE)

    return cwt_img


# 2. 数据集类 (Dataset)
class CWTSequenceDataset(Dataset):
    def __init__(self, X_flat, y, segment_len, num_segments):
        """
        X_flat: 原始的 1D 数据数组 (N_samples, Total_Len)
        """
        self.X = X_flat
        self.y = y
        self.segment_len = segment_len
        self.num_segments = num_segments

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 获取一条长数据 (10240,)
        raw_signal = self.X[idx]

        image_sequence = []

        # 将 10240 切分为 10 段，每段生成一张图
        for i in range(self.num_segments):
            start = i * self.segment_len
            end = start + self.segment_len
            segment = raw_signal[start:end]

            # 生成 CWT 图像 (64, 64)
            img = signal_to_cwt_image(segment)
            image_sequence.append(img)

        # 堆叠成 (10, 64, 64)
        image_sequence = np.array(image_sequence)

        # 增加 Channel 维度 -> (10, 1, 64, 64) (Sequence, Channel, H, W)
        image_sequence = image_sequence[:, np.newaxis, :, :]

        # 转为 Tensor
        return torch.FloatTensor(image_sequence), torch.LongTensor([self.y[idx]]).squeeze()

#(Batch, 10, 1, 64, 64)
#意思是：一个批次有 Batch 个样本，每个样本有 10 张图，每张图是单通道（灰度），分辨率 64x64。
# 3. 数据加载与预处理
def prepare_data(file_path, total_len):
    df = pd.read_csv(file_path)
    X_data = []
    y_data = []

    for i, col in enumerate(df.columns):
        series = df[col].values
        num_samples = len(series) // total_len
        series = series[:num_samples * total_len]
        # Reshape 为 (样本数, 10240)
        X_data.append(series.reshape(-1, total_len))
        y_data.append(np.full(num_samples, i))

    X = np.concatenate(X_data, axis=0)
    y = np.concatenate(y_data, axis=0)
    return X, y


print("正在处理原始数据...")
# 注意：这里我们先不转图像，为了节省内存，在 Dataset 的 __getitem__ 中动态转
raw_X, raw_y = prepare_data(FILE_PATH, TOTAL_LEN)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y, test_size=0.2, stratify=raw_y, random_state=42)

# 创建 Dataset 和 DataLoader
train_ds = CWTSequenceDataset(X_train, y_train, SEGMENT_LEN, NUM_SEGMENTS)
test_ds = CWTSequenceDataset(X_test, y_test, SEGMENT_LEN, NUM_SEGMENTS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# 4. 定义混合模型: CNN提取特征 -> Transformer分析序列
class CNNFeatureExtractor(nn.Module):
    """
    这是一个微型 CNN，用于从单张 CWT 图像中提取特征向量。
    输入: (Batch, 1, 64, 64)
    输出: (Batch, d_model)
    """

    def __init__(self, d_model):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            # Conv 1: 1x64x64 -> 16x32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 2: 16x32x32 -> 32x16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 3: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 展平后的维度: 64 * 8 * 8 = 4096
        self.fc = nn.Linear(64 * 8 * 8, d_model)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class CWTTransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, nhead=4, num_layers=2):
        super(CWTTransformerModel, self).__init__()

        # 1. 特征提取器 (处理每一张图片)
        self.cnn_extractor = CNNFeatureExtractor(d_model)

        # 2. 位置编码 (用于标记10张图片的顺序)
        self.pos_embedding = nn.Parameter(torch.randn(1, NUM_SEGMENTS, d_model))

        # 3. Transformer Encoder (分析图片序列)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 输入 x 形状: (Batch, Seq_Len=10, Channel=1, H, W)
        batch_size, seq_len, c, h, w = x.shape

        # [关键步骤]: 将 Batch 和 Seq 维度合并，以便 CNN 并行处理所有图片
        # view -> (Batch * 10, 1, 64, 64)
        x = x.view(batch_size * seq_len, c, h, w)

        # CNN 提取特征
        # features -> (Batch * 10, d_model)
        features = self.cnn_extractor(x)

        # 还原维度，准备喂给 Transformer
        # features -> (Batch, 10, d_model)
        features = features.view(batch_size, seq_len, -1)

        # 加上位置编码 (广播机制)
        features = features + self.pos_embedding

        # Transformer 序列建模
        # output -> (Batch, 10, d_model)
        transformer_out = self.transformer(features)

        # 取所有时间步特征的平均值进行分类 (Global Average Pooling)
        # -> (Batch, d_model)
        mean_feature = transformer_out.mean(dim=1)

        # 分类
        out = self.classifier(mean_feature)
        return out


# 初始化模型
model = CWTTransformerModel(num_classes=NUM_CLASSES, d_model=D_MODEL).to(device)

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# 5. 训练循环 (与之前类似，但注意数据生成可能变慢)
def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1} 开始训练...")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                print(f"   Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Acc: {100 * correct / total:.2f}%")


# 运行
train()