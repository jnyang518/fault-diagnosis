import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pywt
import cv2

# ================= 配置参数 =================
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

# 数据限制
TRAIN_LIMIT = 102400  # 训练集总数据量
TEST_LIMIT = 25600  # 测试集总数据量

# 样本结构参数
SEGMENT_LEN = 1024  # 单个小片段长度 (一张图对应的点数)
TRAIN_SEQ_LEN = 1  # 【修改】训练集：序列长度改为 1 (单张图)
TEST_SEQ_LEN = 1  # 测试集：序列长度 1

# 目标训练样本数
TARGET_TRAIN_SAMPLES = 100  # 102400 / 1024 刚好等于 100

IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0001
D_MODEL = 128
NUM_CLASSES = 4
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 小波变换
def signal_to_cwt_image(signal_segment):
    scales = np.arange(1, 65)
    coefs, freqs = pywt.cwt(signal_segment, scales, 'morl')
    cwt_img = np.abs(coefs)

    denom = cwt_img.max() - cwt_img.min()
    if denom > 0:
        cwt_img = (cwt_img - cwt_img.min()) / denom

    cwt_img = cv2.resize(cwt_img, IMG_SIZE)
    return cwt_img


# 2. Dataset
class CWTSequenceDataset(Dataset):
    def __init__(self, X_list, y, segment_len, num_segments):
        self.X_list = X_list
        self.y = y
        self.segment_len = segment_len
        self.num_segments = num_segments

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raw_signal = self.X_list[idx]
        image_sequence = []

        # 此时 num_segments = 1，循环只执行一次
        for i in range(self.num_segments):
            start = i * self.segment_len
            end = start + self.segment_len
            segment = raw_signal[start:end]
            img = signal_to_cwt_image(segment)
            image_sequence.append(img)

        image_sequence = np.array(image_sequence)
        # 形状: (1, 1, 64, 64) -> (Seq, Channel, H, W)
        image_sequence = image_sequence[:, np.newaxis, :, :]
        return torch.FloatTensor(image_sequence), torch.LongTensor([self.y[idx]]).squeeze()


# 3. 数据加载与切分
def load_data_simple_split(file_path):
    df = pd.read_csv(file_path)

    X_train_samples = []
    y_train_samples = []
    X_test_samples = []
    y_test_samples = []

    print("正在标准化数据...")
    scaler = StandardScaler()
    all_data = df.values.flatten().reshape(-1, 1)
    scaler.fit(all_data)

    # 窗口大小 = 1 * 1024 = 1024
    window_size = SEGMENT_LEN * TRAIN_SEQ_LEN

    # 计算步长
    # 如果 Target=100, Limit=102400, Window=1024
    # (102400 - 1024) / 99 ≈ 1024.
    # 数据刚好够无缝连接，没有重叠
    if TARGET_TRAIN_SAMPLES > 1:
        stride = int((TRAIN_LIMIT - window_size) / (TARGET_TRAIN_SAMPLES - 1))
    else:
        stride = window_size

    print(f"窗口大小: {window_size}, 计算步长: {stride}")

    for label_idx, col in enumerate(df.columns):
        series = df[col].values
        series = scaler.transform(series.reshape(-1, 1)).flatten()

        # === 1. 构建训练集 ===
        train_part = series[:TRAIN_LIMIT]

        for i in range(TARGET_TRAIN_SAMPLES):
            start = i * stride
            end = start + window_size

            if end > len(train_part):
                break

            sample = train_part[start:end]
            X_train_samples.append(sample)
            y_train_samples.append(label_idx)

        # === 2. 构建测试集 ===
        test_part = series[TRAIN_LIMIT: TRAIN_LIMIT + TEST_LIMIT]
        test_window = SEGMENT_LEN * TEST_SEQ_LEN  # 1024

        n_test = len(test_part) // test_window
        for i in range(n_test):
            start = i * test_window
            end = start + test_window
            sample = test_part[start:end]
            X_test_samples.append(sample)
            y_test_samples.append(label_idx)

    return X_train_samples, np.array(y_train_samples), X_test_samples, np.array(y_test_samples)


X_train_list, y_train, X_test_list, y_test = load_data_simple_split(FILE_PATH)

print(f"训练集样本数: {len(X_train_list)} (预期 400)")
print(f"单个样本长度: {len(X_train_list[0])} (预期 1024)")

train_ds = CWTSequenceDataset(X_train_list, y_train, SEGMENT_LEN, TRAIN_SEQ_LEN)
test_ds = CWTSequenceDataset(X_test_list, y_test, SEGMENT_LEN, TEST_SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# 4. 模型 (无需修改，自动适应 Seq=1)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 8 * 8, d_model)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CWTTransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, max_len=10):
        super(CWTTransformerModel, self).__init__()
        self.cnn = CNNFeatureExtractor(d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.bn_final = nn.BatchNorm1d(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, 1, 1, 64, 64)
        b, seq, c, h, w = x.shape
        x = x.view(b * seq, c, h, w)
        feats = self.cnn(x)  # (Batch, d_model)
        feats = feats.view(b, seq, -1)  # (Batch, 1, d_model)

        # 加上位置编码 (虽然 seq=1 时位置编码是常数，但必须加上以保持逻辑一致)
        if seq <= self.pos_embedding.shape[1]:
            feats = feats + self.pos_embedding[:, :seq, :]

        out = self.transformer(feats)
        mean_out = self.bn_final(out.mean(dim=1))
        return self.classifier(mean_out)


model = CWTTransformerModel(NUM_CLASSES, D_MODEL, max_len=10).to(device)  # max_len 设大一点没事
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)


# 5. 训练
def evaluate(loader, model, name):
    model.eval()
    correct = 0
    total = 0
    pred_counts = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for p in predicted.cpu().numpy(): pred_counts[p] += 1
    print(f"[{name}] 预测分布: {pred_counts}")
    return 100 * correct / total


print("开始训练 (Seq Len = 1)...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = evaluate(train_loader, model, "Train")
    test_acc = evaluate(test_loader, model, "Test")
    print(
        f"Epoch  [{epoch + 1}/{EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")