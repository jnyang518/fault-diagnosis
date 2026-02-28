import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 新增
import pywt
import cv2

# ================= 配置参数 =================
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

# 样本结构参数
SEGMENT_LEN = 256  # 单个样本长度
TRAIN_SEQ_LEN = 1  # 序列长度
TEST_SEQ_LEN = 1  # 测试集序列长度

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
        # 此时 X_list[idx] 已经是切割好的 (1024,) 数组
        raw_signal = self.X_list[idx]
        image_sequence = []

        # num_segments = 1，循环只执行一次
        for i in range(self.num_segments):
            # 直接使用当前样本进行 CWT
            img = signal_to_cwt_image(raw_signal)
            image_sequence.append(img)

        image_sequence = np.array(image_sequence)
        # 形状: (1, 1, 64, 64) -> (Seq, Channel, H, W)
        image_sequence = image_sequence[:, np.newaxis, :, :]
        return torch.FloatTensor(image_sequence), torch.LongTensor([self.y[idx]]).squeeze()


# 3. 数据加载与切分 (修改版)
def load_data_split(file_path):
    df = pd.read_csv(file_path)

    print("正在标准化数据...")
    scaler = StandardScaler()
    all_data = df.values.flatten().reshape(-1, 1)
    scaler.fit(all_data)

    X_all = []
    y_all = []

    print(f"原始数据形状: {df.shape}")

    # 遍历每一列（每一个类别）
    for label_idx, col in enumerate(df.columns):
        series = df[col].values
        series = scaler.transform(series.reshape(-1, 1)).flatten()

        # === 按照 1024 长度切分全量数据 ===
        n_samples = len(series) // SEGMENT_LEN
        for i in range(n_samples):
            start = i * SEGMENT_LEN
            end = start + SEGMENT_LEN
            sample = series[start:end]

            X_all.append(sample)
            y_all.append(label_idx)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    print(f"总样本数: {len(X_all)} (每类约 {len(X_all) // NUM_CLASSES} 个)")

    # === 8:2 随机划分训练集和测试集 ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    return X_train, y_train, X_test, y_test


X_train_list, y_train, X_test_list, y_test = load_data_split(FILE_PATH)

print(f"训练集样本数: {len(X_train_list)}")
print(f"测试集样本数: {len(X_test_list)}")

train_ds = CWTSequenceDataset(X_train_list, y_train, SEGMENT_LEN, TRAIN_SEQ_LEN)
test_ds = CWTSequenceDataset(X_test_list, y_test, SEGMENT_LEN, TEST_SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# 4. 模型 (保持不变)
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
        b, seq, c, h, w = x.shape
        x = x.view(b * seq, c, h, w)
        feats = self.cnn(x)
        feats = feats.view(b, seq, -1)

        if seq <= self.pos_embedding.shape[1]:
            feats = feats + self.pos_embedding[:, :seq, :]

        out = self.transformer(feats)
        mean_out = self.bn_final(out.mean(dim=1))
        return self.classifier(mean_out)


model = CWTTransformerModel(NUM_CLASSES, D_MODEL, max_len=10).to(device)
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