import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ==========================================
# 0. 工具函数：计算参数量
# ==========================================
def count_parameters(model):
    """统计模型的可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数统计:")
    print(f"------------------------------------------")
    print(f"总参数量 (Total Params): {total_params:,}")
    print(f"模型深度 (Layers): ~18 层")
    print(f"------------------------------------------\n")
    return total_params


# ==========================================
# 1. 残差块定义 (ResBlock)
# ==========================================
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ==========================================
# 2. 主模型：1D-ResNet-18
# ==========================================
class ResNet1D_18(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet1D_18, self).__init__()

        # 初始层
        self.initial = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 4 个残差阶段
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ==========================================
# 3. 数据加载
# ==========================================
def load_1d_data(csv_path, signal_length=3600, stride=500):
    print(f"正在读取数据: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
    except:
        return [], [], []

    X, y = [], []
    class_names = df.columns.tolist()
    class_map = {name: i for i, name in enumerate(class_names)}

    for col in class_names:
        full_signal = df[col].values
        num_samples = (len(full_signal) - signal_length) // stride + 1
        for i in range(num_samples):
            segment = full_signal[i * stride: i * stride + signal_length]
            segment = (segment - segment.mean()) / (segment.std() + 1e-6)
            X.append(segment)
            y.append(class_map[col])

    X = np.array(X, dtype=np.float32)
    X = X[:, np.newaxis, :]  # [N, 1, 3600]
    y = np.array(y, dtype=np.int64)
    print(f"数据加载完成: {len(X)} 样本")
    return X, y, class_names


class VibrationDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ==========================================
# 4. 训练流程
# ==========================================
def train_resnet1d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备数据
    X, y, classes = load_1d_data('merged_output.csv', stride=1000)
    if len(X) == 0: return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_dl = DataLoader(VibrationDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_dl = DataLoader(VibrationDataset(X_val, y_val), batch_size=64, shuffle=False)

    # 2. 初始化模型
    model = ResNet1D_18(num_classes=len(classes)).to(device)

    # 🟢 打印参数量
    count_parameters(model)

    # 3. 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练
    epochs = 30
    best_acc = 0.0
    history = {'loss': [], 'acc': []}

    print("\n🚀 开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, label in train_dl:
            x, label = x.to(device), label.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == label).sum().item()
            total += label.size(0)

        scheduler.step()

        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, label in val_dl:
                x, label = x.to(device), label.to(device)
                val_correct += (model(x).argmax(1) == label).sum().item()
                val_total += label.size(0)

        acc = 100 * val_correct / val_total
        history['loss'].append(total_loss / len(train_dl))
        history['acc'].append(acc)

        print(
            f"Epoch {epoch + 1:02d} | Loss: {total_loss / len(train_dl):.4f} | Train Acc: {100 * correct / total:.2f}% | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_resnet1d.pth')

    print(f"最佳准确率: {best_acc:.2f}%")
    plt.plot(history['acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.show()


if __name__ == '__main__':
    train_resnet1d()