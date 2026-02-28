import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys

# ================= 1. 全局配置参数 =================
USE_DUMMY_DATA = False  # 如果没有数据文件，设为 True
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

# 模型超参数
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 30  # 稍微增加 Epoch 以观察动态学习率的效果
LR = 0.001  # 初始学习率
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 2
DROPOUT = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 (保持不变) =================

def create_dummy_data(num_samples=2000, seq_len=128, num_classes=4):
    print(f"[提示]正在生成模拟数据 (样本数={num_samples}, 类别={num_classes})...")
    X = np.random.randn(num_samples, seq_len, 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    class_names = [f"故障_{i}" for i in range(num_classes)]
    return X, y, class_names


def load_data(file_path, seq_len):
    if USE_DUMMY_DATA:
        return create_dummy_data(seq_len=seq_len, num_classes=4)
    try:
        df = pd.read_csv(file_path)
        print(f"成功读取文件: {file_path}")
        class_names = df.columns.tolist()
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {file_path}")
        sys.exit()

    X_data = []
    y_data = []
    for i, col in enumerate(df.columns):
        series = df[col].values
        num_samples = len(series) // seq_len
        series = series[:num_samples * seq_len]
        segments = series.reshape(-1, seq_len, 1)
        X_data.append(segments)
        y_data.append(np.full(segments.shape[0], i))

    X = np.concatenate(X_data, axis=0)
    y = np.concatenate(y_data, axis=0)
    return X, y, class_names


print("--- 正在处理数据 ---")
X, y, CLASS_NAMES = load_data(FILE_PATH, SEQ_LEN)
NUM_CLASSES = len(CLASS_NAMES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

N_train, L, F = X_train.shape
N_test, _, _ = X_test.shape

scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, F)
X_test_flat = X_test.reshape(-1, F)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, L, F)
X_test_scaled = scaler.transform(X_test_flat).reshape(N_test, L, F)

train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ================= 3. 模型定义 (改进版：Residual + Pre-Norm) =================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # [改进点 1] 残差链接配置
        # 设置 norm_first=True (Pre-Norm)。
        # 标准 Transformer 是 Post-Norm: Norm(x + Sublayer(x))
        # Pre-Norm 是: x + Sublayer(Norm(x))
        # Pre-Norm 这里的残差连接更直接，梯度流更稳定，适合深层网络。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            norm_first=True  # 开启 Pre-Norm 模式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, src):
        # src: (Batch, Seq_Len, 1)
        src = self.input_linear(src)  # 线性投影

        # 调整维度满足 Transformer 输入: (Seq_Len, Batch, d_model)
        src = src.permute(1, 0, 2)

        src = self.pos_encoder(src)

        # Transformer 内部已经实现了多头注意力和前馈网络的残差连接 (Add & Norm)
        output = self.transformer_encoder(src)

        # Global Average Pooling
        output = output.permute(1, 0, 2)  # -> (Batch, Seq_Len, d_model)
        output = output.mean(dim=1)  # -> (Batch, d_model)

        output = self.fc(output)
        return output


model = TransformerModel(
    input_dim=1,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# [改进点 2] 动态学习率调度器
# ReduceLROnPlateau: 当指标(如 val_loss)停止改善时，降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 监控指标越小越好 (Loss)
    factor=0.5,  # 学习率调整倍数 (new_lr = lr * 0.5)
    patience=3,  # 容忍多少个 epoch 指标不下降
    verbose=True,  # 打印日志
    min_lr=1e-6  # 最小学习率
)


# ================= 4. 训练与评估循环 (集成 Scheduler) =================

def train_and_validate(model, train_loader, test_loader, epochs):
    print("\n--- 开始训练 (含动态学习率与残差优化) ---")

    # 用于记录 loss 曲线
    history = {'train_loss': [], 'test_loss': [], 'lr': []}

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸，配合残差结构使用效果更佳)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # --- 验证阶段 ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        # 计算平均指标
        avg_train_loss = train_loss / train_total
        avg_train_acc = 100. * train_correct / train_total
        avg_test_loss = test_loss / test_total
        avg_test_acc = 100. * test_correct / test_total

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)

        print(f"Epoch [{epoch + 1:02d}/{epochs}] "
              f"| LR: {current_lr:.6f} "
              f"| Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% "
              f"| Test Loss: {avg_test_loss:.4f} Acc: {avg_test_acc:.2f}%")

        # [关键] 更新学习率
        # 根据测试集 Loss 调整学习率
        scheduler.step(avg_test_loss)

    return history


# 执行训练
history = train_and_validate(model, train_loader, test_loader, EPOCHS)


# ================= 5. 结果可视化 (新增学习率曲线) =================

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Validation Loss')
    ax1.set_title('训练与验证 Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Learning Rate 变化曲线
    ax2.plot(epochs, history['lr'], 'g-o', label='Learning Rate')
    ax2.set_title('动态学习率变化 (Dynamic LR)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')  # 对数坐标显示更清晰
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


plot_training_history(history)


# ================= 6. 混淆矩阵与评估 =================
# (保持原有评估代码)
def plot_confusion_matrix_and_report(model, loader, device, classes):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()
    print("\n--- 分类报告 ---")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))


plot_confusion_matrix_and_report(model, test_loader, device, CLASS_NAMES)