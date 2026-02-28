import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
USE_DUMMY_DATA = False  # 使用真实上传的文件
FILE_PATH = 'gearset30_2.csv'  # 【更新】新数据集路径

# 模型超参数
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 2
DROPOUT = 0.1

# SVD 参数 (用于低秩压缩参数量)
SVD_RANK = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 =================

def load_data(file_path, seq_len):
    if USE_DUMMY_DATA:
        print("警告: 正在使用模拟数据模式")
        # (此处省略模拟数据代码，因为有真实文件)
        return None, None, []

    try:
        df = pd.read_csv(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据形状: {df.shape}")
        class_names = df.columns.tolist()
        print(f"检测到的类别: {class_names}")
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {file_path}")
        sys.exit()

    X_data = []
    y_data = []

    # 遍历每一列（每一类故障）
    for i, col in enumerate(df.columns):
        series = df[col].values
        if np.isnan(series).any():
            series = np.nan_to_num(series)

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

N_train, L, F_dim = X_train.shape
N_test, _, _ = X_test.shape

print(f"训练集样本数: {N_train}, 测试集样本数: {N_test}")

# 标准化
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, F_dim)
X_test_flat = X_test.reshape(-1, F_dim)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, L, F_dim)
X_test_scaled = scaler.transform(X_test_flat).reshape(N_test, L, F_dim)

train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ================= 3. 核心模块定义 (SVD + CNN + Transformer) =================

class ConvEmbedding(nn.Module):
    """CNN 局部特征提取"""

    def __init__(self, in_channels, d_model):
        super(ConvEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, d_model // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        return x


class SVDLinear(nn.Module):
    """SVD 低秩线性层: 将 W (d, d) 分解为 U(d, r) * V(r, d)"""

    def __init__(self, in_features, out_features, rank):
        super(SVDLinear, self).__init__()
        self.down_project = nn.Linear(in_features, rank, bias=False)
        self.up_project = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.up_project(self.down_project(x))


class SVDMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rank, dropout=0.1):
        super(SVDMultiheadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 使用 SVDLinear 替代标准 Linear
        self.q_proj = SVDLinear(d_model, d_model, rank)
        self.k_proj = SVDLinear(d_model, d_model, rank)
        self.v_proj = SVDLinear(d_model, d_model, rank)
        self.out_proj = SVDLinear(d_model, d_model, rank)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class SVDTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, rank, dropout=0.1):
        super(SVDTransformerEncoderLayer, self).__init__()
        self.self_attn = SVDMultiheadAttention(d_model, nhead, rank, dropout)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


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


class CNN_SVD_Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, rank, dropout):
        super(CNN_SVD_Transformer, self).__init__()
        self.embedding = ConvEmbedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            SVDTransformerEncoderLayer(d_model, nhead, rank, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        for layer in self.layers:
            src = layer(src)
        output = src.mean(dim=1)
        output = self.fc(output)
        return output


# ================= 4. 模型实例化与参数统计 =================

model = CNN_SVD_Transformer(
    input_dim=1,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    rank=SVD_RANK,
    dropout=DROPOUT
).to(device)


def count_parameters(model):
    """计算模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print("\n" + "=" * 50)
print(f"【模型参数统计】")
print(f"模型名称: CNN-SVD-Transformer")
print(f"总参数量 (Total Parameters): {total_params:,}")
print(f"SVD Rank 设置: {SVD_RANK} (原始 d_model={D_MODEL})")
print(f"说明: 使用 SVD 低秩近似大幅减少了 Attention 层的参数")
print("=" * 50 + "\n")

# ================= 5. 训练循环 =================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


def train_and_validate(model, train_loader, test_loader, epochs):
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    print(f"--- 开始训练 (Epochs={epochs}) ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_test_loss = test_loss / len(test_loader.dataset)
        acc = 100. * correct / total

        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(acc)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {acc:.2f}%")
        scheduler.step(avg_test_loss)

    return history


history = train_and_validate(model, train_loader, test_loader, EPOCHS)


# ================= 6. 可视化与报告 =================

def plot_results(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Validation Loss')
    plt.title('Loss 曲线')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], color='green', marker='o')
    plt.title('验证集准确率 (Accuracy)')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(model, loader, classes):
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

    print("\n--- 最终分类报告 ---")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


plot_results(history)
plot_confusion_matrix(model, test_loader, CLASS_NAMES)