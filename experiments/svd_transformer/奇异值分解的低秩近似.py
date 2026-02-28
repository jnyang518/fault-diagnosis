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
USE_DUMMY_DATA = False # 【注意】演示模式设为True。若有真实文件，请改为False
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

# 模型超参数
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 2
DROPOUT = 0.1

# SVD 特有参数：低秩压缩
# 原始参数量约为 d*d，压缩后为 d*r + r*d = 2dr
# 如果 r < d/2，则参数减少
SVD_RANK = 64  # 将 256 维压缩到 64 维再还原

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 =================

def create_dummy_data(num_samples=2000, seq_len=128, num_classes=4):
    print(f"[提示] 正在生成模拟数据 (样本数={num_samples}, 类别={num_classes})...")
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
        print(f"[错误] 找不到文件: {file_path}. 请检查路径或将 USE_DUMMY_DATA 设为 True")
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

N_train, L, F_dim = X_train.shape
N_test, _, _ = X_test.shape

scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, F_dim)
X_test_flat = X_test.reshape(-1, F_dim)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, L, F_dim)
X_test_scaled = scaler.transform(X_test_flat).reshape(N_test, L, F_dim)

train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ================= 3. SVD 改进模型定义 =================

# [核心改进 1] SVD 低秩线性层
class SVDLinear(nn.Module):
    """
    使用低秩分解近似全连接层 W (d_in, d_out)
    W approx U * V
    U: (d_in, rank), V: (rank, d_out)
    参数量: d_in*d_out -> (d_in + d_out) * rank
    """

    def __init__(self, in_features, out_features, rank):
        super(SVDLinear, self).__init__()
        self.down_project = nn.Linear(in_features, rank, bias=False)
        self.up_project = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        x = self.down_project(x)
        x = self.up_project(x)
        return x


# [核心改进 2] 基于 SVD 线性层的多头注意力机制
class SVDMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rank, dropout=0.1):
        super(SVDMultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 使用 SVDLinear 替代标准的 nn.Linear
        # Q, K, V 投影
        self.q_proj = SVDLinear(d_model, d_model, rank)
        self.k_proj = SVDLinear(d_model, d_model, rank)
        self.v_proj = SVDLinear(d_model, d_model, rank)

        # 输出投影
        self.out_proj = SVDLinear(d_model, d_model, rank)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, d_model) -- 注意这里我假设输入已经是 batch first 处理后的
        batch_size, seq_len, _ = x.size()

        # 1. 线性投影并分头
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, Num_Heads, d_k) -> (Batch, Num_Heads, Seq_Len, d_k)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 缩放点积注意力 (Scaled Dot-Product Attention)
        # scores: (Batch, Num_Heads, Seq_Len, Seq_Len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context: (Batch, Num_Heads, Seq_Len, d_k)
        context = torch.matmul(attn_weights, V)

        # 3. 拼接头并输出
        # (Batch, Seq_Len, Num_Heads * d_k) -> (Batch, Seq_Len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(context)
        return output


# 包裹 SVD Attention 的编码器层
class SVDTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, rank, dropout=0.1):
        super(SVDTransformerEncoderLayer, self).__init__()
        self.self_attn = SVDMultiheadAttention(d_model, nhead, rank, dropout)

        # Feed Forward Network (也可以选择在这里使用 SVDLinear，暂保持标准以凸显 Attention 的改动)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 2, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Pre-Norm 结构
        # 1. Attention Block
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout1(src2)

        # 2. FFN Block
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


class SVDTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, rank, dropout):
        super(SVDTransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 堆叠自定义的 SVD Encoder Layers
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
        # src: (Batch, Seq_Len, 1) -> (Batch, Seq_Len, d_model)
        src = self.input_linear(src)

        # 转置为 (Seq_Len, Batch, d_model) 适配 PosEncoder
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)

        # 转回 (Batch, Seq_Len, d_model) 适配我们自定义的 Attention
        src = src.permute(1, 0, 2)

        for layer in self.layers:
            src = layer(src)

        # Global Average Pooling
        output = src.mean(dim=1)
        output = self.fc(output)
        return output


# ================= 4. 参数量计算与模型初始化 =================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 1. 初始化 SVD 模型
svd_model = SVDTransformerModel(
    input_dim=1,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    rank=SVD_RANK,  # 这里的 Rank 决定了压缩率
    dropout=DROPOUT
).to(device)


# 2. 为了对比，计算标准模型的参数量（理论计算，不实际实例化占用显存）
# 标准 Attention 4个矩阵 (d, d): 4 * 256 * 256 = 262,144 参数
# SVD Attention 4个投影 (d, r) + (r, d): 4 * (256*64 + 64*256) = 131,072 参数
# 大约减少 50% 的 Attention 参数

# 我们快速定义一个标准模型来获取精确数值
class StandardTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD)
        self.enc = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.in_lin = nn.Linear(1, D_MODEL)
        self.fc = nn.Linear(D_MODEL, NUM_CLASSES)
        # 简单模拟结构用于计数


std_model_params = 0
# 计算标准一层 Encoder 的参数:
# Self Attn: 4 * d_model^2 + 4 * d_model (bias)
# FFN: 2 * d_model * (2*d_model) ...
# 这里直接手动估算或建立临时模型对比
temp_std_model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_MODEL * 2),
    num_layers=NUM_LAYERS
)
std_params_count = count_parameters(temp_std_model) + count_parameters(nn.Linear(1, D_MODEL)) + count_parameters(
    nn.Linear(D_MODEL, NUM_CLASSES))

svd_params_count = count_parameters(svd_model)

print("\n" + "=" * 40)
print(f"参数量对比 (SVD Rank = {SVD_RANK})")
print(f"标准 Transformer (估算): {std_params_count:,} parameters")
print(f"SVD Transformer (当前): {svd_params_count:,} parameters")
reduction = (1 - svd_params_count / std_params_count) * 100
print(f"参数减少量: ▼ {reduction:.2f}%")
print("=" * 40 + "\n")

# ================= 5. 训练循环 =================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(svd_model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


def train_svd_model(model, train_loader, test_loader, epochs):
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    print("--- 开始 SVD 模型训练 ---")
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

        # 验证
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
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {acc:.2f}%")

        scheduler.step(avg_test_loss)

    return history


history = train_svd_model(svd_model, train_loader, test_loader, EPOCHS)


# ================= 6. 结果可视化与混淆矩阵 =================

def plot_results(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Validation Loss')
    plt.title('Loss 曲线 (SVD Optimized)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], color='green', marker='o')
    plt.title('测试集准确率 (Accuracy)')
    plt.ylim(0, 100)
    plt.grid(True)

    plt.show()


def plot_confusion(model, loader):
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

    print("\n--- SVD 模型分类报告 ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('混淆矩阵 (SVD Model)')
    plt.show()


plot_results(history)
plot_confusion(svd_model, test_loader)