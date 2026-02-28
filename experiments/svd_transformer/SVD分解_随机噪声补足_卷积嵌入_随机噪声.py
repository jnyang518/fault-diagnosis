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
import math
import sys

# ================= 1. 全局配置参数 =================
USE_DUMMY_DATA = False
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

# SVD 参数
SVD_RANK = 18  # 低秩压缩维度

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 (保持不变) =================
def load_data_wrapper():
    # 模拟数据生成器
    def create_dummy_data(num_samples=2000, seq_len=128, num_classes=4):
        print("[提示] 生成模拟数据中...")
        t = np.linspace(0, 10, seq_len)
        X, y = [], []
        for i in range(num_samples):
            label = np.random.randint(0, num_classes)
            freq = 5 + label * 5
            noise = np.random.randn(seq_len) * 0.5
            signal = np.sin(2 * np.pi * freq * t) + noise
            X.append(signal.reshape(seq_len, 1))
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y), [f"F_{i}" for i in range(num_classes)]

    if USE_DUMMY_DATA:
        return create_dummy_data(seq_len=SEQ_LEN)
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"成功读取文件: {FILE_PATH}")
        class_names = df.columns.tolist()
        X_data, y_data = [], []
        for i, col in enumerate(df.columns):
            series = df[col].values
            num_samples = len(series) // SEQ_LEN
            series = series[:num_samples * SEQ_LEN]
            segments = series.reshape(-1, SEQ_LEN, 1)
            X_data.append(segments)
            y_data.append(np.full(segments.shape[0], i))

        X = np.concatenate(X_data, axis=0)
        y = np.concatenate(y_data, axis=0)
        return X, y, class_names
    except FileNotFoundError:
        print("[错误] 文件未找到，切换为模拟数据模式")
        return create_dummy_data(seq_len=SEQ_LEN)


X, y, CLASS_NAMES = load_data_wrapper()
NUM_CLASSES = len(CLASS_NAMES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
N_train, L, F_dim = X_train.shape
N_test = X_test.shape[0]
X_train = scaler.fit_transform(X_train.reshape(-1, F_dim)).reshape(N_train, L, F_dim)
X_test = scaler.transform(X_test.reshape(-1, F_dim)).reshape(N_test, L, F_dim)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE,
                         shuffle=False)


# ================= 3. 轻量化核心模块 (核心修改部分) =================

class ConvEmbedding(nn.Module):
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
        return x.permute(0, 2, 1)


class ResidualSVDLinear(nn.Module):
    """
    修改后的 SVD 线性层：
    1. 执行截断 SVD (通过 low-rank分解模拟: Ak = U * V)
    2. 计算当前权重的统计特性 (Mean, Std)
    3. 生成补偿噪声 N ~ Normal(Mean, Std)
    4. 学习参数 alpha
    5. 输出 = Ak * x + alpha * (N * x)
    """

    def __init__(self, in_features, out_features, rank):
        super(ResidualSVDLinear, self).__init__()
        self.rank = min(rank, in_features, out_features)

        # 截断部分 Ak = U * V
        self.down_project = nn.Linear(in_features, self.rank, bias=False)
        self.up_project = nn.Linear(self.rank, out_features, bias=True)

        # 可学习的调节系数 alpha，初始化为一个较小的值，允许模型根据梯度调整
        self.alpha = nn.Parameter(torch.tensor(0.02))

    def forward(self, x):
        # 1. 计算截断 SVD 的主要路径: Ak * x
        # 这种方式避免了显式构建大矩阵，保持了计算效率
        clean_output = self.up_project(self.down_project(x))

        # 仅在训练时加入随机补偿 (类似 Dropout 的机制，但在测试时我们希望使用确定的主要成分)
        if self.training:
            # 2. & 3. 分析统计量并生成噪声
            # 为了获取正确的统计量，我们需要重建等效的权重矩阵 W_approx
            # W_approx = Up * Down
            # 注意：使用 detach() 是为了避免梯度流向统计量的计算，防止模型为了最小化噪声而把权重压缩为0
            with torch.no_grad():
                W_approx = self.up_project.weight @ self.down_project.weight
                mu = W_approx.mean()
                sigma = W_approx.std()

            # 生成符合当前权重统计分布的随机噪声矩阵 N
            # shape: [out_features, in_features]
            noise_weight = torch.randn_like(W_approx, device=x.device) * sigma + mu

            # 4. & 5. 合成: A_new * x = (Ak + alpha * N) * x = Ak*x + alpha * (N*x)
            # F.linear(input, weight) 等价于 input @ weight.T
            noise_output = F.linear(x, noise_weight)

            return clean_output + self.alpha * noise_output

        return clean_output


class SVDMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rank, dropout=0.1):
        super(SVDMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 使用新的 ResidualSVDLinear 替换原有的 SVDLinear
        self.q_proj = ResidualSVDLinear(d_model, d_model, rank)
        self.k_proj = ResidualSVDLinear(d_model, d_model, rank)
        self.v_proj = ResidualSVDLinear(d_model, d_model, rank)
        self.out_proj = ResidualSVDLinear(d_model, d_model, rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, _ = x.size()
        Q = self.q_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(context)


class FullSVDEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, rank, dropout=0.1):
        super(FullSVDEncoderLayer, self).__init__()
        self.self_attn = SVDMultiheadAttention(d_model, nhead, rank, dropout)

        # FFN层也使用 ResidualSVDLinear
        self.linear1 = ResidualSVDLinear(d_model, d_model * 2, rank)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = ResidualSVDLinear(d_model * 2, d_model, rank)

        # 移除了原本独立的 StochasticNoise 层，因为噪声现在已经整合进 Linear 层内部了
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = F.relu(src2)
        # 噪声注入发生在 linear 内部，这里直接 dropout
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
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

    def forward(self, x): return x + self.pe[:x.size(0), :]


class CNN_SVD_Transformer_Shared(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, rank, dropout):
        super(CNN_SVD_Transformer_Shared, self).__init__()
        self.embedding = ConvEmbedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 权重共享
        self.shared_layer = FullSVDEncoderLayer(d_model, nhead, rank, dropout)
        self.num_layers = num_layers

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

        for _ in range(self.num_layers):
            src = self.shared_layer(src)

        output = src.mean(dim=1)
        output = self.fc(output)
        return output


# ================= 4. 模型实例化与训练 =================

model = CNN_SVD_Transformer_Shared(
    input_dim=1,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    rank=SVD_RANK,
    dropout=DROPOUT
).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"模型参数量: {count_parameters(model):,}")

# 检查 alpha 参数是否在优化列表中
# alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n]
# print(f"检测到 {len(alpha_params)} 个可学习的 alpha 参数")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


def train_and_validate(model, train_loader, test_loader, epochs):
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc':<10} | {'Test Loss':<12} | {'Test Acc':<10}")
    print("-" * 65)

    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader.dataset)
        test_acc = 100. * test_correct / test_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        print(
            f"{epoch + 1:<6.0f} | {avg_train_loss:<12.4f} | {train_acc:<9.2f}% | {avg_test_loss:<12.4f} | {test_acc:<9.2f}%")
        scheduler.step(avg_test_loss)

    return history


history = train_and_validate(model, train_loader, test_loader, EPOCHS)


# ================= 5. 绘图与评估 =================
def plot_results(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss', color='red', linestyle='--')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='blue')
    plt.plot(epochs_range, history['test_acc'], label='Test Acc', color='green', linestyle='--')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_results(history)

print("\n--- 最终分类报告 (Test Set) ---")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))