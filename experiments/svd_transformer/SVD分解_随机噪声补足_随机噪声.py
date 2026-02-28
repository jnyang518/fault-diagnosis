import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import math

# ================= 1. 全局配置参数 =================
USE_DUMMY_DATA = False
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

# 模型超参数
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 40  # 增加 Epoch 以配合退火策略
LR = 0.001
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 2
DROPOUT = 0.1
SVD_RANK = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 =================
def load_data_wrapper():
    def create_dummy_data(num_samples=2000, seq_len=128, num_classes=4):
        t = np.linspace(0, 10, seq_len)
        X, y = [], []
        for i in range(num_samples):
            label = np.random.randint(0, num_classes)
            freq = 5 + label * 5
            signal = np.sin(2 * np.pi * freq * t) + np.random.randn(seq_len) * 0.5
            X.append(signal.reshape(seq_len, 1))
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y), [f"Class_{i}" for i in range(num_classes)]

    if USE_DUMMY_DATA: return create_dummy_data(seq_len=SEQ_LEN)
    try:
        df = pd.read_csv(FILE_PATH)
        class_names = df.columns.tolist()
        X_data, y_data = [], []
        for i, col in enumerate(df.columns):
            series = df[col].values
            num_samples = len(series) // SEQ_LEN
            segments = series[:num_samples * SEQ_LEN].reshape(-1, SEQ_LEN, 1)
            X_data.append(segments);
            y_data.append(np.full(segments.shape[0], i))
        return np.concatenate(X_data, axis=0), np.concatenate(y_data, axis=0), class_names
    except:
        return create_dummy_data(seq_len=SEQ_LEN)


X, y, CLASS_NAMES = load_data_wrapper()
NUM_CLASSES = len(CLASS_NAMES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
N_tr, L, F_d = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, F_d)).reshape(N_tr, L, F_d)
X_test = scaler.transform(X_test.reshape(-1, F_d)).reshape(X_test.shape[0], L, F_d)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE,
                         shuffle=False)


# ================= 3. 进阶轻量化模块 =================

class AdvancedResidualSVDLinear(nn.Module):
    """
    改进点：
    1. 逐行统计 (Per-row statistics): 针对每个输出通道生成独立的噪声分布。
    2. 稳定性控制: 引入 eps 防止标准差为 0。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = min(rank, in_features, out_features)
        self.down_project = nn.Linear(in_features, self.rank, bias=False)
        self.up_project = nn.Linear(self.rank, out_features, bias=True)
        # alpha 初始化减小，给予模型更稳健的起点
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        clean_output = self.up_project(self.down_project(x))

        if self.training:
            with torch.no_grad():
                # 显式获取近似权重矩阵 W (Out x In)
                W = self.up_project.weight @ self.down_project.weight
                # 改进：计算每一行（每个神经元）的均值和标准差
                mu = W.mean(dim=1, keepdim=True)
                sigma = W.std(dim=1, keepdim=True) + 1e-6

            # 生成结构化噪声
            noise_w = torch.randn_like(W) * sigma + mu
            noise_output = F.linear(x, noise_w)
            return clean_output + self.alpha * noise_output

        return clean_output


class SVDMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rank, dropout=0.1):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads
        self.q_proj = AdvancedResidualSVDLinear(d_model, d_model, rank)
        self.k_proj = AdvancedResidualSVDLinear(d_model, d_model, rank)
        self.v_proj = AdvancedResidualSVDLinear(d_model, d_model, rank)
        self.out_proj = AdvancedResidualSVDLinear(d_model, d_model, rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.size()
        Q = self.q_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(context)


class FullSVDEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, rank, dropout=0.1):
        super().__init__()
        self.self_attn = SVDMultiheadAttention(d_model, nhead, rank, dropout)
        self.linear1 = AdvancedResidualSVDLinear(d_model, d_model * 2, rank)
        self.linear2 = AdvancedResidualSVDLinear(d_model * 2, d_model, rank)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src):
        src = src + self.dropout1(self.self_attn(self.norm1(src)))
        src2 = F.relu(self.linear1(self.norm2(src)))
        src = src + self.dropout2(self.linear2(src2))
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x): return x + self.pe[:x.size(0), :]


class SVD_Transformer_Final(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, rank, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.shared_layer = FullSVDEncoderLayer(d_model, nhead, rank, dropout)
        self.num_layers = num_layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),  # 增强分类头
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, src):
        src = self.embedding(src).transpose(0, 1)
        src = self.pos_encoder(src)
        for _ in range(self.num_layers):
            src = self.shared_layer(src)
        output = src.permute(1, 0, 2).mean(dim=1)
        return self.fc(output)


# ================= 4. 模型实例化与带退火的训练 =================

model = SVD_Transformer_Final(1, NUM_CLASSES, D_MODEL, N_HEAD, NUM_LAYERS, SVD_RANK, DROPOUT).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"【模型信息】总可学习参数量: {count_parameters(model):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train_model():
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        # 噪声退火逻辑：随着训练进行，手动微调 alpha 的量级（可选）
        # for m in model.modules():
        #    if isinstance(m, AdvancedResidualSVDLinear):
        #        m.alpha.data *= 0.99

        total_loss, correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_correct += (model(inputs).argmax(1) == labels).sum().item()

        acc = 100. * val_correct / len(test_loader.dataset)
        print(f"Epoch {epoch + 1:02d} | Loss: {total_loss / len(train_loader.dataset):.4f} | "
              f"Train Acc: {100. * correct / len(train_loader.dataset):.2f}% | Test Acc: {acc:.2f}%")
        scheduler.step()


train_model()

# ================= 5. 最终评估 =================
print("\n--- 最终分类报告 ---")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        y_true.extend(labels.numpy())
        y_pred.extend(model(inputs.to(device)).argmax(1).cpu().numpy())

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))