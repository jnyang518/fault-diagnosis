import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

# ================= 1. 全局配置参数 =================
USE_DUMMY_DATA = False
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

# 模型超参数
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
D_MODEL = 256
BLOCK_SIZE = 64  # 更改为 64x64 的大格
NUM_LAYERS = 2
DROPOUT = 0.1
NOISE_STD = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")


# ================= 2. 数据加载与预处理 =================
def load_data_wrapper():
    def create_dummy_data(num_samples=2000, seq_len=256, num_classes=4):
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

    if USE_DUMMY_DATA: return create_dummy_data()
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
        return create_dummy_data()


X, y, CLASS_NAMES = load_data_wrapper()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(-1, SEQ_LEN, 1)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(-1, SEQ_LEN, 1)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE,
                         shuffle=False)


# ================= 3. 核心改进：64x64 分块 Rank-1 线性层 =================

class LargeBlockRank1Linear(nn.Module):
    """
    权重矩阵划分为 64x64 的块。
    每一块由两个 64 维向量的外积 (Rank-1) 形成。
    """

    def __init__(self, in_features, out_features, block_size=64):
        super(LargeBlockRank1Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # 针对 256 维，这里 num_blocks 将会是 4x4 (对于 d_model) 或 4x8 (对于 FFN)
        self.num_blocks_row = out_features // block_size
        self.num_blocks_col = in_features // block_size

        # 参数: 每块拥有两个长度为 64 的向量
        self.U = nn.Parameter(torch.randn(self.num_blocks_row, self.num_blocks_col, block_size))
        self.V = nn.Parameter(torch.randn(self.num_blocks_row, self.num_blocks_col, block_size))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x):
        # 计算外积: (row_b, col_b, 64) x (row_b, col_b, 64) -> (row_b, col_b, 64, 64)
        W_blocks = torch.einsum('rci,rcj->rcij', self.U, self.V)

        # 拼接回完整大矩阵 (out_features, in_features)
        W = W_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return F.linear(x, W, self.bias)


# ================= 4. 模型架构 =================

class BlockAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Q, K, V 投影均使用 64x64 分块方案
        self.q_proj = LargeBlockRank1Linear(d_model, d_model, block_size)
        self.k_proj = LargeBlockRank1Linear(d_model, d_model, block_size)
        self.v_proj = LargeBlockRank1Linear(d_model, d_model, block_size)
        self.out_proj = LargeBlockRank1Linear(d_model, d_model, block_size)

    def forward(self, x):
        B, S, D = x.size()
        Q = self.q_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(context)


class BlockTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, block_size, dropout, noise_std):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BlockAttention(d_model, nhead, block_size)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN 部分同样应用 64x64 分块
        self.linear1 = LargeBlockRank1Linear(d_model, d_model * 2, block_size)
        self.linear2 = LargeBlockRank1Linear(d_model * 2, d_model, block_size)
        self.dropout = nn.Dropout(dropout)
        self.noise_std = noise_std

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        res = x
        x = F.relu(self.linear1(self.norm2(x)))
        if self.training: x = x + torch.randn_like(x) * self.noise_std
        x = res + self.dropout(self.linear2(x))
        return x


class UltimateBlockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始 Embedding 使用 Conv1d 提取局部特征
        self.embedding = nn.Sequential(
            nn.Conv1d(1, D_MODEL, 5, padding=2), nn.BatchNorm1d(D_MODEL), nn.ReLU()
        )
        self.shared_layer = BlockTransformerLayer(D_MODEL, 8, BLOCK_SIZE, DROPOUT, NOISE_STD)
        self.fc = nn.Linear(D_MODEL, len(CLASS_NAMES))

    def forward(self, x):
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        # 权重共享循环调用
        for _ in range(NUM_LAYERS):
            x = self.shared_layer(x)
        return self.fc(x.mean(dim=1))


# ================= 5. 训练 =================
model = UltimateBlockTransformer().to(device)
print(f"极度轻量化模型参数量 (64x64 Block): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"{'Epoch':<8} | {'Loss':<10} | {'Test Acc':<10}")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward();
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx.to(device)).argmax(1)
            correct += (pred == by.to(device)).sum().item()

    print(f"Epoch {epoch + 1:02d} | {total_loss / len(train_loader):<10.4f} | {100 * correct / len(y_test):.2f}%")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.to(device))
            correct += (outputs.argmax(1) == batch_y.to(device)).sum().item()

    acc = 100 * correct / len(y_test)
    history['train_loss'].append(total_loss / len(train_loader))
    history['test_acc'].append(acc)
    print(f"Epoch {epoch + 1:02d} | Loss: {total_loss / len(train_loader):.4f} | Test Acc: {acc:.2f}%")

# ================= 5. 结果可视化 =================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1);
plt.plot(history['train_loss']);
plt.title('Loss')
plt.subplot(1, 2, 2);
plt.plot(history['test_acc']);
plt.title('Accuracy (%)')
plt.show()