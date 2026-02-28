import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math

# ================= 1. 配置与数据加载 =================
USE_DUMMY_DATA = False
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

SEQ_LEN = 256
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 4  # 增加深度，因为是共享层，参数量不会增加
SVD_RANK = 16
SPARSE_LAMBDA = 1e-5  # L1 惩罚系数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    # 模拟数据生成逻辑（若文件不存在）
    def dummy():
        X = np.random.randn(1000, SEQ_LEN, 1).astype(np.float32)
        y = np.random.randint(0, 4, 1000)
        return X, y, [f"故障_{i}" for i in range(4)]

    try:
        df = pd.read_csv(FILE_PATH)
        class_names = df.columns.tolist()
        X_list, y_list = [], []
        for i, col in enumerate(df.columns):
            data = df[col].dropna().values
            num = len(data) // SEQ_LEN
            X_list.append(data[:num * SEQ_LEN].reshape(-1, SEQ_LEN, 1))
            y_list.append(np.full(num, i))
        return np.concatenate(X_list), np.concatenate(y_list), class_names
    except:
        return dummy()


X, y, CLASS_NAMES = load_data()

# 8:2 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(-1, SEQ_LEN, 1)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(-1, SEQ_LEN, 1)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
    batch_size=BATCH_SIZE)


# ================= 2. 共享 RPCA 模型定义 =================

class SparseLowRankLinear(nn.Module):
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.rank = min(rank, in_f, out_f)
        self.A = nn.Linear(in_f, self.rank, bias=False)
        self.B = nn.Linear(self.rank, out_f, bias=True)
        self.sparse_weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)

    def forward(self, x):
        return self.B(self.A(x)) + F.linear(x, self.sparse_weight)


class SharedRPCABlock(nn.Module):
    def __init__(self, d_model, nhead, rank, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            SparseLowRankLinear(d_model, d_model * 2, rank),
            nn.ReLU(),
            SparseLowRankLinear(d_model * 2, d_model, rank)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class UltraLightTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Linear(1, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))
        # 核心：整个模型共享这一个层
        self.shared_layer = SharedRPCABlock(D_MODEL, N_HEAD, SVD_RANK, 0.1)
        self.fc = nn.Linear(D_MODEL, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb
        for _ in range(NUM_LAYERS):
            x = self.shared_layer(x)
        return self.fc(x.mean(dim=1))


model = UltraLightTransformer(len(CLASS_NAMES)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ================= 3. 训练循环 =================

history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(EPOCHS):
    model.train()
    train_l, train_c = 0, 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(data)

        # 计算 L1 稀疏惩罚
        l1_loss = sum(torch.norm(p, 1) for n, p in model.named_parameters() if 'sparse_weight' in n)
        loss = criterion(out, label) + SPARSE_LAMBDA * l1_loss

        loss.backward()
        optimizer.step()
        train_l += loss.item()
        train_c += (out.argmax(1) == label).sum().item()

    # 验证
    model.eval()
    test_l, test_c = 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            test_l += criterion(out, label).item()
            test_c += (out.argmax(1) == label).sum().item()

    history['train_loss'].append(train_l / len(train_loader))
    history['test_loss'].append(test_l / len(test_loader))
    history['train_acc'].append(100. * train_c / len(X_train))
    history['test_acc'].append(100. * test_c / len(X_test))

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1:02d} | Train Acc: {history['train_acc'][-1]:.2f}% | Test Acc: {history['test_acc'][-1]:.2f}%")

# ================= 4. 可视化与评估 =================

print(f"\n【模型信息】总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 绘制学习曲线
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(history['train_loss'], label='Train Loss')
ax[0].plot(history['test_loss'], label='Test Loss')
ax[0].set_title("损失函数收敛曲线")
ax[0].legend()

ax[1].plot(history['test_acc'], color='orange', label='Test Accuracy')
ax[1].set_title("测试集准确率随轮次变化")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy (%)")
ax[1].legend()
plt.show()

# 混淆矩阵
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for data, label in test_loader:
        y_pred.extend(model(data.to(device)).argmax(1).cpu().numpy())
        y_true.extend(label.numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap='Blues', values_format='d')
plt.title("测试集混淆矩阵")
plt.show()

print("\n--- 详细分类报告 ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))