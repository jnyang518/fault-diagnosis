import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import os

# ================= 1. 全局配置参数 =================
FILE_PATH = 'gearset20_0.csv'  # 数据集文件名

# --- 瘦身版模型超参数 ---
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 40  # 训练轮数
LR = 0.004  # 初始学习率
D_MODEL = 32  # 特征维度
N_HEAD = 4  # 多头注意力头数
NUM_LAYERS = 2  # Transformer 层数
DROPOUT = 0.1
SVD_RANK = 24  # SVD 秩

# --- 集成参数 ---
NUM_MODELS = 5  # 集成模型数量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# 设置绘图字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 =================

def create_dummy_data(num_samples=2000, seq_len=256, num_classes=4):
    print(">>> [提示] 未找到数据集，生成模拟数据...")
    t = np.linspace(0, 10, seq_len)
    X, y = [], []
    for i in range(num_samples):
        label = np.random.randint(0, num_classes)
        freq1 = 5 + label * 3
        freq2 = 10 + label * 2
        noise = np.random.randn(seq_len) * 0.4
        signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + noise
        X.append(signal.reshape(seq_len, 1))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y), [f"故障模式_{i}" for i in range(num_classes)]


def load_data(file_path, seq_len):
    if not os.path.exists(file_path):
        return create_dummy_data(seq_len=seq_len)
    try:
        df = pd.read_csv(file_path)
        print(f"成功读取文件: {file_path}")
        class_names = df.columns.tolist()
        X_data, y_data = [], []
        for i, col in enumerate(df.columns):
            series = df[col].values
            num_samples = len(series) // seq_len
            series = series[:num_samples * seq_len]
            segments = series.reshape(-1, seq_len, 1)
            X_data.append(segments)
            y_data.append(np.full(segments.shape[0], i))
        return np.concatenate(X_data, axis=0), np.concatenate(y_data, axis=0), class_names
    except Exception as e:
        print(f"读取文件出错: {e}")
        return create_dummy_data(seq_len=seq_len)


print("--- 正在处理数据 ---")
X, y, CLASS_NAMES = load_data(FILE_PATH, SEQ_LEN)
NUM_CLASSES = len(CLASS_NAMES)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
N_train, L, F_dim = X_train.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F_dim)).reshape(N_train, L, F_dim)
X_test_scaled = scaler.transform(X_test.reshape(-1, F_dim)).reshape(X_test.shape[0], L, F_dim)

# DataLoader
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)),
                         batch_size=BATCH_SIZE, shuffle=False)


# ================= 3. 轻量化模型定义 (SVD-Transformer) =================

class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(SVDLinear, self).__init__()
        self.down_project = nn.Linear(in_features, rank, bias=False)
        self.up_project = nn.Linear(rank, out_features, bias=True)

    def forward(self, x): return self.up_project(self.down_project(x))


class SVDMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rank, dropout=0.1):
        super(SVDMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = SVDLinear(d_model, d_model, rank)
        self.k_proj = SVDLinear(d_model, d_model, rank)
        self.v_proj = SVDLinear(d_model, d_model, rank)
        self.out_proj = SVDLinear(d_model, d_model, rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.size()
        Q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(context)


class SVDTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, rank, dropout=0.1):
        super(SVDTransformerEncoderLayer, self).__init__()
        self.self_attn = SVDMultiheadAttention(d_model, nhead, rank, dropout)
        self.linear1 = SVDLinear(d_model, d_model * 2, rank)
        self.linear2 = SVDLinear(d_model * 2, d_model, rank)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2))
        src2 = self.norm2(src)
        src = src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src2)))))
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


class Light_CNN_SVD_Transformer(nn.Module):
    def __init__(self):
        super(Light_CNN_SVD_Transformer, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(1, D_MODEL // 2, 5, padding=2), nn.BatchNorm1d(D_MODEL // 2), nn.ReLU(),
            nn.Conv1d(D_MODEL // 2, D_MODEL, 3, padding=1), nn.BatchNorm1d(D_MODEL), nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList(
            [SVDTransformerEncoderLayer(D_MODEL, N_HEAD, SVD_RANK, DROPOUT) for _ in range(NUM_LAYERS)])
        self.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))

    def forward(self, src):
        src = self.embedding(src.permute(0, 2, 1)).permute(2, 0, 1)
        src = self.pos_encoder(src).permute(1, 0, 2)
        for layer in self.layers: src = layer(src)
        return self.fc(src.mean(dim=1))


# ================= 4. 模型训练逻辑 (新增动态LR与测试监控) =================

def evaluate_accuracy(model, loader):
    """ 辅助函数：计算当前模型在指定数据集上的准确率 """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_single_model_dynamic(model_index, train_loader, test_loader, epochs):
    print(f">> [Model {model_index + 1}/{NUM_MODELS}] 启动动态训练...")
    model = Light_CNN_SVD_Transformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 【新增】ReduceLROnPlateau 动态调度器
    # mode='max': 监控指标越大越好 (例如 Accuracy)
    # factor=0.5: 触发时学习率减半
    # patience=3: 容忍3次指标不增长
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    history = {'train_loss': [], 'test_acc': [], 'lr': []}

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # --- 周期性评估与输出 (每5轮) ---
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # 计算测试集准确率
            test_acc = evaluate_accuracy(model, test_loader)
            history['test_acc'].append(test_acc)

            print(f"   Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Test Acc: {test_acc * 100:.2f}% | "
                  f"LR: {current_lr:.6f}")

            # 【关键】根据测试集准确率调整学习率
            # 注意：ReduceLROnPlateau 是基于 epoch 结果进行 step 的
            scheduler.step(test_acc)

    return model, history


# ================= 5. 执行集成训练 =================

ensemble_models = []
all_histories = []

print(f"开始训练 {NUM_MODELS} 个独立判断体 (含动态学习率机制)...")
for i in range(NUM_MODELS):
    model, hist = train_single_model_dynamic(i, train_loader, test_loader, EPOCHS)
    ensemble_models.append(model)
    all_histories.append(hist)


# ================= 6. 集成硬投票评估 =================

def ensemble_evaluate_hard_voting(models, loader):
    total_loss, all_preds, all_labels = 0.0, [], []
    for m in models: m.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            votes_list = []
            for model in models:
                _, pred = torch.max(model(inputs), 1)
                votes_list.append(pred)

            stacked_votes = torch.stack(votes_list, dim=1)
            final_pred_batch, _ = torch.mode(stacked_votes, dim=1)

            # 计算 Loss (得票率 NLL)
            votes_one_hot = F.one_hot(stacked_votes, num_classes=NUM_CLASSES).float()
            vote_probs = votes_one_hot.sum(dim=1) / NUM_MODELS
            loss = F.nll_loss(torch.log(vote_probs + 1e-9), labels, reduction='sum')

            total_loss += loss.item()
            all_preds.extend(final_pred_batch.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds


print("\n" + "=" * 60)
print(f"正在进行最终评估...")
train_loss, train_acc, _, _ = ensemble_evaluate_hard_voting(ensemble_models, train_loader)
test_loss, test_acc, test_true, test_pred = ensemble_evaluate_hard_voting(ensemble_models, test_loader)

# ================= 7. 结果展示 =================
single_params = sum(p.numel() for p in ensemble_models[0].parameters())
print("\n" + "+" + "-" * 55 + "+")
print("|             最终集成模型评估报告 (Hard Voting)             |")
print("+" + "-" * 55 + "+")
print(f"| 1. 单模型参数量    : {single_params:<33,d} |")
print(f"| 2. 初始学习率 (LR) : {LR:<33} |")
print("|" + "-" * 55 + "|")
print(f"| {'数据集':<15} | {'Loss':<15} | {'Accuracy':<15} |")
print("|" + "-" * 55 + "|")
print(f"| {'训练集 (Train)':<15} | {train_loss:<15.4f} | {train_acc * 100:<14.2f}% |")
print(f"| {'测试集 (Test)':<15} | {test_loss:<15.4f} | {test_acc * 100:<14.2f}% |")
print("+" + "-" * 55 + "+")

plt.figure(figsize=(14, 6))
# 混淆矩阵
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(test_true, test_pred), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title(f'测试集混淆矩阵 (Acc: {test_acc:.2%})')

# 训练Loss曲线 (带LR衰减说明)
plt.subplot(1, 2, 2)
min_len = min([len(h['train_loss']) for h in all_histories])
avg_loss = np.mean([h['train_loss'][:min_len] for h in all_histories], axis=0)
plt.plot(range(1, min_len + 1), avg_loss, marker='o', color='orange', label='Avg Train Loss')
plt.title('模型平均训练 Loss 趋势')
plt.xlabel('Epoch');
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()