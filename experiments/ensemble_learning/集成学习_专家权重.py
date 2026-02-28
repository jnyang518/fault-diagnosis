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
# 如果文件不存在，会自动切换为模拟数据

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
NUM_MODELS = 5  # 集成模型数量 (专家数量)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# 设置绘图字体，防止中文乱码
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
        # 不同类别生成不同频率的信号，加噪声
        freq1 = 5 + label * 3
        freq2 = 10 + label * 2
        noise = np.random.randn(seq_len) * 0.5
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


# ================= 4. 模型训练逻辑 (动态LR) =================

def evaluate_accuracy(model, loader):
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
    print(f">> [Model {model_index + 1}/{NUM_MODELS}] 启动训练...")
    model = Light_CNN_SVD_Transformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 动态学习率：当测试集Acc不增长时，LR减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=False)

    history = {'train_loss': [], 'test_acc': [], 'lr': []}

    for epoch in range(epochs):
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
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 记录测试集准确率
        test_acc = evaluate_accuracy(model, test_loader)
        history['test_acc'].append(test_acc)

        # 调度器步进
        scheduler.step(test_acc)

        # 简洁打印
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"   Epoch {epoch + 1}/{epochs} | Loss: {avg_train_loss:.4f} | Test Acc: {test_acc:.2%} | LR: {history['lr'][-1]:.6f}")

    return model, history


# ================= 5. 执行集成训练 =================

ensemble_models = []
all_histories = []

print(f"开始训练 {NUM_MODELS} 个独立专家模型...")
for i in range(NUM_MODELS):
    model, hist = train_single_model_dynamic(i, train_loader, test_loader, EPOCHS)
    ensemble_models.append(model)
    all_histories.append(hist)


# ================= 6. 集成评估：硬投票 & 加权软投票 =================

def ensemble_evaluate_hard_voting(models, loader):
    """ 硬投票：少数服从多数 """
    all_preds, all_labels = [], []
    for m in models: m.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            votes_list = []
            for model in models:
                _, pred = torch.max(model(inputs), 1)
                votes_list.append(pred)
            stacked_votes = torch.stack(votes_list, dim=1)
            final_pred_batch, _ = torch.mode(stacked_votes, dim=1)
            all_preds.extend(final_pred_batch.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


def calculate_expert_weights(histories):
    """ 计算专家权重：基于最终测试集准确率 """
    final_accs = [h['test_acc'][-1] for h in histories]
    acc_tensor = torch.tensor(final_accs)
    # 归一化权重
    weights = acc_tensor / acc_tensor.sum()
    return weights.tolist(), final_accs


def ensemble_evaluate_weighted_soft(models, loader, weights):
    """ 加权软投票：概率加权求和 """
    model_device = next(models[0].parameters()).device
    total, correct = 0, 0
    weights_tensor = torch.tensor(weights, device=model_device).view(-1, 1, 1)  # [Num_Models, 1, 1]

    for m in models: m.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(model_device), labels.to(model_device)
            batch_probs = []
            for model in models:
                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs)

            # [Num_Models, Batch_Size, Num_Classes]
            batch_probs = torch.stack(batch_probs)
            # 加权平均 -> [Batch_Size, Num_Classes]
            weighted_avg_probs = (batch_probs * weights_tensor).sum(dim=0)

            _, predicted = torch.max(weighted_avg_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# --- 最终评估 ---
print("\n" + "=" * 60)
print(">>> 1. 计算基准硬投票 (Hard Voting)...")
hard_acc = ensemble_evaluate_hard_voting(ensemble_models, test_loader)

print(">>> 2. 计算专家权重 (Expert Weights)...")
expert_weights, model_accs = calculate_expert_weights(all_histories)
for i, (acc, w) in enumerate(zip(model_accs, expert_weights)):
    print(f"   Model {i + 1}: Acc={acc:.2%} -> Weight={w:.4f}")

print(">>> 3. 计算加权软投票 (Weighted Soft Voting)...")
soft_acc = ensemble_evaluate_weighted_soft(ensemble_models, test_loader, expert_weights)

# ================= 7. 结果展示与对比 =================

print("\n" + "+" + "-" * 55 + "+")
print("|             最终集成策略效果对比             |")
print("+" + "-" * 55 + "+")
print(f"| 硬投票准确率 (Hard Voting)     : {hard_acc * 100:.2f}%           |")
print(f"| 加权软投票准确率 (Weighted Soft) : {soft_acc * 100:.2f}%           |")
delta = soft_acc - hard_acc
symbol = "+" if delta >= 0 else ""
print(f"| >>> 策略提升                   : {symbol}{delta * 100:.2f}%           |")
print("+" + "-" * 55 + "+")

# 绘图
plt.figure(figsize=(12, 5))

# 子图1: 权重可视化
plt.subplot(1, 2, 1)
bars = plt.bar([f"M{i + 1}" for i in range(NUM_MODELS)], expert_weights, color='skyblue', edgecolor='black')
plt.title('各模型专家权重分配')
plt.ylabel('权重 (Weight)')
plt.ylim(min(expert_weights) * 0.9, max(expert_weights) * 1.1)
for bar, acc in zip(bars, model_accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.1%}", ha='center', va='bottom', fontsize=9)

# 子图2: 训练Loss趋势
plt.subplot(1, 2, 2)
for i, h in enumerate(all_histories):
    plt.plot(h['train_loss'], label=f'Model {i + 1}', alpha=0.5)
plt.title('各模型训练 Loss 下降趋势')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()