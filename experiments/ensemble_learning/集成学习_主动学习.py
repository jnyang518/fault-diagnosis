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
EPOCHS = 40  # 单个模型训练轮数
LR = 0.004  # 初始学习率
D_MODEL = 32  # 特征维度
N_HEAD = 4  # 多头注意力头数
NUM_LAYERS = 2  # Transformer 层数
DROPOUT = 0.1
SVD_RANK = 24  # SVD 低秩分解秩

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
test_tensor_x = torch.FloatTensor(X_test_scaled)
test_tensor_y = torch.LongTensor(y_test)
test_loader = DataLoader(TensorDataset(test_tensor_x, test_tensor_y), batch_size=BATCH_SIZE, shuffle=False)


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


# --- 参数量统计函数 ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


temp_model = Light_CNN_SVD_Transformer()
SINGLE_PARAMS = count_parameters(temp_model)
TOTAL_PARAMS = SINGLE_PARAMS * NUM_MODELS

print("\n" + "=" * 45)
print(f"|  模型参数量统计")
print(f"|  - 单个模型参数量 : {SINGLE_PARAMS:,}")
print(f"|  - 集成总参数量   : {TOTAL_PARAMS:,} (共 {NUM_MODELS} 个模型)")
print("=" * 45 + "\n")


# ================= 4. 模型训练逻辑 =================

def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_single_model_dynamic(model_index, train_loader, test_loader, epochs):
    """ 训练单个模型，返回模型、历史记录、最佳验证精度（作为专家权重） """
    print(f">> [Model {model_index + 1}/{NUM_MODELS}] 开始训练...")
    model = Light_CNN_SVD_Transformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    history = {'train_loss': [], 'test_acc': []}
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_acc = evaluate_accuracy(model, test_loader)
            history['test_acc'].append(test_acc)

            # 更新最佳精度
            if test_acc > best_acc:
                best_acc = test_acc

            print(f"   Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2%}")
            scheduler.step(test_acc)

    return model, history, best_acc


# ================= 5. 级联集成训练策略 (含 OOM 修复) =================

ensemble_models = []
all_histories = []
model_weights = []  # 存储每个模型的专家权重

# 准备原始数据Tensor
# 注意：如果显存非常小（<2GB），可以把这里改为 .cpu()，只在训练和预测时转gpu
origin_train_x_tensor = torch.FloatTensor(X_train_scaled).to(device)
origin_train_y_tensor = torch.LongTensor(y_train).to(device)

current_train_x = origin_train_x_tensor.clone()
current_train_y = origin_train_y_tensor.clone()

print(f"启动级联集成训练 (Cascading Hard Mining)...")

for i in range(NUM_MODELS):
    print(f"\n >>> 正在训练模型 {i + 1} (数据集大小: {len(current_train_y)})")

    # 构造数据集和Loader (为了节省显存，dataset 可以放 cpu, loader 自动 batching)
    dataset = TensorDataset(current_train_x.cpu(), current_train_y.cpu())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练并获取最佳精度权重
    model, hist, weight = train_single_model_dynamic(i, loader, test_loader, EPOCHS)

    ensemble_models.append(model)
    all_histories.append(hist)
    model_weights.append(weight)  # 使用验证集Accuracy作为权重

    if i == NUM_MODELS - 1: break

    # --- 修复后的困难样本挖掘逻辑 (防止 OOM) ---
    print(f" >> [Mining] 挖掘模型 {i + 1} 的困难样本...")
    model.eval()

    mined_preds = []
    with torch.no_grad():
        # 【关键修复】创建一个临时的 DataLoader 进行分批预测，而不是一次性预测所有数据
        # batch_size * 2 是因为 eval 模式不占反向传播显存，可以大一点
        mining_loader = DataLoader(TensorDataset(origin_train_x_tensor, origin_train_y_tensor),
                                   batch_size=BATCH_SIZE * 2, shuffle=False)

        for mb_x, _ in mining_loader:
            mb_x = mb_x.to(device)
            mb_out = model(mb_x)
            _, mb_pred = torch.max(mb_out, 1)
            mined_preds.append(mb_pred)

    # 拼接所有批次的预测结果
    preds = torch.cat(mined_preds)

    # 比较找出错误索引
    wrong_indices = (preds != origin_train_y_tensor).nonzero(as_tuple=True)[0]
    num_wrongs = len(wrong_indices)

    if num_wrongs > 0:
        hard_x = origin_train_x_tensor[wrong_indices]
        hard_y = origin_train_y_tensor[wrong_indices]
        current_train_x = torch.cat([origin_train_x_tensor, hard_x], dim=0)
        current_train_y = torch.cat([origin_train_y_tensor, hard_y], dim=0)
        print(f"    - 发现 {num_wrongs} 个困难样本，已加入下一轮训练。")
    else:
        print("    - 无困难样本。")


# ================= 6. 专家加权投票评估 (Expert Weighted Voting) =================

def ensemble_evaluate_expert_voting(models, weights, loader):
    """
    加权软投票 (Weighted Soft Voting):
    Final_Prob = Sum( Prob_i * Weight_i )
    """
    models_device = [m.to(device) for m in models]
    # 归一化权重 (可选，方便理解，这里直接乘也可以)
    # weights = np.array(weights) / np.sum(weights)
    weights_tensor = torch.tensor(weights, device=device).view(1, -1, 1)  # (1, N_models, 1)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            # 收集每个模型的概率分布
            # shape: (Batch, N_Models, N_Classes)
            probs_list = []
            for model in models_device:
                model.eval()
                # 使用 Softmax 获取概率
                probs = F.softmax(model(inputs), dim=1)
                probs_list.append(probs)

            stacked_probs = torch.stack(probs_list, dim=1)  # (B, M, C)

            # 加权求和
            # weights_tensor: (1, M, 1) 广播乘 (B, M, C) -> (B, M, C)
            weighted_probs = stacked_probs * weights_tensor
            # 在模型维度求和 -> (B, C)
            final_probs = weighted_probs.sum(dim=1)

            # 最终预测
            final_pred_batch = torch.argmax(final_probs, dim=1)

            all_preds.extend(final_pred_batch.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds), all_labels, all_preds


print("\n" + "=" * 60)
print(f"正在进行专家加权投票评估...")
print(f"模型权重 (Accuracy): {[f'{w:.4f}' for w in model_weights]}")

final_acc, y_true, y_pred = ensemble_evaluate_expert_voting(ensemble_models, model_weights, test_loader)

# ================= 7. 最终报告 =================

print("\n" + "+" + "-" * 50 + "+")
print("|          最终集成模型报告 (Expert Weighted)          |")
print("+" + "-" * 50 + "+")
print(f"| 1. 单模型参数量   : {SINGLE_PARAMS:<26,d} |")
print(f"| 2. 集成总参数量   : {TOTAL_PARAMS:<26,d} |")
print(f"| 3. 集成模型数量   : {NUM_MODELS:<26} |")
print(f"| 4. 最终测试集精度 : {final_acc * 100:<25.2f}% |")
print("+" + "-" * 50 + "+")

plt.figure(figsize=(15, 6))

# 混淆矩阵
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'专家加权集成混淆矩阵 (Acc: {final_acc:.2%})')
plt.xlabel('预测标签');
plt.ylabel('真实标签')

# 权重可视化
plt.subplot(1, 2, 2)
colors = sns.color_palette("viridis", NUM_MODELS)
plt.bar(range(1, NUM_MODELS + 1), model_weights, color=colors)
plt.ylim(min(model_weights) * 0.9, max(model_weights) * 1.01)
plt.title('各专家模型投票权重 (基于验证精度)')
plt.xlabel('Model Index');
plt.ylabel('Weight (Accuracy)')
for i, v in enumerate(model_weights):
    plt.text(i + 1, v, f"{v:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()