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
# [开关] 如果没有数据文件，设为 True 以生成模拟数据进行测试
USE_DUMMY_DATA = False
FILE_PATH = '大唐天桥山电场齿轮箱数据.csv'

# 模型超参数
SEQ_LEN = 256
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 2
DROPOUT = 0.2
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# 设置中文字体 (避免绘图乱码，根据系统调整，Windows通常是SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 =================

def create_dummy_data(num_samples=2000, seq_len=128, num_classes=4):
    """生成模拟数据用于代码测试"""
    print(f"[提示]正在生成模拟数据 (样本数={num_samples}, 类别={num_classes})...")
    X = np.random.randn(num_samples, seq_len, 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    class_names = [f"故障_{i}" for i in range(num_classes)]
    return X, y, class_names


def load_data(file_path, seq_len):
    """读取CSV并切分数据"""
    if USE_DUMMY_DATA:
        return create_dummy_data(seq_len=seq_len, num_classes=4)

    try:
        df = pd.read_csv(file_path)
        print(f"成功读取文件: {file_path}")
        class_names = df.columns.tolist()
        print(f"检测到的故障类别: {class_names}")
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {file_path}")
        print("请将 USE_DUMMY_DATA 设为 True 或检查文件路径。")
        sys.exit()

    X_data = []
    y_data = []

    # 假设每一列代表一个故障类别
    for i, col in enumerate(df.columns):
        series = df[col].values
        # 截断数据以整除 seq_len
        num_samples = len(series) // seq_len
        series = series[:num_samples * seq_len]

        # 重塑: (样本数, 序列长度, 1)
        segments = series.reshape(-1, seq_len, 1)
        X_data.append(segments)

        # 生成标签
        y_data.append(np.full(segments.shape[0], i))

    X = np.concatenate(X_data, axis=0)
    y = np.concatenate(y_data, axis=0)
    return X, y, class_names


# 加载数据
print("--- 正在处理数据 ---")
X, y, CLASS_NAMES = load_data(FILE_PATH, SEQ_LEN)
NUM_CLASSES = len(CLASS_NAMES)  # 自动计算类别数
print(f"数据加载完成: X shape={X.shape}, y shape={y.shape}, 类别数={NUM_CLASSES}")

# 划分训练集和测试集 (8:2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化 (StandardScaler)
N_train, L, F = X_train.shape
N_test, _, _ = X_test.shape

scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, F)
X_test_flat = X_test.reshape(-1, F)

# 仅在训练集上fit
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, L, F)
X_test_scaled = scaler.transform(X_test_flat).reshape(N_test, L, F)

# 转为 Tensor
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")


# ================= 3. 模型定义 (Transformer) =================

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src: (Batch, Seq_Len, 1) -> (Batch, Seq_Len, d_model)
        src = self.input_linear(src)
        # -> (Seq_Len, Batch, d_model) (Transformer requirement)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Global Average Pooling: -> (Batch, Seq_Len, d_model) -> (Batch, d_model)
        output = output.permute(1, 0, 2)
        output = output.mean(dim=1)
        output = self.fc(output)
        return output


# 初始化模型
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


# ================= 4. 训练与评估循环 =================

def train_and_validate(model, train_loader, test_loader, epochs):
    print("\n--- 开始训练 ---")
    for epoch in range(epochs):
        # 训练
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
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证
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

        avg_train_loss = train_loss / train_total
        avg_train_acc = 100. * train_correct / train_total
        avg_test_loss = test_loss / test_total
        avg_test_acc = 100. * test_correct / test_total

        print(f"Epoch [{epoch + 1:02d}/{epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% "
              f"| Test Loss: {avg_test_loss:.4f} Acc: {avg_test_acc:.2f}%")


# 执行训练
train_and_validate(model, train_loader, test_loader, EPOCHS)

# ================= 5. 单样本预测演示 =================
print("\n--- 单样本预测演示 ---")


def predict_single_sample(model, sample_tensor, true_label, class_names):
    model.eval()
    with torch.no_grad():
        input_tensor = sample_tensor.unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

        print(f"真实标签: {true_label} ({class_names[true_label]})")
        print(f"预测类别: {prediction} ({class_names[prediction]})")
        print(f"类别概率: {probabilities.cpu().numpy()[0]}")
        if prediction == true_label:
            print("结果: 正确 ✅")
        else:
            print("结果: 错误 ❌")


# 从测试集取一个样本测试
sample_idx = 0
sample_data = test_dataset[sample_idx][0]
sample_label = test_dataset[sample_idx][1].item()
predict_single_sample(model, sample_data, sample_label, CLASS_NAMES)


# ================= 6. 混淆矩阵与详细评估 (新增部分) =================

def plot_confusion_matrix_and_report(model, loader, device, classes):
    print("\n--- 正在生成混淆矩阵与评估报告 ---")
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

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('混淆矩阵 (Confusion Matrix)')
    plt.ylabel('真实标签 (True Label)')
    plt.xlabel('预测标签 (Predicted Label)')
    plt.tight_layout()
    plt.show()

    # 打印分类报告
    print("\n--- 分类报告 (Classification Report) ---")
    # 注意：如果类别名中有中文，可能对齐会稍微乱一点，但内容是准确的
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))


# 执行评估
plot_confusion_matrix_and_report(model, test_loader, device, CLASS_NAMES)