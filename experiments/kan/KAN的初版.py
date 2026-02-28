import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math
import os


# ==========================================
# 1. KAN (Kolmogorov-Arnold Network) 实现
# ==========================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                       -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # [Fix 1] 修复初始化时的维度不匹配问题
            noise = (torch.rand(self.grid_size + 1, self.in_features,
                                self.out_features) - 1 / 2) * self.scale_noise / self.grid_size

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        # 1. Base Output
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 2. Spline Output
        x_bs = self.b_splines(x).view(x.size(0), -1)

        # [Fix 2] 修复 Forward 时的维度不匹配问题
        if self.enable_standalone_scale_spline:
            # 先对权重进行缩放，再进行线性变换
            scaled_spline_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
            weight_for_linear = scaled_spline_weight.view(self.out_features, -1)
        else:
            weight_for_linear = self.spline_weight.view(self.out_features, -1)

        spline_output = F.linear(x_bs, weight_for_linear)

        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)


class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KAN, self).__init__()
        self.layer1 = KANLinear(input_dim, hidden_dim)
        self.layer2 = KANLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# ==========================================
# 2. 数据处理与加载
# ==========================================
def load_and_process_data(file_path):
    if not os.path.exists(file_path):
        print(f"[警告] 找不到文件 {file_path}，正在生成随机模拟数据以供测试...")
        X = np.random.rand(1000, 4)
        y = np.random.randint(0, 2, size=1000)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    print(f"正在读取数据: {file_path} ...")
    df = pd.read_csv(file_path)

    feature_cols = ['label_1', 'label_2', 'label_3', 'label_4']
    X = df[feature_cols].values

    print("正在生成模拟故障标签 (基于K-Means聚类)...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    y = kmeans.fit_predict(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # 配置
    FILE_NAME = '大唐天桥山电场齿轮箱数据.csv'
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 1. 准备数据
    X, y = load_and_process_data(FILE_NAME)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 8:2 切分
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 2. 初始化模型
    input_dim = 4
    hidden_dim = 8
    output_dim = 2

    model = KAN(input_dim, hidden_dim, output_dim).to(DEVICE)

    # ==========================================
    # [新增] 统计并输出模型参数量
    # ==========================================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("--------------------------------------------------")
    print(f"模型结构:\n{model}")
    print(f"★ 模型总参数量 (Trainable Parameters): {total_params}")
    print("--------------------------------------------------")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 3. 训练循环
    print("\n开始训练 KAN 模型...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100 * correct_train / total_train if total_train > 0 else 0

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val if total_val > 0 else 0

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val/Test Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()