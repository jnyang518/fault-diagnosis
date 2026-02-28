import torch
import torch.nn.functional as F
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据生成
N = 10240  # 序列长度
d = 64  # 嵌入维度

Q = torch.randn(N, d, device=device)
K = torch.randn(N, d, device=device)
V = torch.randn(N, d, device=device)


# --- 传统注意力机制 ---
def traditional_attention(Q, K, V):
    # 计算 QK^T
    # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d, dtype=torch.float32, device=device))
    # 注：图片中下一页的代码补全了计算过程
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d, dtype=torch.float32, device=device))
    #K.transpose(-2, -1) 的核心作用是将矩阵的最后两个维度进行转置（行列互换）。
    # 应用 Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 计算注意力输出
    output = torch.matmul(attention_weights, V)
    return output


# --- 稀疏注意力机制（固定步长的跳跃注意力） ---
def sparse_attention(Q, K, V, stride=16):
    N, d = Q.size()
    output = torch.zeros_like(Q, device=device)

    for i in range(0, N, stride):
        # 选择固定步长的跳跃位置
        indices = torch.arange(i, N, stride, device=device)
        K_sparse = K[indices]
        V_sparse = V[indices]

        # 计算稀疏点积
        # 注意：这里对当前分块 [i:i+1] 与稀疏后的 K 进行计算
        scores = torch.matmul(Q[i:i + 1], K_sparse.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d, dtype=torch.float32, device=device))

        attention_weights = F.softmax(scores, dim=-1)
        output[i:i + 1] = torch.matmul(attention_weights, V_sparse)

    return output


# --- 测量传统注意力机制的时间 ---
start_time = time.time()
output_traditional = traditional_attention(Q, K, V)
end_time = time.time()
traditional_time = end_time - start_time
print("Traditional Attention Time: {:.6f} seconds".format(traditional_time))

# --- 测量稀疏注意力机制的时间 ---
start_time = time.time()
output_sparse = sparse_attention(Q, K, V)
end_time = time.time()
sparse_time = end_time - start_time
print("Sparse Attention Time: {:.6f} seconds".format(sparse_time))

# --- 计算加速比 ---
speedup = traditional_time / sparse_time
print("Speedup: {:.2f}x".format(speedup))


def print_model_stats(N, d, stride, name):
    print(f"--- {name} 统计 ---")
    if name == "Traditional Attention":
        # 矩阵乘法 Q*K^T 的元素计算次数
        ops = N * N * d
        memory = N * N * 4 / (1024 ** 2)  # MB (float32)
    else:
        # 稀疏模式下的计算次数
        ops = N * (N / stride) * d
        memory = N * (N / stride) * 4 / (1024 ** 2)  # MB

    print(f"序列长度 N: {N}")
    print(f"可训练参数数量: 0 (纯函数运算)")
    print(f"理论计算量 (FLOPs): ~{ops:.0f} ")
    print(f"Score 矩阵内存占用: ~{memory:.2f} MB")
    print("-" * 30)


# 执行输出
print_model_stats(N, d, 16, "Traditional Attention")
print_model_stats(N, d, 16, "Sparse Attention")