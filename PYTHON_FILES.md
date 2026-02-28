# Python 文件分类与作用说明

本仓库的 `.py` 文件基本都是**故障诊断相关的实验脚本**（多为单文件可运行脚本），围绕不同的特征表示与模型结构（Transformer / SVD 低秩 / 稀疏化 / 小波时频图 / 集成学习等）进行对比实验与可视化分析。

> 运行建议：多数脚本使用相对路径读取/写入数据与图片（如 `gearset20_0.csv`、`大唐天桥山电场齿轮箱数据.csv`、`merged_output.csv`）。**请在仓库根目录执行**：`python 子目录/脚本.py`，避免因为当前工作目录不同导致找不到数据文件。

## 目录结构（按“同类”归档）

- `analysis/`：数据/算法的辅助分析与小工具脚本
- `experiments/transformer_baseline/`：Transformer 基线模型（不做低秩/稀疏化）
- `experiments/svd_transformer/`：SVD 低秩/噪声补偿/稀疏化/共享层等“轻量化 Transformer”相关实验
- `experiments/wavelet_cwt_transformer/`：小波时频图（CWT）+ CNN/Transformer 的实验
- `experiments/ensemble_learning/`：集成学习（硬投票/加权软投票/困难样本挖掘）实验
- `experiments/cnn_resnet/`：CNN/ResNet（1D）结构的对比实验
- `experiments/kan/`：KAN（Kolmogorov–Arnold Network）相关实验

---

## `analysis/`（分析与工具）

- `analysis/plot_line.py`
  - 作用：最小示例，生成一张折线图并保存为 `line_plot.png`（用于验证绘图环境/导出是否正常）。
- `analysis/时域频域预分析.py`
  - 作用：对 `gearset20_0.csv` 的各列信号做**时域波形**与**FFT 频域幅值谱**的快速可视化，用于初步观察不同类别/通道的差异。
- `analysis/稀疏矩阵.py`
  - 作用：对比“传统全注意力”和“固定步长跳跃的稀疏注意力”的**运行时间**与**理论计算量/内存占用**（偏性能分析/原理验证）。

---

## `experiments/transformer_baseline/`（Transformer 基线）

- `experiments/transformer_baseline/transformer架构.py`
  - 作用：标准 Transformer Encoder 做序列分类的基线脚本（含数据读取、切片、标准化、训练/评估、混淆矩阵等），数据默认 `大唐天桥山电场齿轮箱数据.csv`。
- `experiments/transformer_baseline/transformer+残差链接+动态学习率.py`
  - 作用：在基线基础上加入 **Pre-Norm/残差稳定训练**、**梯度裁剪**与 **ReduceLROnPlateau 动态学习率**，用于观察训练稳定性与收敛变化。

---

## `experiments/svd_transformer/`（低秩 / 稀疏 / 共享层 / 噪声相关）

### 1) SVD 低秩 Transformer（Attention/FFN 压缩）

- `experiments/svd_transformer/奇异值分解的低秩近似.py`
  - 作用：用 `SVDLinear` 替代部分全连接层，在注意力投影上做低秩近似，展示“参数量压缩 + 分类训练”的完整流程。
- `experiments/svd_transformer/奇异值分解加上卷积嵌入.py`
  - 作用：在 SVD-Transformer 前加 `ConvEmbedding`（1D 卷积提取局部特征）再做 Transformer 分类。
- `experiments/svd_transformer/奇异值分解加上卷积嵌入利用30.2的数据.py`
  - 作用：与上一类相近，但数据源改为 `gearset30_2.csv`，并包含对 NaN 的处理与更长训练轮数配置。
- `experiments/svd_transformer/奇异值分解加上卷积嵌入后采用新的数据gearset.py`
  - 作用：与卷积嵌入 + SVD-Transformer 思路相同，数据源使用 `gearset20_0.csv`，并更强调“按列作为类别”的处理流程。

### 2) SVD + 噪声注入 / 噪声补偿（鲁棒性实验）

- `experiments/svd_transformer/SVD分解_随机噪声.py`
  - 作用：对注意力与 FFN 的线性层做 SVD 化，并在训练过程中注入随机噪声（提高鲁棒性/正则化），同时采用“共享层重复调用”的方式控制参数量。
- `experiments/svd_transformer/SVD分解_卷积嵌入_随机噪声.py`
  - 作用：在上一脚本基础上加入卷积嵌入（`ConvEmbedding`），并保持共享层与噪声注入机制。
- `experiments/svd_transformer/SVD分解_随机噪声补足_随机噪声.py`
  - 作用：更“结构化”的噪声补偿版本：根据当前权重统计量生成噪声项（并含余弦退火等训练策略），用于探索低秩近似带来的信息损失如何用噪声项补足。
- `experiments/svd_transformer/SVD分解_随机噪声补足_卷积嵌入_随机噪声.py`
  - 作用：在“噪声补足”思路下叠加卷积嵌入与共享层，属于更复杂的组合实验版本。

### 3) 低秩 + 稀疏（RPCA/共享空间）与极限轻量化探索

- `experiments/svd_transformer/低秩加稀疏加共享空间.py`
  - 作用：在 Transformer 的 FFN 中引入“低秩项 + 稀疏项”（并对稀疏项加 L1 惩罚），核心层使用共享结构重复堆叠，属于 RPCA/共享空间方向的轻量化尝试。
- `experiments/svd_transformer/线向量低秩_卷积嵌入.py`
  - 作用：将线性层按块（如 `64x64`）拆分并用 rank-1 外积参数化，配合共享层与噪声注入做极限参数压缩探索。
  - 备注：该脚本更偏研究/验证性质，包含重复训练循环/变量引用等风险点，若要作为稳定基线使用建议先做清理与单元验证。

---

## `experiments/wavelet_cwt_transformer/`（小波时频图 + 深度模型）

- `experiments/wavelet_cwt_transformer/小波变换.py`
  - 作用：对指定列的两段信号做 CWT（连续小波变换）并画出时域 + 时频图，用于直观解释“时频表示”的含义。
- `experiments/wavelet_cwt_transformer/小波变换加transformer.py`
  - 作用：把长序列切分为多个片段，每段做 CWT 得到图像序列；用 CNN 提取每张图的特征，再用 Transformer 对图像序列建模，最终分类。
- `experiments/wavelet_cwt_transformer/小波变换加transformer的不同batch.py`
  - 作用：与上一思路相近，但在数据切片方式/训练样本构造上更“控量”（例如限制训练/测试数据规模、把序列长度设为 1 等），用于探索 batch/样本量对训练的影响。
- `experiments/wavelet_cwt_transformer/小波变化+transformer+全部数据.py`
  - 作用：将全量数据按固定长度切分、每段生成 CWT 图像并训练 CNN+Transformer，属于“全量切片训练”的变体实现。

---

## `experiments/ensemble_learning/`（集成学习）

- `experiments/ensemble_learning/集成学习.py`
  - 作用：训练多个轻量模型并进行**硬投票（Hard Voting）**集成评估，输出混淆矩阵与集成效果。
- `experiments/ensemble_learning/集成学习_专家权重.py`
  - 作用：在硬投票基础上，增加“按单模型验证/测试表现归一化得到权重”的**加权软投票（Weighted Soft Voting）**，并可视化各专家权重。
- `experiments/ensemble_learning/集成学习_主动学习.py`
  - 作用：实现一种“级联困难样本挖掘（Cascading Hard Mining）”式的集成训练：后续模型在前序模型错误样本的基础上增广训练集，并用模型表现作为权重做加权投票评估。

---

## `experiments/cnn_resnet/`（CNN/ResNet 对比）

- `experiments/cnn_resnet/奇异值分解加上卷积嵌入无法适用于merged_output数据，transformer结构对于长序列的信号不具有很好的提取能力.py`
  - 作用：1D-ResNet18 风格网络对 `merged_output.csv` 的长序列切片做分类训练，并保存最优权重（如 `best_resnet1d.pth`），作为“长序列上 CNN/ResNet”对比实验。

---

## `experiments/kan/`（KAN）

- `experiments/kan/KAN的初版.py`
  - 作用：KAN（Kolmogorov–Arnold Network）线性层/两层网络的实现示例，包含数据读取/标准化/聚类生成标签等流程，用于验证 KAN 在该任务/数据形态上的可行性。

