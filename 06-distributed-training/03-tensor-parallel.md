# 张量并行（Tensor Parallelism）

> 将单个层的矩阵运算切分到多张卡上并行计算。当模型的单层参数就超过单卡显存时，必须使用张量并行。由于每层都需要通信，它要求卡之间有很高的互联带宽（如 NVLink），因此通常只在同一台机器内使用。

## 核心概念

### 为什么需要张量并行？

数据并行要求每张卡存一份完整模型。当单层的参数/激活值就超过单卡显存时，数据并行无能为力。

张量并行将**单个矩阵乘法**切分到多张卡上并行计算。

### Megatron-style 张量并行

以 Transformer 的 FFN 层为例：$Y = \text{GeLU}(XW_1) W_2$

**列切分（Column Parallel）：切分 $W_1$**

```
           W1 ──── 列切分 ──→ [W1_0 | W1_1]
           
GPU 0:  Y0 = GeLU(X · W1_0)     ← X 通过 f (identity) 传入
GPU 1:  Y1 = GeLU(X · W1_1)     ← X 通过 f (identity) 传入

注意: GeLU 是非线性函数
  GeLU(AB) ≠ GeLU(A) + GeLU(B)
  但 GeLU([X·W1_0]) 和 GeLU([X·W1_1]) 分别计算是正确的！
  因为列切分后每张卡拿到完整输入 X 的子空间投影
```

**行切分（Row Parallel）：切分 $W_2$**

```
           W2 ──── 行切分 ──→ [W2_0]
                               [W2_1]

GPU 0:  Z0 = Y0 · W2_0     ← Y0 从上一步的列切分来
GPU 1:  Z1 = Y1 · W2_1     ← Y1 从上一步的列切分来

Z = Z0 + Z1  ← 需要 AllReduce (ḡ 操作)
```

**完整的 FFN 张量并行流程：**

```
      Input X
         │
    f (identity/拷贝)
      ╱      ╲
   GPU 0     GPU 1
  X·W1_0    X·W1_1        ← Column Parallel (无通信)
  GeLU()    GeLU()
  ×W2_0     ×W2_1         ← Row Parallel
      ╲      ╱
    ḡ (AllReduce)          ← 一次通信
         │
      Output Z
```

**MHA（Multi-Head Attention）的张量并行：**

```
多头注意力天然适合切分——每张卡分几个 head:

8 heads, 2 GPUs:
  GPU 0: Head 0-3 的 Q, K, V 计算 + Attention + Output 投影
  GPU 1: Head 4-7 的 Q, K, V 计算 + Attention + Output 投影
  
  → AllReduce 合并输出

每个 Transformer 层需要 2 次 AllReduce:
  1. MHA 的输出 AllReduce
  2. FFN 的输出 AllReduce
```

## 关键细节

### $f$ 和 $\bar{f}$ 操作

论文中定义了两对操作：

| 操作 | Forward | Backward |
|------|---------|----------|
| $f$ | Identity（直接拷贝） | AllReduce（聚合梯度） |
| $\bar{f}$ | AllReduce（聚合输出） | Identity（直接拷贝） |

- 列并行的输入端用 $f$：forward 不通信，backward AllReduce 梯度
- 行并行的输出端用 $\bar{f}$：forward AllReduce 结果，backward 不通信

**两次 AllReduce 在 forward 和 backward 中分别发生一次→ 总通信量最小化。**

### 通信量分析

```
对于 hidden_size = h, 序列长度 = s, batch_size = b, dtype = FP16 (2 bytes):

每次 AllReduce 的数据量:
  data_size = b × s × h × sizeof(dtype)   // 即 b × s × h × 2 bytes

TP degree = T 时:
  单次 AllReduce 通信量 = 2 × (T-1)/T × data_size
  （Ring AllReduce 的 ReduceScatter + AllGather 两阶段，系数 2 来自此处）

每个 Transformer 层:
  Forward:  2 次 AllReduce (MHA 输出 + FFN 输出)
  Backward: 2 次 AllReduce (对应梯度)
  共 4 次 AllReduce

  每层总通信量 = 4 × 2 × (T-1)/T × data_size
              = 8(T-1)/T × b·s·h × sizeof(dtype)

→ 与 hidden_size × seq_len 成正比 → 需要高带宽互联
```

### 为什么 TP 必须放在机内？

```
TP 的通信特点:
  - 频率高: 每一层都要通信（而非每个 step 通信一次）
  - 延迟敏感: 通信在计算的关键路径上（不能完全 overlap）
  - 数据量: 和 hidden_size × seq_len 成正比

NVLink (900 GB/s) vs InfiniBand (50 GB/s) → 18x 差距
→ TP 跨机会严重拖慢训练
```

### 序列并行 (Sequence Parallelism)

Megatron-LM v2 引入的优化：将 non-tensor-parallel 的操作（LayerNorm、Dropout）也沿序列维度切分：

```
标准 TP:
  LayerNorm → [TP Region] → LayerNorm → [TP Region]
  ↑ 这些在每张卡上都是完整的序列 → 冗余显存

序列并行:
  LayerNorm (seq/T) → AllGather → [TP Region] → ReduceScatter → LayerNorm (seq/T)
  ↑ 非 TP 区域也切分了 → 进一步节省显存
  ↑ 把 AllReduce 分解为 AllGather + ReduceScatter，通信量不变
```

## 代码示例

```python
# Megatron-style Column Parallel Linear
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_group):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_group = tp_group
        
        # 每张卡只持有 output_size/tp_size 列
        self.output_size_per_partition = output_size // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )
    
    def forward(self, x):
        # f: forward = identity, backward = allreduce
        output = F.linear(x, self.weight)  # 本地矩阵乘
        return output  # 返回部分结果，由后续的 RowParallel 处理


class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        
        # 每张卡只持有 input_size/tp_size 行
        self.input_size_per_partition = input_size // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )
    
    def forward(self, x):
        output = F.linear(x, self.weight)  # 本地矩阵乘
        # ḡ: forward = allreduce, backward = identity
        dist.all_reduce(output, group=self.tp_group)
        return output
```

## 常见问题

**Q: TP 和 DP 能同时用吗？**

A: 可以，而且这是标准做法。例如 64 张 H100（8 台 8 卡机器）：TP=8（机内 8 卡 NVLink）× DP=8（跨 8 台机器）。

**Q: TP degree 设多大合适？**

A: 通常 = 单机 GPU 数量（如 8 for DGX）。增加 TP degree 要求更高的互联带宽，跨机 TP 通常得不偿失。如果模型不太大，TP=2 或 TP=4 可能就够。

**Q: 为什么 Attention 的 head 数必须能被 TP degree 整除？**

A: 因为 TP 把 head 均匀分配到各卡。如果不能整除（如 6 heads, TP=4），就会出现负载不均衡。所以大模型设计时通常保证 num_heads 是 2 的幂或能被常见 TP degree 整除。

## 延伸阅读

- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2020
- [Megatron-LM v2: Sequence Parallelism](https://arxiv.org/abs/2205.05198) — Korthikanti et al., 2022
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **张量并行（Tensor Parallelism, TP）** | 将单层的矩阵按列或按行切分到多张卡上并行计算 |
| **列切分（Column Parallel）** | 将权重矩阵按列切分，每张卡计算输出的一部分 |
| **行切分（Row Parallel）** | 将权重矩阵按行切分，各卡的输出需要 AllReduce 合并 |
| **$f$ 和 $\bar{f}$** | Megatron 论文中的符号。$f$: forward 时不通信，backward 时 AllReduce；$\bar{f}$: 反过来 |
| **序列并行（Sequence Parallelism）** | 将非 TP 区域的操作（LayerNorm、Dropout）也沿序列维度切分，进一步节省显存 |
| **TP Degree** | 张量并行的度，即参与切分的 GPU 数。通常 = 机内 GPU 数（如 8） |
