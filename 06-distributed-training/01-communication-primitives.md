# 通信原语

> AllReduce、AllGather、ReduceScatter——这些是分布式训练中 GPU 之间交换数据的基础操作（称为“集合通信”）。理解它们的语义和实现算法是掌握分布式训练的前提。

## 为什么需要集合通信？

当你用 4 张 GPU 做数据并行训练时，每张卡用不同的数据计算出了自己的梯度。但参数更新时需要用**所有卡的平均梯度**——这意味着 4 张卡必须交换梯度并求和。怎么交换？总不能每张卡都把自己的梯度发给其他 3 张卡吧（通信量会爆炸）？

这就需要精心设计的**集合通信算法**：让 N 张卡用尽可能少的通信量完成数据交换。其中最核心的就是 **AllReduce**——“每张卡贡献自己的数据，全局求和后，每张卡都拿到相同的结果”。Ring AllReduce 算法让这个操作的通信量几乎不随 GPU 数量增长，是现代分布式训练的基础。

## 核心概念

### 集合通信操作（Collective Operations）

```
假设 4 个 GPU (rank 0-3)，每个持有一个向量:

Broadcast:        rank 0 的数据 → 所有 rank
  R0:[A]          R0:[A]  R1:[A]  R2:[A]  R3:[A]

Reduce:           所有 rank → 聚合到 rank 0
  R0:[A] R1:[B]   R0:[A+B+C+D]
  R2:[C] R3:[D]

AllReduce:        所有 rank → 聚合 → 每个 rank 都得到结果
  R0:[A] R1:[B]   R0:[A+B+C+D]  R1:[A+B+C+D]
  R2:[C] R3:[D]   R2:[A+B+C+D]  R3:[A+B+C+D]

AllGather:        收集所有 rank 的数据到每个 rank
  R0:[A] R1:[B]   R0:[ABCD]  R1:[ABCD]
  R2:[C] R3:[D]   R2:[ABCD]  R3:[ABCD]

ReduceScatter:    聚合 + 按 rank 分散
  R0:[a0,a1,a2,a3]   R0:[a0+b0+c0+d0]
  R1:[b0,b1,b2,b3]   R1:[a1+b1+c1+d1]
  R2:[c0,c1,c2,c3]   R2:[a2+b2+c2+d2]
  R3:[d0,d1,d2,d3]   R3:[a3+b3+c3+d3]

AlltoAll:         每个 rank 向每个 rank 发送不同数据
  R0:[a0,a1,a2,a3]   R0:[a0,b0,c0,d0]
  R1:[b0,b1,b2,b3]   R1:[a1,b1,c1,d1]
  R2:[c0,c1,c2,c3]   R2:[a2,b2,c2,d2]
  R3:[d0,d1,d2,d3]   R3:[a3,b3,c3,d3]
```

### AllReduce — 分布式训练最核心的操作

数据并行训练中，每一步梯度同步都需要 AllReduce：

```
Step 1: 每张卡用自己的数据算 local gradient
Step 2: AllReduce 将所有卡的梯度求平均 → 每张卡得到相同的平均梯度
Step 3: 每张卡用相同的平均梯度更新模型
→ 保证所有卡的模型参数始终一致
```

### NCCL (NVIDIA Collective Communications Library)

NCCL 是 NVIDIA 的集合通信库，专为 GPU 间通信优化：

- 自动检测 GPU 拓扑（NVLink、NVSwitch、PCIe、网络）
- 自动选择最优的通信算法
- 支持 InfiniBand、RoCE 等 RDMA 网络
- **几乎所有 PyTorch 分布式训练都用 NCCL 后端**

```python
# PyTorch 中初始化 NCCL 后端
import torch.distributed as dist
dist.init_process_group(backend='nccl')

# 手动调用 AllReduce
tensor = torch.randn(1024, device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
tensor /= dist.get_world_size()  # 求平均
```

## 关键细节

### Ring AllReduce 算法

最经典的 AllReduce 实现。分两个阶段：

```
4 个 GPU，数据分成 4 份 [a, b, c, d]

═══ Phase 1: ReduceScatter (N-1 步) ═══

每步：每个 GPU 向下一个 GPU 发送一份数据，接收上一个 GPU 发来的数据并累加

Step 1:
  GPU0: [a0, b0, c0, d0] → 发 d0 给 GPU1
  GPU1: [a1, b1, c1, d1] → 发 a1 给 GPU2
  GPU2: [a2, b2, c2, d2] → 发 b2 给 GPU3
  GPU3: [a3, b3, c3, d3] → 发 c3 给 GPU0

结果:
  GPU0: [a0, b0, c0+c3, d0]
  GPU1: [a1, b1, c1, d0+d1]
  GPU2: [a1+a2, b2, c2, d2]
  GPU3: [a3, b2+b3, c3, d3]

... (类似步骤重复 N-2 次)

Phase 1 完成后: 每个 GPU 持有结果的 1/N 份（完整的）

═══ Phase 2: AllGather (N-1 步) ═══

类似 Phase 1，但不做累加，只是把完整的份分发给所有 GPU
```

#### 通信量分析

```
N 个 GPU，数据大小 D bytes

Ring AllReduce:
  Phase 1 (ReduceScatter): 每步每 GPU 发送 D/N, 共 N-1 步 → (N-1)/N × D
  Phase 2 (AllGather):     每步每 GPU 发送 D/N, 共 N-1 步 → (N-1)/N × D
  
  总通信量 = 2(N-1)/N × D
  
  当 N 很大时 → 约 2D（与 GPU 数量无关！）
```

这就是 Ring AllReduce 的关键优势：**通信量不随 GPU 数量增长**（渐近意义上）。

#### Tree AllReduce

```
        Reduce Phase:                  Broadcast Phase:
         GPU0                             GPU0
        ╱    ╲                           ╱    ╲
    GPU0     GPU2                    GPU0     GPU2
    ╱  ╲     ╱  ╲                  ╱  ╲     ╱  ╲
  G0   G1  G2   G3              G0   G1  G2   G3

延迟: O(log N) 步    （vs Ring 的 O(N) 步）
带宽: 每步传 D bytes  （vs Ring 的 D/N bytes）

→ Tree 在小数据量时更好（延迟低）
→ Ring 在大数据量时更好（带宽利用率高）
```

NCCL 会根据数据大小自动选择 Ring 或 Tree 算法。

### 通信与计算的 Overlap

```
朴素做法:
  [Forward] → [Backward] → [AllReduce 全部梯度] → [Update]
  
Overlap 做法（PyTorch DDP 默认）:
  [Forward] → [Backward layer N] [AllReduce grad N]
                [Backward layer N-1] [AllReduce grad N-1]
                ...
  
  → 梯度通信和反向传播重叠，大幅减少总时间
```

### 网络拓扑对通信的影响

```
8 卡 DGX 机内:
  NVSwitch: 任意两卡 900 GB/s (H100)
  → AllReduce 非常快

多机之间:
  InfiniBand HDR: 200 Gbps (~25 GB/s) per link
  InfiniBand NDR: 400 Gbps (~50 GB/s) per link
  
  → 跨机通信比机内慢 ~20-40x!
  → 分布式策略要尽量减少跨机通信
```

```
常见拓扑策略（以 Megatron-LM 为例）:
  
  机内 (8卡, NVLink): Tensor Parallel (通信密集，要求低延迟)
  机间 (多节点):      Pipeline Parallel + Data Parallel (容忍更高延迟)
```

## 代码示例

### PyTorch 中使用集合通信

```python
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def demo_collectives(rank, world_size):
    setup(rank, world_size)
    
    # AllReduce
    tensor = torch.ones(1024, device=f'cuda:{rank}') * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # tensor 现在 = [1+2+3+4, 1+2+3+4, ...] = [10, 10, ...]
    
    # AllGather
    tensor = torch.ones(256, device=f'cuda:{rank}') * rank
    gather_list = [torch.zeros(256, device=f'cuda:{rank}') for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    # gather_list = [全0, 全1, 全2, 全3]
    
    # ReduceScatter
    input_list = [torch.ones(256, device=f'cuda:{rank}') * (i + rank) 
                  for i in range(world_size)]
    output = torch.empty(256, device=f'cuda:{rank}')
    dist.reduce_scatter(output, input_list)
    
    dist.destroy_process_group()
```

## 常见问题

**Q: AllReduce 和 Parameter Server 方案有什么区别？**

A: Parameter Server 使用中心化的服务器节点来聚合梯度，存在带宽瓶颈。AllReduce 是去中心化的——每个 GPU 既是发送者也是接收者。对于 GPU 集训，AllReduce 几乎完全替代了 PS 架构。

**Q: NCCL 和 Gloo 有什么区别？**

A: NCCL 是 NVIDIA 专为 GPU 设计的通信库，支持 NVLink 和 RDMA。Gloo 是 Facebook 开发的 CPU 通信库。GPU 训练用 NCCL，CPU 训练或小规模测试用 Gloo。

**Q: 如何判断训练是不是通信瓶颈？**

A: 用 PyTorch Profiler 或 Nsight Systems 看时间线：如果 GPU 在等待通信完成（compute 空闲而 nccl 在跑），就是通信瓶颈。关键指标是 **通信/计算比**——计算时间应该远大于通信时间才能高效 scale。

## 延伸阅读

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) — Ring AllReduce 经典文章
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **集合通信（Collective Communication）** | 多个进程（GPU）协作完成的数据交换操作的统称 |
| **AllReduce** | 每个 GPU 贡献自己的数据，全局聚合（如求和）后，每个 GPU 都收到相同的结果 |
| **AllGather** | 每个 GPU 将自己的数据广播给所有其他 GPU，最终每个 GPU 都持有全部数据 |
| **ReduceScatter** | 全局聚合后，结果被切分并分散到各 GPU（每个 GPU 只拿到结果的 1/N） |
| **All-to-All** | 每个 GPU 向每个其他 GPU 发送不同的数据块，类似矩阵转置。MoE 中用于 token 分发 |
| **Ring AllReduce** | 一种经典的 AllReduce 实现算法，N 个 GPU 排成环形传递数据，通信量与 GPU 数量几乎无关 |
| **NCCL** | NVIDIA Collective Communications Library，NVIDIA 的 GPU 集合通信库，自动选择最优通信算法和路径 |
| **Rank** | 分布式训练中每个参与进程的唯一编号，通常一个 rank 对应一张 GPU |
| **World Size** | 参与训练的总进程（GPU）数 |
