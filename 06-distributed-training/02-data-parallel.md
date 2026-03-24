# 数据并行：DDP、FSDP 与 ZeRO

> 数据并行是最常用的分布式训练策略，理解 DDP → ZeRO → FSDP 的演进是核心。

## 核心概念

### 数据并行的基本思想

```
每张卡持有完整的模型副本，输入数据被分成 N 份:

GPU 0: Model Copy + Data Shard 0 → Gradient 0 ─┐
GPU 1: Model Copy + Data Shard 1 → Gradient 1 ─┤ AllReduce → Avg Gradient
GPU 2: Model Copy + Data Shard 2 → Gradient 2 ─┤
GPU 3: Model Copy + Data Shard 3 → Gradient 3 ─┘

每张卡用相同的平均梯度更新 → 模型始终同步
等效 batch size = per_gpu_batch × N
```

### DP vs DDP

| 方面 | `torch.nn.DataParallel` (DP) | `torch.nn.parallel.DistributedDataParallel` (DDP) |
|------|---------------------------|--------------------------------------------------|
| 进程模型 | 单进程多线程 | **多进程**（每 GPU 一个进程） |
| 通信 | GPU 0 收集梯度（瓶颈） | **AllReduce**（对等通信） |
| GIL 影响 | 受 Python GIL 限制 | 不受（独立进程） |
| 性能 | 差（不推荐使用） | **好** |
| 代码 | `model = nn.DataParallel(model)` | `model = DDP(model)` |

**结论：永远用 DDP，不要用 DP。**

### DDP 的工作流程

```python
# DDP 核心流程
model = DDP(model, device_ids=[local_rank])

for batch in dataloader:
    loss = model(batch)      # 1. 各卡独立 forward
    loss.backward()          # 2. 反向传播，自动触发梯度 AllReduce
                             #    DDP 用 bucket 机制: 将梯度分组，
                             #    边计算边通信（overlap）
    optimizer.step()         # 3. 各卡用相同的平均梯度更新
    optimizer.zero_grad()
```

**Gradient Bucketing**：DDP 不是等所有梯度算完再 AllReduce，而是将参数分成 bucket（默认 25MB），每个 bucket 满了就立即开始 AllReduce，与后续层的反向传播重叠。

### 显存瓶颈：为什么 DDP 不够

```
训练一个 7B 参数的模型 (FP16):

模型参数:        7B × 2 bytes = 14 GB
梯度:            7B × 2 bytes = 14 GB
Adam 优化器状态:  7B × (4+4+4) bytes = 84 GB  ← 最大的开销！
  - FP32 参数副本:   7B × 4 = 28 GB
  - FP32 momentum:   7B × 4 = 28 GB
  - FP32 variance:   7B × 4 = 28 GB

总计: 14 + 14 + 84 = 112 GB  > 单卡 80 GB

问题: DDP 中每张卡都保存完整的 112 GB → 显存浪费！
```

### ZeRO：切分冗余状态

**ZeRO (Zero Redundancy Optimizer)** 的核心洞察：DDP 中每张卡保存了完整的模型参数/梯度/优化器状态，这些都是**冗余的**。

```
ZeRO Stage 1: 切分优化器状态 (OS)
  每张卡只保存 1/N 的 optimizer states
  显存节省: ~4x (84 GB → 21 GB per GPU with 4 GPUs)
  通信: 和 DDP 相同

ZeRO Stage 2: 切分优化器状态 + 梯度 (OS+G)
  每张卡只保存 1/N 的 optimizer states 和 1/N 的 gradients
  显存节省: ~8x
  通信: 用 ReduceScatter 替代 AllReduce（通信量相同）

ZeRO Stage 3: 切分所有 (OS+G+P)
  每张卡只保存 1/N 的 optimizer states、gradients 和 parameters
  显存节省: ~N × (线性扩展！)
  通信: forward/backward 时需要 AllGather 收集参数
         相比 DDP/ZeRO-2 的 2Φ，ZeRO-3 为 3Φ（多 50%）
```

```
ZeRO 各 Stage 显存对比（7B 模型，4 GPU）:

                    DDP      ZeRO-1    ZeRO-2    ZeRO-3
参数 (FP16)         14 GB    14 GB     14 GB     3.5 GB
梯度 (FP16)         14 GB    14 GB     3.5 GB    3.5 GB
Optimizer (FP32)    84 GB    21 GB     21 GB     21 GB
────────────────────────────────────────────────────────
总计/卡             112 GB   49 GB     38.5 GB   28 GB
```

### FSDP (Fully Sharded Data Parallel)

FSDP 是 PyTorch 官方的 ZeRO Stage 3 实现：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# 类似 ZeRO Stage 3
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    # ShardingStrategy.SHARD_GRAD_OP → ZeRO-2
    # ShardingStrategy.NO_SHARD → DDP
)
```

FSDP 的执行流程：

```
Forward Pass:
  层 1: AllGather 参数 → 计算 → 释放非本地参数
  层 2: AllGather 参数 → 计算 → 释放非本地参数
  ...

Backward Pass:
  层 N: AllGather 参数 → 计算梯度 → ReduceScatter 梯度 → 释放非本地参数
  层 N-1: AllGather 参数 → 计算梯度 → ReduceScatter 梯度 → ...
  ...

→ 任何时刻，只有当前正在计算的层的参数在显存中
→ 显存消耗 ≈ 最大单层参数 + 1/N 的总参数/梯度/优化器
```

## 关键细节

### 通信量对比

| 策略 | 每步每卡通信量 | 说明 |
|------|--------------|------|
| DDP / ZeRO-1 | $2\Phi$ | AllReduce 梯度 |
| ZeRO-2 | $2\Phi$ | ReduceScatter 梯度 + AllGather 更新后参数 |
| ZeRO-3 / FSDP | $3\Phi$ | Forward AllGather + Backward AllGather + ReduceScatter |

其中 $\Phi$ = 参数量 × sizeof(dtype)。

ZeRO-3 / FSDP 的 $3\Phi$ 相比 DDP / ZeRO-2 的 $2\Phi$ 多了 50%。代价是每次 forward 和 backward 都需要额外的 AllGather 来收集当前层的完整参数。这是**通信换显存的 trade-off**——通信量增加 50%，但显存节省与 GPU 数量线性成正比。

### 混合精度 + ZeRO 的显存计算

```
Param (FP16): Φ × 2 bytes, 但 ZeRO-3 只存 1/N → Φ × 2 / N
Grad (FP16): 同上 → Φ × 2 / N
Optimizer (FP32 copy + momentum + variance): Φ × 12 / N (ZeRO-1+)
Activations: 取决于 batch size 和 checkpoint 策略（独立于 ZeRO）

总显存/卡 ≈ Φ × 16 / N + Activations
```

### FSDP 的关键配置

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# 混合精度
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# 自动按 Transformer 层切分
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取下一层参数
    # cpu_offload=CPUOffload(offload_params=True),     # 参数卸载到 CPU
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,  # 控制同时进行的 AllGather 数量
)
```

## 代码示例

### 完整的 DDP 训练脚本

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 模型
    model = MyModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # 数据：DistributedSampler 确保每张卡拿到不同的数据
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 每 epoch 重新 shuffle
        for batch in dataloader:
            batch = batch.cuda(rank)
            loss = model(batch).loss
            loss.backward()      # 自动 AllReduce 梯度
            optimizer.step()
            optimizer.zero_grad()
    
    dist.destroy_process_group()

# 启动: torchrun --nproc_per_node=4 train.py
```

## 常见问题

**Q: DDP 的 `find_unused_parameters=True` 是做什么的？**

A: 如果模型有条件执行的分支（某些参数在 forward 中没被用到），DDP 默认会报错。设置 `find_unused_parameters=True` 允许 DDP 跳过未使用参数的梯度同步。但这有性能开销，应尽量避免。

**Q: ZeRO-3 / FSDP 的性能比 DDP 差多少？**

A: 通信量多 ~50%。在机内（NVLink 带宽充足时）性能差距约 5-15%。但 FSDP 能训练更大的模型或用更大的 batch size，实际吞吐量可能反而更高。

**Q: FSDP 和 DeepSpeed ZeRO-3 该选哪个？**

A: PyTorch 原生生态建议 FSDP（更好的 PyTorch 集成、`torch.compile` 支持）。性能上两者接近。DeepSpeed 在易用性（配置化）和额外功能（如 CPU offload、NVMe offload）上更丰富。

## 延伸阅读

- [ZeRO 论文](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2020
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Getting Started with DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **数据并行（Data Parallelism）** | 每张卡持有完整的模型副本，输入数据被切分到各卡并行计算，梯度通过 AllReduce 同步 |
| **DDP** | DistributedDataParallel，PyTorch 官方的多进程数据并行实现，用 AllReduce 同步梯度 |
| **FSDP** | Fully Sharded Data Parallel，PyTorch 官方的 ZeRO-3 实现，将参数/梯度/优化器状态切分到各卡 |
| **ZeRO** | Zero Redundancy Optimizer，通过切分优化器状态(Stage 1)、梯度(Stage 2)、参数(Stage 3) 来节省显存 |
| **Gradient Bucketing** | DDP 将参数梯度分组，每组满了就立即启动 AllReduce，与后续反向传播重叠执行 |
| **$\Phi$** | 本文中表示“参数量 × 每参数字节数”，即模型参数的总字节数 |
