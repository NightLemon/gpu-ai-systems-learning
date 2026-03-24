# 性能分析（Profiling）

> 优化之前必须先 Profile——找到真正的瓶颈。

## 核心工具

### 1. PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5: break
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()

# 打印 top 算子
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# TensorBoard 查看
# tensorboard --logdir=./log/profiler
```

### 2. Nsight Systems（系统级时间线）

```bash
nsys profile -o trace \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    python train.py

# 用 Nsight Systems GUI 打开 trace.nsys-rep
# 能看到: kernel 执行、内存拷贝、CPU/GPU overlap、NCCL 通信
```

### 3. Nsight Compute（单 kernel 深度分析）

```bash
ncu --set full -o kernel_analysis python train.py
# 分析: roofline 定位、stall 原因、occupancy、内存吞吐
```

### 4. 快速诊断

```python
# GPU 利用率
import torch
torch.cuda.utilization()  # 0-100%

# 显存
torch.cuda.memory_allocated() / 1e9  # 已用 (GB)
torch.cuda.memory_reserved() / 1e9   # 已分配 (GB)
torch.cuda.max_memory_allocated() / 1e9  # 峰值

# nvidia-smi 持续监控
# nvidia-smi dmon -s u -d 1  (每秒采样 GPU 利用率)
```

### Profile → 优化的常见模式

| Profile 发现 | 瓶颈 | 优化方向 |
|-------------|------|---------|
| GPU utilization 低 | 数据加载/CPU | 增加 num_workers, pin_memory |
| 大量小 kernel launch | Kernel launch 开销 | torch.compile, 算子融合 |
| NCCL 占比高 | 通信 | 减小 TP degree, overlap 通信 |
| 显存不足频繁 OOM | 显存 | Gradient checkpoint, 减 batch |
| Attention 耗时长 | Attention | FlashAttention |

## 延伸阅读

- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
