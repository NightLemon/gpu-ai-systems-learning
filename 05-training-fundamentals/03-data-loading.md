# 数据加载优化

> 数据加载是训练流水线中容易被忽视的瓶颈。如果 GPU 在等数据，那再快的卡也发挥不出来。本节讲解如何优化 PyTorch DataLoader、选择合适的数据格式和诊断数据加载瓶颈。

## 一个典型的调优场景

你刚启动了多卡训练，`nvidia-smi` 显示 GPU 利用率只有 40-60%，而且周期性地掉到 0% 再回升。这通常意味着 GPU 在等 CPU 喂数据。

检查点清单：
1. **`num_workers` 是不是太小？** 默认是 0（主进程加载），应该设为 4-8 per GPU
2. **`pin_memory` 开了吗？** 开启后 CPU→GPU 传输速度提升 2-3x
3. **数据是不是在训练时做 tokenize？** 应该预处理为二进制格式，训练时只做加载
4. **数据在网络文件系统 (NFS) 上？** 考虑缓存到本地 NVMe SSD

解决了这些，GPU 利用率可能立刻从 50% 跳到 90%+。

## 核心概念

### DataLoader 关键参数

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,         # CPU 预处理进程数
    pin_memory=True,       # 使用 pinned memory 加速 CPU→GPU 传输
    prefetch_factor=2,     # 每个 worker 预取的 batch 数
    persistent_workers=True, # 避免每 epoch 重建 worker 进程
)
```

**`num_workers` 调优**：太少 → GPU 等数据；太多 → CPU/内存压力大。经验起点：4-8 per GPU。

**`pin_memory=True`**：将数据放入页锁定内存 → CPU→GPU 传输走 DMA，速度提升 2-3x。

### 大规模数据加载

```
问题: 大规模训练的数据集可能有 TB 级别
  - 不能全部放内存
  - 分布式训练中 N 张卡同时读不同数据

方案:
  1. WebDataset / Mosaic StreamingDataset: 流式读取，不需要全部加载
  2. 预处理为 tokenized 格式（避免训练时 tokenize）
  3. 数据并行: DistributedSampler 确保各卡数据不重复
```

```python
# Streaming Dataset (Mosaic ML)
from streaming import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset(
    remote='s3://my-bucket/training-data',
    local='/tmp/cache',
    shuffle=True,
    batch_size=32,
)

dataloader = StreamingDataLoader(dataset, batch_size=32)
```

### 诊断数据加载瓶颈

```python
# 用 PyTorch Profiler 看数据加载时间
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]
) as prof:
    for i, batch in enumerate(dataloader):
        if i > 10: break
        # training step

# 如果 CPU 活动中 DataLoader 占比高 → 数据加载是瓶颈
```

## 延伸阅读

- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
- [Mosaic StreamingDataset](https://github.com/mosaicml/streaming)
- [WebDataset](https://github.com/webdataset/webdataset)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **DataLoader** | PyTorch 中负责从数据集读取数据并组装成 batch 的组件 |
| **num_workers** | DataLoader 使用的 CPU 子进程数，负责并行读取和预处理数据 |
| **pin_memory** | 将数据放入页锁定内存，CPU→GPU 传输时可以用 DMA 直接拷贝，速度更快 |
| **prefetch_factor** | 每个 worker 预先加载的 batch 数，减少 GPU 等待数据的空闲时间 |
| **DistributedSampler** | 多卡训练时的数据采样器，确保每张卡拿到不同的数据子集 |
| **StreamingDataset** | 流式数据加载，不需要将全部数据下载到本地，边读边训，适合 TB 级数据集 |
