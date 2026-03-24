# 数据加载优化

> 数据加载是训练流水线中容易被忽视的瓶颈。

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
