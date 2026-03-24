# 存储系统

> 大规模训练的数据读取和 Checkpoint 保存挑战。

## 核心概念

### 存储需求分析

```
训练数据: TB-PB 级，需要高吞吐顺序读
Checkpoint: 模型参数 + 优化器状态，每隔 N 步保存
  - 7B 模型: ~56 GB per checkpoint (FP16 params + FP32 optimizer)
  - 175B 模型: ~2 TB per checkpoint
  - 保存间隔: 每 1000-5000 步 → 每天多次
  → 需要高带宽突发写入能力
```

### 分布式文件系统

| 系统 | 特点 | 典型用户 |
|------|------|---------|
| **Lustre** | 高吞吐并行文件系统，HPC 标配 | 国家级超算 |
| **GPFS (Spectrum Scale)** | IBM，企业级 | AI 研究实验室 |
| **BeeGFS** | 开源并行文件系统 | 中小规模集群 |
| **对象存储 (S3/GCS)** | 无限容量，延迟高 | 数据集存储 |
| **NVMe over Fabrics** | 低延迟网络 SSD | Checkpoint 加速 |

### Checkpoint 优化

```python
# 大模型 Checkpoint 的挑战:
# 1. 体积大: 175B 模型 → ~2 TB
# 2. 保存时 GPU 空闲（同步写入）
# 3. 恢复时需要读回所有参数

# 异步 Checkpoint (PyTorch 2.0+)
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

# 使用 FSDP 的分片 checkpoint: 每卡只保存自己的 shard
# → 并行写入，速度约 N 倍
save({"model": model.state_dict()}, checkpoint_id=f"step_{step}")
```

## 延伸阅读

- [Lustre Documentation](https://www.lustre.org/)
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
