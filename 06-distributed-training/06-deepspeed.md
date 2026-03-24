# DeepSpeed

> 微软的深度学习优化库，以 ZeRO 系列为核心，覆盖训练和推理全链路。

## 核心概念

### DeepSpeed 是什么

DeepSpeed 提供一套完整的大模型训练/推理优化方案，核心特性：

```
训练优化:
  ├── ZeRO Stage 1/2/3    显存优化（核心）
  ├── ZeRO-Offload        参数/优化器卸载到 CPU/NVMe
  ├── ZeRO-Infinity       支持万亿参数模型
  ├── Pipeline Parallelism 流水线并行
  ├── MoE Support          专家并行
  ├── Sparse Attention     稀疏注意力
  └── Curriculum Learning  课程学习

推理优化:
  ├── DeepSpeed-Inference  Tensor 并行推理
  ├── ZeRO-Inference       多卡推理
  └── DeepSpeed-MII        推理部署框架
```

### 使用方式：配置驱动

DeepSpeed 的核心设计哲学是**配置化**——大部分优化通过 JSON 配置开启：

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    },
    
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": true
}
```

### 快速集成

```python
import deepspeed

# 用 DeepSpeed 包装模型
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)

# 训练循环（和普通 PyTorch 几乎一样）
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)  # 替代 loss.backward()
    model.step()          # 替代 optimizer.step()
```

## 关键细节

### ZeRO Stage 对比

```
          Model  Gradient  Optimizer    通信      使用场景
          Params  Storage   State      开销
────────────────────────────────────────────────────────
ZeRO-1    ✗       ✗        ✓ 分片     同 DDP    大多数场景的起点
ZeRO-2    ✗       ✓ 分片   ✓ 分片     同 DDP    模型中等大小
ZeRO-3    ✓ 分片  ✓ 分片   ✓ 分片     +50%      模型极大

✗ = 每卡全量副本
✓ 分片 = 切分到多卡
```

### ZeRO-Offload：利用 CPU 内存

```
GPU 显存不够？把不常用的数据放到 CPU：

ZeRO-Offload (Stage 2):
  GPU: 参数 (FP16) + 梯度计算
  CPU: 优化器状态 (FP32) + 参数更新
  
  流程: GPU 计算梯度 → 传到 CPU → CPU 做 Adam 更新 → 传回 GPU

ZeRO-Infinity (Stage 3):
  GPU: 当前层的参数 + 计算
  CPU: 所有参数、梯度、优化器
  NVMe SSD: 溢出到磁盘
  
  → 理论上可以训练任意大的模型（只要有足够的 CPU/NVMe）
```

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        }
    }
}
```

### 关键配置参数解析

```json
{
    "zero_optimization": {
        "stage": 3,
        
        // AllGather 的 bucket size (bytes)
        // 越大 → 通信更高效，但显存开销更大
        "allgather_bucket_size": 5e8,  // 500 MB
        
        // ReduceScatter 的 bucket size
        "reduce_bucket_size": 5e8,
        
        // 是否重叠通信和计算
        "overlap_comm": true,
        
        // Stage 3 特有：预取参数
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        // 参数 < threshold → 不分片（小参数通信开销大于收益）
        
        // Stage 3: 最大可复用缓冲区
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}
```

### HuggingFace Transformers 集成

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=True,
    
    # DeepSpeed 集成
    deepspeed="ds_config.json",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

## 常见问题

**Q: DeepSpeed 和 PyTorch FSDP 该怎么选？**

| 考量 | DeepSpeed | FSDP |
|------|-----------|------|
| 易用性 | JSON 配置，少改代码 | 需要代码修改（wrap policy） |
| CPU/NVMe Offload | ✅ 成熟 | 有限 |
| torch.compile 支持 | ❌ 有限 | ✅ 原生支持 |
| MoE 支持 | ✅ | 需要额外工作 |
| 社区生态 | HuggingFace 原生支持 | PyTorch 原生 |
| 维护 | 微软 | Meta/PyTorch |

**Q: ZeRO Stage 选几？**

经验法则：
- **Stage 1**：首选，显存够就用它（开销最小）
- **Stage 2**：Stage 1 OOM 时升级
- **Stage 3**：模型很大（如 70B），Stage 2 仍 OOM 时用
- **Stage 3 + Offload**：单机 GPU 显存严重不足时的最后手段

**Q: DeepSpeed 的 `gradient_accumulation_steps` 怎么和 micro batch 配合？**

```
train_batch_size = micro_batch_per_gpu × num_gpus × gradient_accumulation_steps

例: micro_batch=4, 8 GPUs, grad_accum=4
→ effective batch size = 4 × 8 × 4 = 128
```

## 延伸阅读

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO 论文](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2020
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840) — Ren et al., 2021
- [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) — Rajbhandari et al., 2021
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
