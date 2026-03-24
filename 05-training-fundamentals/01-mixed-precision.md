# 混合精度训练

> 用低精度（FP16/BF16）训练，获得接近 FP32 的精度和 2x+ 的速度。

## 核心概念

### 数据格式对比

```
FP32:  1 sign + 8 exponent + 23 mantissa  → 范围大、精度高、慢
FP16:  1 sign + 5 exponent + 10 mantissa  → 范围小、容易溢出、快
BF16:  1 sign + 8 exponent + 7 mantissa   → 范围和FP32一样、精度低、快
TF32:  1 sign + 8 exponent + 10 mantissa  → A100 Tensor Core 专用
FP8:   E4M3 (4+3) 或 E5M2 (5+2)          → H100+ 支持, 最快
```

**BF16 是目前大模型训练的首选**——范围和 FP32 一致（不需要 loss scaling），精度够用。

### 混合精度的工作方式

```
Master Weights (FP32) ─── 保持高精度的权重副本
         │
    Cast to FP16/BF16
         │
    Forward Pass (FP16/BF16) ──→ Tensor Core 加速
         │
    Loss Scaling (FP16 only)
         │
    Backward Pass (FP16/BF16)
         │
    Unscale Gradients
         │
    Update Master Weights (FP32) ─── 用 FP32 精度做更新
```

### PyTorch AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # FP16 需要，BF16 不需要

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):  # 或 torch.float16
        output = model(batch)
        loss = criterion(output)
    
    # FP16: 需要 scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # BF16: 不需要 scaler（范围够大，不会 underflow）
    # loss.backward()
    # optimizer.step()
```

### Loss Scaling（仅 FP16 需要）

FP16 的最小正数 ≈ 6e-8。训练中的梯度可能小于这个值 → 变成 0（underflow）。

```
Loss Scaling:
  1. loss × scale (如 1024) → 放大 loss
  2. backward → 梯度也被放大
  3. 梯度 / scale → 恢复正确的梯度值
  4. 如果发现 inf/nan: 跳过本步更新, 减小 scale
  5. 连续多步无 inf: 增大 scale
```

BF16 的指数位和 FP32 一样，不需要 loss scaling。这是 BF16 成为主流的重要原因。

## 常见问题

**Q: BF16 精度比 FP16 低（7 vs 10 mantissa bits），为什么效果反而好？**

A: 因为训练中数值的动态范围比精度更重要。FP16 最大值只有 65504，很容易溢出。BF16 最大值和 FP32 一样（~3.4e38），几乎不需要担心溢出。

**Q: TF32 怎么开启？**

A: PyTorch 默认开启。`torch.backends.cuda.matmul.allow_tf32 = True`。它只影响 matmul 操作——使用 FP32 输入/输出，但 Tensor Core 内部以 TF32 精度计算（更快，精度足够）。

## 延伸阅读

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., 2018
- [PyTorch AMP Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
