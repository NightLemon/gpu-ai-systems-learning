# 混合精度训练

> 用低精度浮点数（FP16 或 BF16）做前向/反向传播，用 FP32 保存主权重做参数更新——兼顾训练速度和数值稳定性。这是当前几乎所有大模型训练的标配。

## 为什么不能全用 FP32 训练？

回想 Ch02 中的 GPU 架构：H100 的 FP32 算力大约是 67 TFLOPS，而 BF16 Tensor Core 峰值远高于此。按 dense 口径常见值约 **989 TFLOPS**，按带结构化稀疏的宣传口径则常写成 **1979 TFLOPS**。无论采用哪种口径，低精度 Tensor Core 吞吐都远高于 FP32，所以如果你全程用 FP32 训练，就无法充分利用现代训练卡的主要算力。

但直接用 FP16 又有问题：FP16 的数值范围很小（最大值只有 65504），训练中的梯度可能非常小，小到 FP16 表示不了（underflow 变成 0），导致训练不收敛或 loss 突然变成 NaN。

**混合精度**是折中方案：前向/反向传播用低精度（快），参数更新用 FP32（稳）。而 BF16 的出现更是解决了 FP16 范围太小的问题——BF16 和 FP32 数值范围相同，不需要 Loss Scaling，成为了大模型训练的首选。

## 核心概念

### 数据格式对比

```
FP32:  1 sign + 8 exponent + 23 mantissa  → 范围大、精度高、慢
FP16:  1 sign + 5 exponent + 10 mantissa  → 范围小、容易溢出、快
BF16:  1 sign + 8 exponent + 7 mantissa   → 范围和FP32一样、精度低、快
TF32:  1 sign + 8 exponent + 10 mantissa  → Ampere+ Tensor Core 可加速部分 FP32 matmul
FP8:   E4M3 (4+3) 或 E5M2 (5+2)          → Hopper+ 支持，需结合量化策略和精度回归使用
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
import torch
from torch import amp

scaler = amp.GradScaler("cuda")  # FP16 需要，BF16 通常不需要

for batch in dataloader:
     optimizer.zero_grad()

     with amp.autocast("cuda", dtype=torch.bfloat16):  # 或 torch.float16
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

A: 需要分版本和后端来看。旧接口里，`torch.backends.cuda.matmul.allow_tf32` 控制 matmul，`torch.backends.cudnn.allow_tf32` 控制 cuDNN；自 PyTorch 1.12 起，matmul 的旧默认值已经不是简单的“默认开启”。当前文档更推荐使用新的 precision 设置接口做更细粒度控制。实践上，把它理解为“FP32 工作负载是否允许内部用 TF32 加速”会更准确。

## 延伸阅读

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., 2018
- [PyTorch AMP Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **混合精度（Mixed Precision）** | 前向/反向传播用低精度（FP16/BF16）加速，参数更新用 FP32 保证数值稳定性 |
| **FP32 / FP16 / BF16** | 32/16/16 位浮点格式。BF16 和 FP32 数值范围相同（指数位相同）但精度较低，是大模型训练的首选 |
| **TF32** | TensorFloat-32，NVIDIA 定义的格式，8 位指数 + 10 位尾数。A100+ 的 Tensor Core 可用它加速部分 FP32 矩阵运算；具体默认行为取决于 PyTorch 版本和后端设置 |
| **FP8** | 8 位浮点格式（E4M3 或 E5M2），Hopper+ 原生支持；吞吐潜力高，但需要校准、scale 管理和任务级精度回归 |
| **Loss Scaling** | 将 loss 乘以一个缩放因子后再反向传播，防止 FP16 梯度因太小而变成零（underflow）。BF16 不需要 |
| **AMP** | Automatic Mixed Precision，PyTorch 提供的自动混合精度训练接口。新代码更推荐使用 `torch.amp` |
| **GradScaler** | PyTorch AMP 中的动态 Loss Scaling 管理器，自动调整缩放因子 |
