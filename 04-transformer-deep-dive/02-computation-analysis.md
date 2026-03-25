# 计算与显存分析

> 能精确计算一个模型的 FLOPs（浮点运算量）和显存占用——这是 GPU 系统优化工程师的基本功。知道计算量和显存占用后，你才能判断一个模型需要多少张卡、用什么并行策略。

## 这一节解决什么问题？

当你拿到一个新模型时，第一个问题往往是：“我需要多少张卡才能装下/训起来？”要回答这个问题，你需要能快速估算：

- **显存占用**：模型参数 + 梯度 + 优化器状态 + 激活值，加起来需要多少 GB？
- **计算量**：训练一个 token 需要多少 FLOPs？一张卡的算力够不够？
- **并行策略**：单层参数超过单卡显存 → 必须用 TP；总参数的优化器状态超过单卡 → 要 ZeRO/FSDP

本节给你这些估算的公式和直觉。

## 核心概念

### Transformer 层结构

```
一个 Decoder-only Transformer 层:

Input (B, S, H)
  │
  ├─ LayerNorm
  ├─ Self-Attention: Q,K,V Projection → Attention → Output Projection
  ├─ Residual Add
  ├─ LayerNorm
  ├─ FFN: Up Projection → Activation → Down Projection
  └─ Residual Add
Output (B, S, H)
```

### FLOPs 计算

对于一个线性层 $Y = XW$，$X \in R^{m \times k}$，$W \in R^{k \times n}$：

$$\text{FLOPs} = 2mkn$$

（每个输出元素需要 k 次乘法和 k-1 次加法 ≈ 2k FLOPs）

```
一个 Transformer 层的 FLOPs (per token):

Self-Attention:
  QKV Projection:  3 × 2 × H × H = 6H²         (3个 H→H 的线性变换)
  Attention Score:  2 × S × H                     (Q @ K^T, 序列长度相关)
  Attention × V:    2 × S × H                     (Score @ V)
  Output Proj:      2 × H × H = 2H²

FFN (SwiGLU, 如 LLaMA):
  Up + Gate:        2 × 2 × H × (8H/3) ≈ 10.67H² (两个 up projection)
  Down:             2 × (8H/3) × H ≈ 5.33H²

每层总 FLOPs ≈ 24H² + 4SH  (per token)
(SwiGLU FFN, intermediate_size ≈ 8H/3)
```

### 完整模型的 FLOPs

$$\text{FLOPs per token} \approx 2P$$

其中 P = 模型参数量。这是一个著名的近似公式。

```
训练（forward + backward）每个 token:

Forward:  ~2P FLOPs
Backward: ~4P FLOPs (梯度计算约 2x forward)
总计:     ~6P FLOPs per token

例: LLaMA 7B, 训练 1T tokens
  总 FLOPs = 6 × 7B × 1T = 42 ZFLOPs
  H100 BF16: 1979 TFLOPS, MFU=40%
  所需 GPU 时间 = 42E21 / (1979E12 × 0.4) = 53E6 秒 = ~614 GPU·天
  1024 张 H100: ~0.6 天 (理想情况)
```

### 显存占用分析

```
训练显存 (混合精度, AdamW):

1. 模型参数 (FP16):     2P bytes
2. 梯度 (FP16):          2P bytes
3. 优化器状态 (FP32):
   - FP32 参数副本:      4P bytes
   - Momentum:           4P bytes
   - Variance:           4P bytes
4. 激活值:               取决于 batch, seq_len, checkpoint

模型状态总计 = 16P bytes

LLaMA 7B: 16 × 7B = 112 GB (不含激活值)
```

**激活值显存**（每层）：

```
输入: B × S × H × 2 bytes
Attention Score: B × n_heads × S × S × 2 bytes  ← 和 S² 成正比！
FFN 中间结果: B × S × FFN_dim × 2 bytes

LLaMA 7B, B=1, S=4096:
  每层激活值 ≈ 200-400 MB (取决于是否 checkpoint)
  32 层 ≈ 6-13 GB
  
B=32, S=4096:
  32 层 ≈ 200-400 GB → 必须用 Gradient Checkpointing!
```

## 常见问题

**Q: 如何快速估算一个模型能不能装进某张卡？**

A: 推理只需模型参数 + KV-Cache。FP16 推理下 `显存 ≈ 2P + KV-Cache` bytes。训练需要 `~16P + 激活值` bytes。

**Q: 为什么大模型的显存占用比参数量大这么多？**

A: 主要是优化器状态（FP32 参数副本 + 两个动量 = 12P bytes）和激活值。这就是 ZeRO/FSDP 切分优化器状态的动机。

## 延伸阅读

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022

---

## 术语表

| 术语 | 说明 |
|------|------|
| **FLOPs** | Floating Point Operations，浮点运算次数。单次 forward 约 2P FLOPs，forward+backward 约 6P FLOPs（P = 参数量） |
| **激活值（Activations）** | 前向传播的中间结果（每层的输出），反向传播时需要用到，因此训练时需要缓存在显存中 |
| **优化器状态** | Adam/AdamW 优化器维护的额外变量（FP32 参数副本 + momentum + variance），每个参数占 12 字节 |
| **SwiGLU** | 一种 FFN 激活函数，用 SiLU 的门控机制替代标准 GeLU，效果更好但参数量略多 |
| **RMSNorm** | Root Mean Square Layer Normalization，比标准 LayerNorm 更快（省去了均值计算） |
| **Scaling Law** | 模型性能与参数量/数据量/计算量之间的幂律关系，用于预测扩大模型的收益 |
