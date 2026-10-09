# 04 - Transformer 深入

> 你不需要会从零设计模型，但必须看得懂 Transformer 的**计算结构和显存账**——否则后面所有训练并行策略、KV-Cache 优化、注意力 kernel 的设计动机都讲不通。本章按“系统工程师”视角组织：少推公式、多算 FLOPs/显存、对比主流模型的架构差异（MHA/GQA/MLA、SwiGLU、RoPE）以及它们对推理服务的影响。

## 本章内容

| 文件 | 主题 | 要点 |
|------|------|------|
| [01-attention-mechanism.md](01-attention-mechanism.md) | Attention 机制 | Self-Attention、MHA/GQA/MQA/MLA、位置编码、O(n²) 由来 |
| [02-computation-analysis.md](02-computation-analysis.md) | 计算与显存分析 | 6N 公式、激活值显存、训练 16P 显存账 |
| [03-common-architectures.md](03-common-architectures.md) | 常见架构 | GPT-3 / LLaMA / Mistral / DeepSeek-V2 的关键设计差异 |

## 前置知识

- 知道神经网络、矩阵乘、softmax 的基本含义
- 用 PyTorch 跑过简单模型；能看懂 `nn.Linear` / `nn.LayerNorm`
- **不需要**懂 backprop 推导细节，会看 FLOPs 计数即可
- 推荐先读 [Ch02 GPU 架构](../02-gpu-architecture/README.md)，本章会引用 H100 算力/HBM 带宽做估算

## 学完本章你能回答

1. 为什么 Attention 的显存和计算量都是 O(n²d)？这个平方项具体出在哪一步？
2. MHA / GQA / MQA / MLA 各自的 KV-Cache 大小是多少？为什么 LLaMA-3 全系列都用 GQA？
3. 怎么用"6N × tokens"公式 30 秒估出一个训练任务的总 FLOPs？
4. 训练一个 7B 模型，混合精度 AdamW 下显存账（参数 + 梯度 + 优化器 + 激活值）怎么算到 ~112 GB？
5. 为什么 RoPE 比 Learned Position Embedding 更适合长上下文外推？

## 与前后章的关系

- **承接 Ch02**：本章会用 H100 的 dense FP16/BF16 ≈ 989 TFLOPS、HBM 3.35 TB/s 做 roofline 与显存估算
- **铺垫 Ch05**：训练显存账（16P bytes）是 Ch05 混合精度、checkpointing 的出发点
- **铺垫 Ch06**：FLOPs 估算和模型结构是 Ch06 分布式并行策略选型的依据
- **铺垫 Ch07**：KV-Cache 大小、attention O(n²) 是 Ch07 推理优化所有技巧的动机
