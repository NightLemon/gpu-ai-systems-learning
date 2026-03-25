# Attention 机制

> 从基础的 Scaled Dot-Product Attention 到 GQA（Grouped Query Attention）——理解注意力机制的计算原理和实用变体。Attention 是 Transformer 架构的核心，它让模型在处理每个 token 时能“看到”序列中的其他所有 token。

## 为什么系统工程师要理解 Attention？

你不需要从零设计模型，但你需要能回答这些问题：
- 为什么 Attention 的显存和计算量与序列长度的平方成正比？（这直接影响你能支持多长的上下文）
- 为什么 GQA 能让推理速度提升 2-4 倍？（因为减少了 KV-Cache 大小）
- 为什么 FlashAttention 能同时加速和省显存？（因为避免了 N×N 矩阵写入 HBM）
- Tensor Parallel 怎么切分 Attention 的头？（多头设计天然支持按头切分）

这些问题的答案都藏在 Attention 的计算结构里。理解它不是为了做算法研究，而是为了让你在做系统优化时知道“为什么这么做是对的”。

## 核心概念

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```
输入: Q (query), K (key), V (value), 维度均为 (batch, seq_len, d_model)

每个 token 的 Q 和所有 token 的 K 计算相似度 → softmax → 加权 V

步骤:
  Q × K^T → [B, N, N] 的 attention score 矩阵     ← O(N²d) 计算
  / sqrt(d_k) → 缩放                                ← 防止内积过大导致 softmax 梯度消失
  softmax → attention weight                         ← O(N²) 计算
  × V → [B, N, d] 输出                             ← O(N²d) 计算
```

### Multi-Head Attention (MHA)

```python
# 将 d_model 分成 h 个 head，各自独立做 attention 再拼接
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4096, n_heads=32):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 128
        self.W_q = nn.Linear(d_model, d_model)  # 投影到 Q
        self.W_k = nn.Linear(d_model, d_model)  # 投影到 K
        self.W_v = nn.Linear(d_model, d_model)  # 投影到 V
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
    
    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Q, K, V: [B, n_heads, N, d_head]
        
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = attn @ V  # [B, n_heads, N, d_head]
        
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)
```

### MHA → MQA → GQA

```
MHA:  32 Q heads, 32 K heads, 32 V heads  → 每个 Q head 有独立的 KV
MQA:  32 Q heads, 1 K head,  1 V head    → 所有 Q head 共享 1 组 KV
GQA:  32 Q heads, 8 K heads, 8 V heads   → 每 4 个 Q head 共享 1 组 KV

KV-Cache 大小:  MHA >> GQA >> MQA
模型质量:       MHA ≥ GQA >> MQA  (GQA 是甜蜜点)
```

### 位置编码

Transformer 本身对位置不敏感，需要注入位置信息：

| 方式 | 代表模型 | 特点 |
|------|---------|------|
| Sinusoidal | 原始 Transformer | 固定，不可学习 |
| Learned | GPT-2 | 可学习，但长度固定 |
| **RoPE** | LLaMA, Mistral | 旋转位置编码，可外推 |
| ALiBi | BLOOM | 加在 attention score 上的线性 bias |

RoPE（Rotary Position Embedding）是当前最流行的选择——通过对 Q、K 做旋转变换注入相对位置信息，天然支持长度外推。

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [GQA 论文](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [RoPE 详解](https://arxiv.org/abs/2104.09864) — Su et al., 2021

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Attention（注意力）** | 一种让模型动态地“关注”输入序列中不同位置的机制。通过 Q、K 的相似度计算权重，再对 V 加权求和 |
| **Q / K / V** | Query、Key、Value。输入经过三个线性投影得到，Q 和 K 用于计算相似度，V 是被加权求和的内容 |
| **MHA（Multi-Head Attention）** | 多头注意力。将 Q/K/V 分成多个“头”，各自独立计算 Attention 后拼接，让模型能同时关注不同类型的关系 |
| **GQA（Grouped Query Attention）** | 分组查询注意力。多个 Q 头共享一组 K/V 头，在质量和 KV-Cache 大小之间取得平衡。LLaMA 2/3、Mistral 等均采用 |
| **MQA（Multi-Query Attention）** | 多查询注意力。所有 Q 头共享同一组 K/V，KV-Cache 最小但质量稍差 |
| **RoPE** | Rotary Position Embedding，旋转位置编码。通过对 Q/K 向量做旋转变换注入位置信息，当前最主流的位置编码方式 |
| **ALiBi** | Attention with Linear Biases，在 Attention Score 上加一个与距离成正比的偏置来编码位置 |
| **Softmax** | 将任意实数向量归一化为概率分布（所有元素非负且和为 1）的函数，在 Attention 中用于计算权重 |
