# Attention 机制

> 从 Scaled Dot-Product 到 GQA——理解 Attention 的计算和变体。

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
