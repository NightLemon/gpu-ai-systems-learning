# 常见 LLM 架构

> GPT、LLaMA、Mistral、DeepSeek——了解当前主流大语言模型的关键设计差异。这些模型都是基于 Transformer 的 Decoder-only 架构（只用解码器，自回归生成），但在 Attention 变体、FFN 结构、位置编码等细节上有重要差异。

## 架构对比

| 特性 | GPT-3 | LLaMA 2 | LLaMA 3 | Mistral 7B | DeepSeek-V2 |
|------|-------|---------|---------|------------|-------------|
| 参数量 | 175B | 7B/70B | 8B/70B/405B | 7B | 236B (21B active) |
| 层数 | 96 | 32/80 | 32/80/126 | 32 | 60 |
| Hidden | 12288 | 4096/8192 | 4096/8192/16384 | 4096 | 5120 |
| Heads | 96 | 32/64 | 32/64/128 | 32 | 128 |
| KV Heads | 96 (MHA) | 32/8 (GQA) | 8/8/8 (GQA) | 8 (GQA) | MLA |
| FFN | GeLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU+MoE |
| Norm | LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| Pos Enc | Learned | RoPE | RoPE | RoPE | RoPE |
| Context | 2K | 4K | 128K | 32K (sliding) | 128K |

### 关键设计演进

```
GPT-3 → LLaMA 的变化:
  ✅ Post-Norm → Pre-Norm (RMSNorm): 训练更稳定
  ✅ GeLU → SwiGLU: 效果更好
  ✅ Learned PE → RoPE: 支持长度外推
  ✅ MHA → GQA: 推理更快（减少 KV-Cache）
  
LLaMA 2 → Mistral 的创新:
  ✅ Sliding Window Attention: 更长上下文、更低推理成本
  ✅ Rolling KV-Cache: KV-Cache 不再无限增长
  
Dense → MoE (Mixtral/DeepSeek):
  ✅ 相同计算量下更强的能力
  ✅ Expert Parallel 新挑战

DeepSeek-V2 的创新:
  ✅ MLA (Multi-head Latent Attention): KV-Cache 压缩到极致
```

### SwiGLU FFN

```python
# 标准 FFN: Y = GeLU(XW1) × W2
# SwiGLU:   Y = (SiLU(XW_gate) ⊙ XW_up) × W_down

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

SwiGLU 比标准 FFN 多一个 gate projection（参数量增加 ~50%），但在相同参数量下效果更好（所以 LLaMA 的 intermediate_size 从 4H 降到了 ~2.7H）。

### RMSNorm

```python
# LayerNorm: (x - mean) / std × γ + β  (有 mean shift)
# RMSNorm:   x / RMS(x) × γ            (无 mean shift，更快)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

## 延伸阅读

- [LLaMA 论文](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [LLaMA 2](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — DeepSeek, 2024

---

## 术语表

| 术语 | 说明 |
|------|------|
| **Decoder-only** | Transformer 的一种变体，只用解码器（带 causal mask），每个 token 只能看到它前面的 token。GPT/LLaMA/Mistral 都是此类 |
| **Pre-Norm vs Post-Norm** | LayerNorm 放在 Attention/FFN 之前（Pre-Norm）还是之后（Post-Norm）。Pre-Norm 训练更稳定，是当前主流 |
| **SwiGLU** | Swish-Gated Linear Unit，一种 FFN 激活函数，用 SiLU 门控机制替代 GeLU，效果更好 |
| **RMSNorm** | Root Mean Square Normalization，省略均值计算的简化版 LayerNorm，更快且效果相当 |
| **Sliding Window Attention** | Mistral 引入，每个 token 只关注前面固定窗口内的 token，降低长序列的计算和显存开销 |
| **MLA** | Multi-head Latent Attention，DeepSeek-V2 提出，将 KV 投影到低秩空间压缩 KV-Cache |
| **MoE** | Mixture of Experts，混合专家模型。每个 token 只激活部分 FFN（专家），总参数大但计算量可控 |
