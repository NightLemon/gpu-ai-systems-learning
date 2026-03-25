# FlashAttention

> 一种 IO-aware（显式考虑显存读写开销）的 Attention 算法。标准 Attention 会将 N×N 的中间矩阵写入显存（HBM），FlashAttention 通过分块（Tiling）将计算保持在片上高速存储（SRAM）中完成，避免大量 HBM 读写，实现 2-4x 加速 + 显存线性下降。

## 为什么 FlashAttention 能同时又快又省？

通常你习惯的“速度”和“显存”是两个独立的维度——加速通常意味着更多资源消耗。但 FlashAttention 同时做到了两者，秘密在于：

标准 Attention 的性能瓶颈不在计算，而在“读写 HBM”。它先算出一个 N×N 的 Attention Score 矩阵（可能有几百 MB），写入 HBM；然后读出来做 softmax；再写回；再读出来乘 V……光是这个矩阵的读写就把 HBM 带宽占满了。

FlashAttention 的关键洞察：**这个 N×N 矩阵完全不需要存在 HBM 里**。通过分块（Tiling）+ 在线 Softmax（Online Softmax），可以一次处理一小块，在片上 SRAM 中完成计算后直接丢弃。HBM 读写量大幅减少 → 既省显存又快。

## 核心概念

### 标准 Attention 的问题

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

```
标准实现 (PyTorch):

S = Q @ K.T / sqrt(d)    # [B, H, N, N]  写到 HBM  ← N² 显存
P = softmax(S)            # [B, H, N, N]  读+写 HBM
P = dropout(P)            # [B, H, N, N]  读+写 HBM
O = P @ V                 # [B, H, N, d]  写到 HBM

HBM 读写量: O(N²)  ← 当 N=4096, 这个矩阵 = 4096² × 4B ≈ 64 MB per head
显存: O(N²)        ← 序列越长越爆

问题: 中间结果 S 和 P 是 N×N 矩阵，必须写到 HBM → memory-bound
```

### FlashAttention 的核心 Trick

**不把 N×N 的中间矩阵写到 HBM！** 用 tiling 把计算分块在 SRAM (shared memory) 中完成。

```
关键挑战: softmax 是全局操作 (需要知道整行的max和sum)
  softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))

解决: Online Softmax (分块递推更新)
  处理第 1 块: m1 = max(block1), l1 = sum(exp(block1 - m1))
  处理第 2 块: m2 = max(m1, max(block2))
              l2 = l1 × exp(m1 - m2) + sum(exp(block2 - m2))
              修正之前块的结果...
  → 每块处理完就丢弃，不需要存 N×N 矩阵
```

```
FlashAttention Tiling 示意:

Q 分成 Tr 块, K/V 分成 Tc 块

for each Q_block (大小 Br × d):     ← 外层循环
  load Q_block 到 SRAM
  for each K_block, V_block:         ← 内层循环
    load K_block, V_block 到 SRAM
    compute S_block = Q_block × K_block^T   ← 在 SRAM 中
    online softmax update                    ← 在 SRAM 中
    partial O += softmax(S_block) × V_block  ← 在 SRAM 中
  write O 到 HBM                      ← 每个 Q block 只写一次
```

### 效果

```
HBM 读写量: O(N² d² / M)  (M = SRAM 大小)
  vs 标准: O(N² + Nd)
  
  实际: FlashAttention 在 A100 上快 2-4x

显存: O(N)  (只存 O, logsumexp, 不存 N×N 矩阵)
  vs 标准: O(N²)
  
  → 可以训练更长的序列！比如从 2K 扩展到 16K+
```

## 关键细节

### FlashAttention v1 vs v2 vs v3

| 版本 | 关键改进 | 加速 |
|------|---------|------|
| **v1** | 基本 tiling + online softmax | 2-4x vs standard |
| **v2** | 优化 warp 分配（减少非 GEMM 操作），backward 优化 | 对 v1 再快 2x |
| **v3** (Hopper) | 利用 Hopper 异步特性（TMA、wgmma），FP8 支持 | 对 v2 再快 1.5-2x |

### 为什么 FlashAttention 快？Roofline 分析

```
标准 Attention:
  主要操作是 N×N 的矩阵乘和 softmax
  中间结果都要过 HBM → memory-bound
  
FlashAttention:
  Tiling 后，大部分操作在 SRAM 中完成
  只有 Q/K/V 的加载和 O 的写回需要 HBM
  → 更接近 compute-bound

从 Roofline 角度:
  标准 Attention 的 arithmetic intensity 低（频繁读写 HBM）
  FlashAttention 提高了 arithmetic intensity → 更好地利用算力
```

### 实际使用

```python
# PyTorch 2.0+ 内置
import torch
import torch.nn.functional as F

# 自动使用 FlashAttention（如果满足条件）
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,      # FlashAttention
    enable_math=False,       # 禁用标准数学实现
    enable_mem_efficient=False  # 禁用 memory-efficient attention
):
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

# 或者直接调用（PyTorch 2.0+ 会自动选择最优实现）
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 手动安装 flash-attn 包（更多特性）
# pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)
```

### FlashAttention 的使用条件

```
必须满足:
  - GPU: Ampere (A100) 及以上 (需要 SM80+)
  - dtype: FP16 或 BF16（不支持 FP32）
  - head_dim: 通常 ≤ 256

不支持/需要特殊处理:
  - 自定义 attention mask（v2 支持部分自定义 mask）
  - 需要 attention scores（FlashAttention 不输出 N×N 矩阵）
  - Cross-attention（支持，但 Q 和 KV 序列长度不同时有限制）
```

## 常见问题

**Q: PyTorch 的 `scaled_dot_product_attention` 和 `flash-attn` 包有什么区别？**

A: PyTorch 内置的 SDPA 可以自动选择 FlashAttention、Memory-Efficient Attention 或普通数学实现。`flash-attn` 包通常更新更快，支持更多特性（如 variable-length sequences、alibi positional encoding 等）。性能上 `flash-attn` 往往略优。

**Q: FlashAttention 对训练和推理都有效吗？**

A: 是的。训练时主要好处是减少显存（可以用更长的序列或更大的 batch）和加速。推理时 prefill 阶段受益最大（需要计算完整的 N×N attention），decode 阶段 N=1 所以优势较小。

**Q: FlashDecoding 是什么？**

A: FlashDecoding 是针对推理 decode 阶段的优化。标准 decode 时 Q 只有 1 个 token，KV-cache 可能很长。FlashDecoding 沿 KV 维度并行（不同的 warp 处理 KV-cache 的不同 chunk），然后用一次 reduce 合并结果，从而在 decode 阶段也能充分利用 GPU 并行度。

## 延伸阅读

- [FlashAttention v1](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [FlashAttention v2](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [FlashAttention v3](https://arxiv.org/abs/2407.08608) — Shah et al., 2024
- [flash-attn GitHub](https://github.com/Dao-AILab/flash-attention)
- [FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

---

## 术语表

| 术语 | 说明 |
|------|------|
| **IO-aware** | 算法设计时显式考虑内存/显存的读写开销，而不仅仅优化计算量 |
| **Tiling（分块）** | 将大矩阵切成小块，在片上 SRAM 中完成计算，避免将中间结果写回 HBM |
| **Online Softmax** | 分块递推计算 Softmax 的方法，不需要先算出完整的 N×N 矩阵 |
| **SRAM** | Static RAM，GPU 片上的高速存储（Shared Memory / L1 Cache），带宽远高于 HBM |
| **FlashDecoding** | 针对 decode 阶段（Q 只有 1 个 token）的优化，沿 KV 维度并行 |
| **SDPA** | Scaled Dot-Product Attention，PyTorch 2.0+ 的 `F.scaled_dot_product_attention()` 接口，自动选择最优实现 |
