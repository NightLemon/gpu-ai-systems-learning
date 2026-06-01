# KV-Cache

> KV-Cache 是大语言模型自回归生成（逐 token 输出）时的核心缓存机制。它缓存已生成 token 的 Key 和 Value 向量，避免每一步都重新计算。不理解 KV-Cache 就无法理解 LLM 推理的显存和性能特性。

## 为什么 KV-Cache 是推理优化的第一课？

当你用 ChatGPT 获得一段 100 个 token 的回复时，模型实际上执行了 100 次 前向传播（每次只生成 1 个 token）。如果没有 KV-Cache，每次生成都要对**所有已有 token** 重新计算 Attention——第 100 次生成时要对 99 个历史 token 重新算一遍 K 和 V，计算量为 O(N²)，完全不可接受。

KV-Cache 的思路很简单：既然历史 token 的 K、V 不会变，**算一次就缓存下来，后续生成时直接复用**。这把每步的计算量从 O(N²) 降到 O(N)。

但这带来了新问题：**KV-Cache 要占显存**。对于一个 7B 模型，单个请求的 KV-Cache 可能占 512 MB（GQA）到 2 GB（MHA）。并发 100 个请求，KV-Cache 就可能超过模型本身的大小。推理优化的很大一部分工作就是在管理这块显存（PagedAttention、KV-Cache 量化、GQA 等）。

## 核心概念

### 为什么需要 KV-Cache？

自回归生成每次只生成一个 token，但 Attention 需要看到所有之前的 token：

```
没有 KV-Cache:
  Step 1: Attention("The")                          → 计算 K,V for "The"
  Step 2: Attention("The", "capital")                → 重新计算 K,V for "The", "capital"
  Step 3: Attention("The", "capital", "of")          → 重新计算 K,V for 所有 3 个 token
  ...
  Step N: Attention(所有 N 个 token)                  → 计算量 ∝ N²

有 KV-Cache:
  Step 1: 计算 K₁,V₁ → 缓存 → Attention
  Step 2: 计算 K₂,V₂ → 缓存追加 → Attention(Q₂, [K₁,K₂], [V₁,V₂])
  Step 3: 计算 K₃,V₃ → 缓存追加 → Attention(Q₃, [K₁,K₂,K₃], [V₁,V₂,V₃])
  ...
  → 每步只需计算新 token 的 Q,K,V → 计算量从 O(N²) 降到 O(N)
```

### KV-Cache 显存占用

$$\text{KV-Cache size} = 2 \times L \times H \times S \times D \times B$$

其中：
- $2$ = K 和 V 各一份
- $L$ = 层数
- $H$ = KV Head 数（GQA 时小于 Q Head 数）
- $S$ = 序列长度
- $D$ = Head Dimension（通常 = hidden\_size / num\_heads）
- $B$ = Batch Size

```
7B MHA-style 模型 (FP16):
  L=32, H=32 (MHA), D=128, S=4096, B=1
  KV-Cache = 2 × 32 × 32 × 4096 × 128 × 2 bytes = 2 GB per request!

7B GQA-style 模型 (FP16, H_kv=8):
  KV-Cache = 2 × 32 × 8 × 4096 × 128 × 2 bytes = 512 MB per request

Llama 3 70B 量级模型 (FP16, GQA H_kv=8):
  L=80, D=128, S=8192
  KV-Cache = 2 × 80 × 8 × 8192 × 128 × 2 bytes = 2.56 GB per request
  
→ 并发 100 个请求 → KV-Cache 就要 256 GB!
```

### GQA (Grouped Query Attention) 对 KV-Cache 的影响

```
MHA (Multi-Head Attention):     每个 Q head 有自己的 K,V head
  Q heads = 32, KV heads = 32
  KV-Cache ∝ 32

MQA (Multi-Query Attention):    所有 Q head 共享一组 K,V
  Q heads = 32, KV heads = 1
  KV-Cache ∝ 1 (减少 32x!)

GQA (Grouped Query Attention):  Q head 分组，每组共享一组 K,V
  Q heads = 32, KV heads = 8 (每 4 个 Q head 共享)
  KV-Cache ∝ 8 (减少 4x)

→ GQA 是目前的主流选择（Llama 2 70B、Llama 3、Mistral 等模型家族都采用或部分采用类似思路）
→ 在 KV-Cache 节省和模型质量之间取得平衡
```

## 关键细节

### KV-Cache 的内存碎片问题

```
问题：不同请求的序列长度不同，KV-Cache 的大小也不同

静态分配（朴素方式）:
  为每个请求预分配 max_seq_len 的 KV-Cache
  → 实际用 500 tokens，但预留了 4096 → 浪费 ~88%!

  Request A (实际 500 tokens):  [████████████░░░░░░░░░░░░░░░░░░░░░] 4096 预留
  Request B (实际 200 tokens):  [█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 4096 预留
  
  → 显存利用率极低，能同时服务的请求数大幅减少

→ vLLM 的 PagedAttention 解决方案见 06-vllm-architecture.md
```

### KV-Cache 量化

可以对 KV-Cache 做量化以进一步减少显存：

```
FP16 KV-Cache:  baseline
INT8 KV-Cache:  显存减半，精度影响很小
FP8 KV-Cache:   显存减半，Hopper 原生支持
INT4 KV-Cache:  显存 1/4，精度有一定影响

实现: KV-Cache 在写入时量化，读出时反量化
关键: per-token 或 per-channel 的 scale factor
```

### Prefill 和 Decode 的 KV-Cache 行为

```
Prefill 阶段:
  输入: 完整 prompt (N tokens)
  计算: 所有 N 个 token 的 K,V
  一次性写入 KV-Cache: N × 2 × L × H × D

Decode 阶段:
  每步: 1 个新 token
  计算: 1 个 token 的 K,V
  追加到 KV-Cache: 1 × 2 × L × H × D
  Attention 读取: 整个 KV-Cache (growing)
```

## 代码示例

```python
# 简化的 KV-Cache 实现
class KVCache:
    def __init__(self, max_batch, max_seq_len, num_layers, num_kv_heads, head_dim, dtype=torch.float16):
        self.k_cache = torch.zeros(
            num_layers, max_batch, num_kv_heads, max_seq_len, head_dim, 
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_lens = torch.zeros(max_batch, dtype=torch.int32)
    
    def update(self, layer_idx, batch_idx, k, v):
        """追加新的 K, V 到 cache"""
        pos = self.seq_lens[batch_idx]
        self.k_cache[layer_idx, batch_idx, :, pos] = k
        self.v_cache[layer_idx, batch_idx, :, pos] = v
        self.seq_lens[batch_idx] += 1
    
    def get(self, layer_idx, batch_idx):
        """获取完整的 K, V cache"""
        seq_len = self.seq_lens[batch_idx]
        k = self.k_cache[layer_idx, batch_idx, :, :seq_len]
        v = self.v_cache[layer_idx, batch_idx, :, :seq_len]
        return k, v
```

## 常见问题

**Q: KV-Cache 和模型权重哪个更耗显存？**

A: 取决于 batch size 和序列长度。对于长上下文 + 大 batch 的场景，KV-Cache 可能远超模型权重。例如 7B 模型权重 14GB，但 batch=64、seq=4096 的 KV-Cache 可达 128GB。

**Q: 为什么 KV-Cache 的读取是推理的瓶颈？**

A: Decode 阶段每生成一个 token，需要读取整个 KV-Cache 做 Attention。随着序列变长，KV-Cache 越来越大，读取量线性增长 → memory-bound 加剧。

**Q: MLA (Multi-head Latent Attention) 和 KV-Cache 有什么关系？**

A: DeepSeek-V2 提出的 MLA 通过将 KV 投影到低秩空间来压缩 KV-Cache。效果类似 GQA 但更激进——KV-Cache 可以压缩到 MHA 的 ~5-10%，同时保持接近 MHA 的质量。

## 延伸阅读

- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180) — vLLM 论文
- [DeepSeek-V2: MLA](https://arxiv.org/abs/2405.04434) — DeepSeek 2024

---

## 术语表

| 术语 | 说明 |
|------|------|
| **KV-Cache** | 缓存已生成 token 的 Key 和 Value 向量，让后续 token 不需重新计算历史 KV，将 Attention 复杂度从 O(N²) 降到 O(N) |
| **自回归生成（Autoregressive Generation）** | 每次只生成一个 token，将它追加到输入中再生成下一个，重复直到完成 |
| **Prefill 阶段** | 处理完整 prompt 的阶段，并行计算所有 token 的 KV。Compute-bound |
| **Decode 阶段** | 逐 token 生成阶段，每次只产出 1 个 token。需读取全量模型权重，Memory-bound |
| **GQA** | Grouped Query Attention，多个 Q 头共享一组 KV 头，减少 KV-Cache 大小 |
| **MLA** | Multi-head Latent Attention，DeepSeek-V2 提出的 KV 压缩方案 |
