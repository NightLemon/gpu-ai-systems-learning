# 07 - 推理优化

> 从 KV-cache 到 vLLM：让大模型推理又快又省。

## 本章内容

| 文件 | 主题 | 要点 |
|------|------|------|
| [01-kv-cache.md](01-kv-cache.md) | KV-Cache | 自回归生成的缓存机制、显存计算 |
| [02-quantization.md](02-quantization.md) | 量化 | GPTQ、AWQ、FP8、W4A16 |
| [03-continuous-batching.md](03-continuous-batching.md) | Continuous Batching | Static vs Dynamic Batching、Iteration-level 调度 |
| [04-speculative-decoding.md](04-speculative-decoding.md) | 投机解码 | Draft-Verify 范式、加速原理 |
| [05-flash-attention.md](05-flash-attention.md) | FlashAttention | IO-aware 算法、Tiling、硬件感知优化 |
| [06-vllm-architecture.md](06-vllm-architecture.md) | vLLM 架构 | PagedAttention、调度器、Serving 架构 |
| [07-tensorrt-llm.md](07-tensorrt-llm.md) | TensorRT-LLM | 图优化、Kernel 融合、FP8 |
| [08-inference-framework-guide.md](08-inference-framework-guide.md) | **框架选型** | vLLM vs TRT-LLM vs SGLang 决策指南 |

## LLM 推理的两个阶段

```
Prompt: "The capital of France is"

═══ Prefill 阶段 (Compute-bound) ═══
  处理整个 prompt → 并行计算所有 token 的 KV
  类似训练的 forward pass → GEMM 密集 → GPU 利用率高

═══ Decode 阶段 (Memory-bound) ═══
  逐 token 生成:
    "Paris" → "." → " It" → " is" → ...
  每次只生成 1 个 token → Batch=1 的 GEMM → 带宽瓶颈
  每次都要读整个模型权重 → 算力利用率极低 (~1-5%)
```

这个 **Prefill-Decode 的二阶段特性** 是所有 LLM 推理优化的出发点。

## 核心挑战

```
7B 模型推理:
  模型权重 (FP16): 14 GB
  生成 1 个 token: 读 14 GB 权重 + 少量计算
  
  H100 带宽: 3.35 TB/s
  → 理论最快: 14 GB / 3.35 TB/s = 4.2 ms/token = ~240 tokens/s (单请求)
  
  H100 算力: 1979 TFLOPS
  → 计算时间: ~0.04 ms (远小于读权重时间)
  
  → Decode 阶段是严重的 Memory-bound!
  → 优化方向: 减少权重大小（量化）、减少读取次数（batching）、减少 KV 缓存（PagedAttention）
```

## 学完本章你能回答

1. KV-Cache 的显存占用公式是什么？7B 模型、4K 上下文需要多少 KV-Cache？
2. INT4 量化能带来多少推理加速？精度损失多大？
3. Continuous Batching 比 Static Batching 好在哪？
4. Speculative Decoding 为什么能保证和原模型输出分布完全一致？
5. FlashAttention 的核心 trick 是什么？为什么能同时省显存和提速？
6. vLLM 的 PagedAttention 解决了什么问题？
