# vLLM 架构

> PagedAttention + Continuous Batching + 高效调度——目前最流行的开源 LLM 推理引擎。

## 核心概念

### PagedAttention — vLLM 的核心创新

借鉴 OS 虚拟内存的分页机制管理 KV-Cache：

```
传统方式: 为每个请求连续分配 max_seq_len 的 KV-Cache
  Request A (500/4096 tokens): [████████████████░░░░░░░░░░░░░░░░] 88% 浪费
  Request B (200/4096 tokens): [████████░░░░░░░░░░░░░░░░░░░░░░░░] 95% 浪费
  显存碎片严重，能服务的并发请求少

PagedAttention: KV-Cache 以固定大小的 Block (Page) 分配
  Block Size = 16 tokens

  Request A (500 tokens): Page[0] → Page[1] → ... → Page[31]  (32 pages)
  Request B (200 tokens): Page[0] → Page[1] → ... → Page[12]  (13 pages)
  
  物理显存中:
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬──┐
  │A-0 │A-1 │B-0 │A-2 │B-1 │A-3 │B-2 │... │Free│Free│  │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴──┘
  
  页表 (Block Table):
  Request A: [物理块0, 物理块1, 物理块3, 物理块5, ...]
  Request B: [物理块2, 物理块4, 物理块6, ...]
  
  → 不需要连续内存！按需分配，大幅减少碎片
  → KV-Cache 显存利用率显著提高（接近实际使用量，而非预分配上限）
```

### PagedAttention 内核

```
标准 Attention: Q × K^T → softmax → × V
  K, V 在显存中连续 → 标准 GEMM

PagedAttention: Q × K_paged^T → softmax → × V_paged  
  K, V 分散在不同的物理 Block 中
  → 需要特殊的 Attention Kernel（按 Block 读取 KV）

伪代码:
  for each block in block_table[request_id]:
    k_block = physical_blocks[block].k  # 读 16 个 token 的 K
    v_block = physical_blocks[block].v  # 读 16 个 token 的 V
    score = Q @ k_block.T
    partial_output += softmax(score) @ v_block
  → 本质是 FlashAttention 的变体，tiling 粒度 = page size
```

### vLLM 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      vLLM Engine                         │
│                                                          │
│  ┌──────────────┐    ┌─────────────────────────────┐    │
│  │   Scheduler   │───→│      Model Executor          │    │
│  │ (请求调度,     │    │  (模型推理, PagedAttention    │    │
│  │  KV-Cache     │    │   kernel, Tensor Parallel)   │    │
│  │  管理)        │    │                               │    │
│  └──────┬───────┘    └─────────────────────────────┘    │
│         │                                                │
│  ┌──────▼───────┐    ┌─────────────────────────────┐    │
│  │ Block Manager │    │     Block Table               │    │
│  │ (物理块分配    │    │  Request → [Physical Block]   │    │
│  │  回收/共享)   │    │                               │    │
│  └──────────────┘    └─────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              API Server (FastAPI/gRPC)             │   │
│  │       OpenAI-compatible API + Streaming            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 使用方式

```python
# 启动 vLLM serving
# vllm serve meta-llama/Llama-2-7b-hf \
#     --tensor-parallel-size 2 \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.9

# Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

outputs = llm.generate(["The capital of France is"], sampling_params)
print(outputs[0].outputs[0].text)
```

## 关键细节

### Prefix Caching

多个请求共享相同的 prompt 前缀时，可以复用 KV-Cache：

```
Request A: "You are a helpful assistant. User: What is AI?"
Request B: "You are a helpful assistant. User: What is ML?"

共享前缀: "You are a helpful assistant. User: What is "
→ 共享前缀的 KV-Cache Block 可以被复用（引用计数管理）
→ 减少重复计算和显存占用

使用: vllm serve ... --enable-prefix-caching
```

### Preemption（抢占）

```
当显存不足以容纳所有请求的 KV-Cache:

策略 1: Swap (默认)
  将低优先级请求的 KV-Cache 从 GPU 换到 CPU
  GPU 显存释放 → 处理高优先级请求
  之后再换回来继续生成

策略 2: Recomputation
  丢弃低优先级请求的 KV-Cache
  之后重新 prefill（重计算代替 swap 的 PCIe 带宽消耗）
```

### 关键配置参数

```bash
vllm serve <model> \
    # 并行
    --tensor-parallel-size 8 \         # TP degree
    --pipeline-parallel-size 2 \       # PP degree
    
    # 显存
    --gpu-memory-utilization 0.9 \     # GPU 显存使用比例
    --max-model-len 32768 \            # 最大序列长度
    --block-size 16 \                  # PagedAttention block size
    
    # 调度
    --max-num-seqs 256 \               # 最大并发请求数
    --max-num-batched-tokens 8192 \    # 每次迭代最大 token 数
    
    # 量化
    --quantization awq \               # 量化方式
    --dtype float16 \                  # 数据类型
    
    # 优化
    --enable-prefix-caching \          # 前缀缓存
    --use-v2-block-manager \           # 新版 block manager
    --enable-chunked-prefill            # 分块 prefill
```

### 性能指标

```
关键指标:
  TTFT (Time to First Token):  首 token 延迟（用户感知的响应速度）
  TPOT (Time per Output Token): 每 token 生成时间（生成速度）
  Throughput (tokens/s):        系统总吞吐量
  
vLLM vs HuggingFace 原生推理 (LLaMA 7B, A100):
  Throughput: ~14-24x 提升 (取决于并发)
  TTFT:       略有增加 (调度开销)
  TPOT:       相近 (单请求) → 大量改善 (高并发时)
```

## 常见问题

**Q: vLLM 和 TensorRT-LLM 该怎么选？**

| 方面 | vLLM | TensorRT-LLM |
|------|------|-------------|
| 易用性 | ✅ pip install，立即可用 | 需要编译模型（build engine） |
| 模型支持 | 广（快速跟进新模型） | 需要模型转换 |
| 性能 | 优秀 | 通常更好（kernel 优化更深） |
| 灵活性 | ✅ 开源，易于修改 | 部分闭源 |
| 量化 | AWQ/GPTQ/FP8 | FP8/INT4/INT8（更深度优化） |

**Q: PagedAttention 的 Block Size 选多大？**

A: 默认 16 tokens。更大的 block → 更少的 page table 开销但更多显存浪费（最后一个 block 可能没填满）。更小的 block → 更细粒度的内存管理但 page table 更大。16 是一个不错的平衡。

**Q: vLLM 支持多模态模型吗？**

A: 支持。vLLM 0.4+ 支持 LLaVA 等视觉语言模型，处理图片输入的 embedding。

## 延伸阅读

- [vLLM 论文: Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM 文档](https://docs.vllm.ai/)
